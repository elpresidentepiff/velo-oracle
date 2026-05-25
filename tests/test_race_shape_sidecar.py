from datetime import date

from src.race_shape.form_history_parser import parse_rp_profile_html
from src.race_shape.ride_profile import JockeyRideProfileBuilder


# ── Synthetic HTML (fallback path) ───────────────────────────────────────────

HTML = """
<table>
  <tr>
    <td>2026-05-01</td><td>YORK</td><td>Class 4 Handicap</td>
    <td>6f</td><td>Good</td><td>6 / 12 btn 4¼L</td><td>9/4</td><td>Jockey: A Rider</td>
  </tr>
  <tr>
    <td>2026-05-08</td><td>NEWMARKET</td><td>Novice Stakes</td>
    <td>7f</td><td>Good To Soft</td><td>1 / 9 by ½L</td><td>11/2</td><td>Jockey: A Rider</td>
  </tr>
  <tr>
    <td>2026-05-15</td><td>ASCOT</td><td>Class 3 Handicap</td>
    <td>1m</td><td>Soft</td><td>8 / 10 btn 6L</td><td>2/1</td><td>Jockey: A Rider</td>
  </tr>
</table>
"""


def test_parse_rp_profile_html_extracts_core_run_fields():
    runs = parse_rp_profile_html(HTML, horse="Test Horse")

    assert len(runs) == 3
    assert runs[0].horse == "Test Horse"
    assert runs[0].race_date == date(2026, 5, 1)
    assert runs[0].course == "York"
    assert runs[0].finishing_position == 6
    assert runs[0].field_size == 12
    assert runs[0].sp_decimal == 3.25
    assert runs[0].jockey == "A Rider"
    assert runs[0].was_well_fancied is True


def test_jockey_profile_marks_well_fancied_execution_variance():
    runs = parse_rp_profile_html(HTML, horse="Test Horse")
    profiles = JockeyRideProfileBuilder().build(runs, min_rides=1)

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.jockey == "A Rider"
    assert profile.rides == 3
    assert profile.well_fancied_rides == 2
    assert profile.well_fancied_failures == 2
    assert profile.anomaly_rate == 1.0
    assert {anomaly.severity for anomaly in profile.anomalies} == {"MEDIUM", "HIGH"}


# ── Real RP fixture (hp-formTable structure) ─────────────────────────────────
# Three rows from Fircombe Hall's profile page.
# Row 0: PLACED  23Apr26  Southwell AW  2/6  SP 3/1  Hollie Doyle (uid 92695)
# Row 1: WIN     30Mar26  Wolverhampton  1/12  SP 9/1  Natasha Cookson (uid 101585)
# Row 2: LOSS    11Apr26  Southwell AW  6/9  SP 9/2  Hollie Doyle (uid 92695)

RP_FIXTURE_HTML = """
<html><body>
<table class="hp-formTable">
  <!-- Row 0: PLACED 23Apr26, Southwell AW, 2/6, SP 3/1, Hollie Doyle -->
  <tr class="ui-table__row">
    <td><div class="hp-formTable__dateWrapper">
      <a href="/results/394/southwell-aw/2026-04-23/918292">23Apr26</a>
    </div></td>
    <td><a href="/profile/course/394/southwell-aw">Southwell (AW) Sth</a></td>
    <td>6f</td>
    <td>St</td>
    <td>8-12 p</td>
    <td>2 / 6 btn 2L Raft Up 9-2</td>
    <td>3/1</td>
    <td><a href="/profile/jockey/92695/hollie-doyle">Hollie Doyle</a></td>
    <td>54</td><td>37</td><td>57</td><td>&#8212;</td>
  </tr>
  <!-- Row 1: WIN 30Mar26, Wolverhampton, 1/12, SP 9/1, Natasha Cookson (+ 5 claim outside link) -->
  <tr class="ui-table__row">
    <td><div class="hp-formTable__dateWrapper">
      <a href="/results/1046/wolverhampton-aw/2026-03-30/916408">30Mar26</a>
    </div></td>
    <td><a href="/profile/course/1046/wolverhampton-aw">Wolverhampton (AW)</a></td>
    <td>7f 32y</td>
    <td>Sd</td>
    <td>9-0</td>
    <td>1 / 12 by &#190;L Forever Noah 11-1</td>
    <td>9/1</td>
    <td><a href="/profile/jockey/101585/miss-natasha-cookson">Miss Natasha Cookson</a> 5</td>
    <td>62</td><td>45</td><td>55</td><td>&#8212;</td>
  </tr>
  <!-- Row 2: LOSS 11Apr26, Southwell AW, 6/9, SP 9/2, Hollie Doyle -->
  <tr class="ui-table__row">
    <td><div class="hp-formTable__dateWrapper">
      <a href="/results/394/southwell-aw/2026-04-11/917501">11Apr26</a>
    </div></td>
    <td><a href="/profile/course/394/southwell-aw">Southwell (AW) Sth</a></td>
    <td>6f</td>
    <td>St</td>
    <td>8-12</td>
    <td>6 / 9 btn 4&#188;L Piperstown 9-8</td>
    <td>9/2</td>
    <td><a href="/profile/jockey/92695/hollie-doyle">Hollie Doyle</a></td>
    <td>44</td><td>30</td><td>57</td><td>&#8212;</td>
  </tr>
</table>
</body></html>
"""


def test_real_fixture_dates_parse_from_rp_abbr_format():
    """Proof A: non-zero dates from 23Apr26 / 30Mar26 / 11Apr26 format."""
    runs = parse_rp_profile_html(RP_FIXTURE_HTML, horse="Fircombe Hall")
    assert len(runs) == 3
    assert runs[0].race_date == date(2026, 4, 23), f"Expected 2026-04-23, got {runs[0].race_date}"
    assert runs[1].race_date == date(2026, 3, 30), f"Expected 2026-03-30, got {runs[1].race_date}"
    assert runs[2].race_date == date(2026, 4, 11), f"Expected 2026-04-11, got {runs[2].race_date}"


def test_real_fixture_jockey_extracted_from_href():
    """Proof B: jockey name comes from the profile/jockey link, not text inference."""
    runs = parse_rp_profile_html(RP_FIXTURE_HTML, horse="Fircombe Hall")
    # Row 0 and 2: Hollie Doyle, uid 92695
    assert runs[0].jockey == "Hollie Doyle", f"Got: {runs[0].jockey}"
    assert runs[2].jockey == "Hollie Doyle", f"Got: {runs[2].jockey}"
    # Row 1: Natasha Cookson — "5" claim marker is outside the link and must not appear
    assert "Cookson" in runs[1].jockey, f"Got: {runs[1].jockey}"
    assert "5" not in runs[1].jockey, f"Claim marker leaked into name: {runs[1].jockey}"


def test_real_fixture_sp_no_position_collision():
    """Proof C: SP=4.0 (3/1) not 3.0 (from position 2/6). No collision with position cell."""
    runs = parse_rp_profile_html(RP_FIXTURE_HTML, horse="Fircombe Hall")
    # Row 0: SP cell is "3/1" = 4.0 decimal; position cell is "2 / 6 btn ..."
    assert runs[0].sp_decimal == 4.0, f"Expected 4.0 (3/1), got {runs[0].sp_decimal}"
    # Row 2: SP cell is "9/2" = 5.5 decimal; position is "6 / 9 ..."
    assert runs[2].sp_decimal == 5.5, f"Expected 5.5 (9/2), got {runs[2].sp_decimal}"
    # Row 1: SP cell is "9/1" = 10.0 decimal
    assert runs[1].sp_decimal == 10.0, f"Expected 10.0 (9/1), got {runs[1].sp_decimal}"


def test_real_fixture_result_types():
    """Row 0=PLACED, Row 1=WIN, Row 2=LOSS."""
    runs = parse_rp_profile_html(RP_FIXTURE_HTML, horse="Fircombe Hall")
    assert runs[0].finishing_position == 2
    assert runs[0].field_size == 6
    assert runs[1].finishing_position == 1
    assert runs[1].field_size == 12
    assert runs[2].finishing_position == 6
    assert runs[2].field_size == 9


def test_no_scoring_imports():
    """Proof F: parser module must not import live scoring code."""
    import importlib, sys
    forbidden = {
        "src.velo", "src.intelligence.sqpe", "app.engine",
        "app.playbooks", "scripts.run_prime_today",
    }
    import src.race_shape.form_history_parser  # already loaded
    loaded = set(sys.modules.keys())
    for mod in forbidden:
        assert mod not in loaded, f"Forbidden import found: {mod}"
