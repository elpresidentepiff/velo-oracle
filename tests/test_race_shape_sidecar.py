from datetime import date

from src.race_shape.form_history_parser import parse_rp_profile_html
from src.race_shape.ride_profile import JockeyRideProfileBuilder


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
