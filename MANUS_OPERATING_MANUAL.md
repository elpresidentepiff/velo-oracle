Good. Now we’re talking about governance, not just shiny models.

Below is the operating manual for Manus so it doesn’t burn your credits doing interpretive dance in a loop.


---

0. Prime Directive for Manus

> Do not retrain, reinstall, or “refactor” anything unless explicitly instructed.
Default: load existing artefacts, run once, log, exit.



Manus has two modes only:

1. OFFLINE MODE – Training & optimisation


2. RACE MODE – Same-day inference & staking



If Manus mixes those, it’s fired.


---

1. OFFLINE MODE – What Manus is allowed to do

This is where Manus earns its keep. But it runs occasionally (e.g. nightly / weekly), not every time you think the word “VÉLØ”.

1.1 Data pipeline (single pass, no loops)

When you say something like:

> “Manus, refresh the stack on history up to YYYY-MM-DD”



Manus should:

1. Load data once

Historical race-level + runner-level features.

Don’t re-download the same files inside loops.

Cache locally: data/raw/, data/processed/.



2. Build features once per refresh

Base tabular features (already exist in your current pipeline).

Sequence tensors for FormSequenceModel

Shape: [n_runners, seq_len, n_features_seq].


Intent features for IntentModel

Columns like perf_delta, market_delta, trainer/jockey hot-cold scores.


Persist to data/features_v11.parquet (or similar). No recomputing in sub-functions.



3. Train models (ONE PASS EACH)
Under no circumstances should Manus do “auto-search” loops where it retrains the same thing 15 times for fun.

Base models (existing):

GLM / Logistic

Random Forest / GBDT

KNN or NB


New v11 models:

FormSequenceModel (LSTM)

IntentModel (GMM)


SQPEMetaBrain:

Build meta features:

b/t/k probabilities

race & horse cluster IDs

seq_win_proba, seq_place_proba

intent_scalar (and optionally p_on)


Fit once per refresh.




4. Optimise staking – LIMITED runs

GAStakingOptimizer:

Use historical RunnerDecision + results.

Run GA for fixed n_generations (e.g. 40–60).
No “until convergence” nonsense.


RLStakingAgent:

Train in shadow mode on history:

For each historical race:

Use bucket + odds to choose action.

Compute reward from realised P&L.

Call update(...).



Save Q-table once done.




5. Persist artefacts & exit Manus must save:

/models/base/*.pkl

/models/form_sequence.pt

/models/intent_gmm.pkl

/models/sqpe_meta.pkl

/models/ga_theta.json

/models/rl_qtable.npy


Then stop.
No post-analysis chat, no “just checking another metric”.




---

2. RACE MODE – What Manus does on card day

Trigger example:

> “Manus, run race-day stack for [track/date].”



Race mode = read-only models, write-only decisions.
Absolutely no training here.

2.1 Per race workflow

For each race you ask for (e.g. Newcastle 15:18):

1. Ingest racecard data

Use The Racing API / RP scrape output you give it.

Build only today’s features from:

Base tabular features

Sequence slice for each runner (last N runs)

Intent features (using most recent history)




2. Load models (once per process)

Load all artefacts into memory at start:

Base models

FormSequenceModel

IntentModel

SQPEMetaBrain

GAStakingOptimizer.best_theta_

RLStakingAgent Q-table


Do not reload per race.



3. Generate probabilities

Get:

p_win, p_place from base models.

seq_win_proba, seq_place_proba from FormSequenceModel.

intent_scalar (and optionally p_on) from IntentModel.


Build meta features and call SQPEMetaBrain.predict_for_race.



4. Bucket & decision layer

Use your DecisionModel v11:

Classify each runner into:

core_strike

value_place

chaos_longshot

fade


Include structural flags:

race chaos score

market manipulation hint

jockey/trainer intent score



Output one list of RunnerDecision objects.
No second “rethink” pass unless you explicitly ask for it.



5. Staking (credit-safe)

Staking must be deterministic and single-pass:

Get theta = ga_optimizer.best_theta_.

For each RunnerDecision:

Lookup live odds.

Use GA to compute base stake.

Use RLStakingAgent.stake_for_runner(bucket, odds, bank, train=False)
to get an upper cap.

Final stake = min(ga_stake, rl_stake_cap).


Return:

Runner

Bucket

p_win, p_place

Odds used

Stake suggested



No simulation, no Monte-Carlo, no iterative “refine stake” loops.


6. Logging for post-race

Log once:

Input features hash

Model versions

Decisions + stakes

Time stamp


That’s it. No long commentary unless you ask.





---

3. Post-Race Learning – When Manus is allowed back in

After results are in and you say:

> “Log results for [race] and update learning.”



Manus should:

1. Append to results_history.parquet:

Runner ID

Finishing position / win flag / place flag

SP, BFSP, place terms

Bucket used

Stake used, P&L



2. Update RL agent only, short run:

Load RLStakingAgent + Q-table.

For each logged bet:

Compute reward (e.g. P&L / bank_at_race).

Call update(...).


Save Q-table back.




No retraining of meta models, no GA reruns here.
You batch those in OFFLINE MODE later.


---

4. Credit-Protection Rules for Manus

This is the “don’t be an idiot” section.

1. No nested LLM calls inside loops

Feature loops, race loops, sequence loops must be pure Python / sklearn / torch.

LLM use is only for:

Generating human-readable summaries when you ask.

Writing / revising code files when you explicitly instruct.




2. Idempotent operations

Before downloading / recomputing anything, Manus checks:

If data/features_v11.parquet already exists for that date.

If models have a newer timestamp than the data.


If yes, skip.



3. No “self-diagnosis rambles”

Logs go to disk, not to you, unless you ask.

Default race-day output = single concise summary + structured decisions.



4. Strict task boundaries

Each call to Manus must:

Read task description.

Execute once.

Return result.

Stop.


No “let me also quickly explore XYZ” improvisation.



5. Shadow mode defaults

For any new component (e.g. the RL agent initially), Manus should:

Run it in paper-trading mode until you explicitly say “activate live stakes”.

That means: compute stakes, log them, but mark them live=false.






---

5. What Manus MUST NOT do (ever)

Retrain any model just because new data exists, unless you explicitly request a refresh.

Rebuild the entire environment / reinstall dependencies every time.

Re-run hyperparameter searches or GA with different seeds “just to see.”

Call the LLM to “explain” internal arrays or debug messages in 20 different phrasings.

Change staking logic or buckets without surfacing it to you.



---

TL;DR for Manus (since agents are slow readers)

1. Offline:
Load history → build features → train models → run GA → train RL on history → save → exit.


2. Race-day:
Load saved models → build features for today → get probs → bucket runners → compute stakes once (GA + RL cap) → log → exit.


3. Post-race:
Log results → update RL Q-table → save → exit.



If Manus follows this, your credits go into learning & decisions, not into philosophical monologues from a confused agent.

And yes, I’ll be watching Manus too.