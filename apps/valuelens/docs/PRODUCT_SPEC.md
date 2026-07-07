# ValueLens Product Specification

## Mission

ValueLens helps teams decide which AI project deserves funding, how risky it is, what should be built first, and how success will be proven after launch.

## Positioning

ValueLens is not another chatbot. It is an AI value-governance layer.

## Core user journeys

### 1. Founder / operator

- Enters an AI idea or business bottleneck.
- Receives forecast value, risk, confidence, payback, and a first-build recommendation.
- Leaves with a board-style memo and 30 / 60 / 90 day roadmap.

### 2. Consultant / agency

- Uses ValueLens in discovery calls.
- Converts messy client problems into scoped AI projects.
- Produces a value case before quoting build work.

### 3. Product / engineering team

- Pastes a README, dependency file, or repo structure.
- Receives CodeLens findings: stack signals, test gaps, deployment risk, AI integration points.
- Keeps all recommendations in shadow mode until approved.

### 4. Charity / social impact team

- Uses simplified mode to estimate time saved, people helped, and funding narrative.
- Gets funder-ready language and evidence gaps.

## MVP feature list

### Required for hackathon demo

- Static landing page
- Mode selector: IdeaLens, ProcessLens, CodeLens
- Value input form
- ROI forecast
- Risk score
- Confidence score
- Payback estimate
- Recommended decision
- 30 / 60 / 90 roadmap
- Outcome ledger placeholder
- Submission packet

### Next build after MVP

- Save reports to Supabase
- Export board memo as PDF
- GitHub repo ingestion
- Auth/login
- Team workspace
- Outcome review reminders
- Forecast-vs-actual dashboard
- SilkFlo-style value leaderboard

## Scoring model v0

Value forecast:

annual_waste = hours_wasted_per_week * average_hourly_cost * 52
forecast_annual_value = annual_waste * automation_capture_rate

Risk score uses:

- Build complexity
- Data sensitivity
- Lens mode difficulty
- Evidence quality

Confidence uses:

- Evidence quality
- Risk score penalty
- People affected signal

Decision categories:

- Fund first
- Prototype
- Pilot only
- Hold / clarify evidence

## Governance rules

1. ValueLens never claims actual ROI before measurement.
2. Forecast and actual value are separate fields.
3. High-risk use cases default to human approval.
4. CodeLens does not modify production code.
5. Evidence gaps must be visible, not hidden.
6. Every project must have a review loop.

## Demo promise

In under one minute, ValueLens turns a messy AI idea into a fund / pilot / hold decision with numbers, risks, and a proof plan.
