# ValueLens MVP

ValueLens is a hackathon-ready AI value and governance product. It turns an AI idea, business process, or codebase summary into a board-ready value case, risk audit, implementation roadmap, and outcome ledger.

## Core doctrine

Input -> Value Forecast -> Risk Audit -> Build Plan -> Outcome Ledger -> Correction Loop

Most AI products generate answers. ValueLens records the forecast, checks the result, and learns from the gap.

## MVP modes

1. IdeaLens: analyse a proposed AI project or automation idea.
2. ProcessLens: analyse a messy business process and identify the first automation worth funding.
3. CodeLens: analyse a codebase summary, README, dependency file, or file tree for AI readiness and delivery risk.
4. ValueLedger: store forecast metrics and compare them against actual outcomes after deployment.

## Hackathon pitch

ValueLens is the missing governance layer between AI hype and funded AI execution.

It helps teams answer four questions:

- Is this AI project worth funding?
- What is the risk?
- What should we build first?
- How will we prove it worked?

## Demo path

Open `index.html` in a browser. The MVP is intentionally static and dependency-free so it can be presented quickly, hosted anywhere, or rebuilt inside Bolt.new.

## Product boundary

ValueLens does not modify production code. CodeLens produces a shadow audit first, then recommends safe next actions. High-risk automations default to human approval.
