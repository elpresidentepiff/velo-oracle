# ValueLens Demo Seed Cases

Use these examples during the hackathon demo.

## Case 1: Customer complaint triage

Mode: ProcessLens

Project name: Customer complaint triage assistant

Problem: A small operations team spends too much time reading customer complaints, classifying urgency, drafting replies, and escalating edge cases. Sensitive customer data may be present, so full automation is risky. The first build should keep a human approval checkpoint.

Inputs:
- People affected: 8
- Hours wasted per week: 42
- Average hourly cost: 35
- Complexity: Medium
- Data sensitivity: Medium
- Evidence quality: Partial

Expected story:
This is a strong controlled-pilot candidate. It should not fully automate customer-facing decisions. The first build should draft, classify, and recommend with human approval.

## Case 2: Invoice checking

Mode: IdeaLens

Project name: Invoice validation copilot

Problem: Finance staff manually check supplier invoices against purchase orders. The process is repetitive, slow, and vulnerable to missed errors.

Inputs:
- People affected: 5
- Hours wasted per week: 28
- Average hourly cost: 40
- Complexity: Medium
- Data sensitivity: Medium
- Evidence quality: Strong

Expected story:
Good commercial ROI. Begin with assisted validation and exception flagging, not autonomous payment approval.

## Case 3: Codebase readiness audit

Mode: CodeLens

Project name: Support portal AI readiness scan

Problem: README shows a React frontend, Python API layer, and database-backed support tickets. The team wants to know where AI can safely help without creating delivery risk. There is little visible test coverage in the summary.

Inputs:
- People affected: 6
- Hours wasted per week: 20
- Average hourly cost: 50
- Complexity: High
- Data sensitivity: High
- Evidence quality: Weak

Expected story:
ValueLens should recommend a shadow scan and guarded prototype, not full automation. Risk is high because sensitive data and limited evidence make the project unsafe to rush.
