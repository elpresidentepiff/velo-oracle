-- ValueLens MVP schema
-- Forecast -> Actual -> Correction loop

CREATE TABLE IF NOT EXISTS value_cases (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_name TEXT NOT NULL,
  lens_mode TEXT NOT NULL CHECK (lens_mode IN ('idea', 'process', 'code')),
  problem_statement TEXT NOT NULL,
  people_affected INTEGER DEFAULT 1,
  hours_wasted_per_week NUMERIC DEFAULT 0,
  average_hourly_cost NUMERIC DEFAULT 0,
  build_complexity TEXT NOT NULL,
  data_sensitivity TEXT NOT NULL,
  evidence_quality TEXT NOT NULL,
  forecast_annual_value NUMERIC DEFAULT 0,
  forecast_payback_months NUMERIC DEFAULT 0,
  risk_score NUMERIC DEFAULT 0,
  confidence_score NUMERIC DEFAULT 0,
  recommended_decision TEXT NOT NULL,
  recommended_first_build TEXT,
  assumptions_json JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS value_outcomes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  value_case_id UUID NOT NULL REFERENCES value_cases(id) ON DELETE CASCADE,
  actual_hours_saved_per_month NUMERIC,
  actual_cost_saved NUMERIC,
  actual_revenue_gain NUMERIC,
  risk_incidents INTEGER DEFAULT 0,
  adoption_notes TEXT,
  forecast_vs_actual_delta NUMERIC,
  outcome_status TEXT CHECK (outcome_status IN ('pending', 'positive', 'neutral', 'negative', 'killed')) DEFAULT 'pending',
  lesson_learned TEXT,
  reviewed_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_lens_findings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  value_case_id UUID NOT NULL REFERENCES value_cases(id) ON DELETE CASCADE,
  finding_type TEXT NOT NULL CHECK (finding_type IN ('signal', 'risk', 'opportunity', 'missing_evidence')),
  severity TEXT CHECK (severity IN ('low', 'medium', 'high')) DEFAULT 'medium',
  finding TEXT NOT NULL,
  recommended_action TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
