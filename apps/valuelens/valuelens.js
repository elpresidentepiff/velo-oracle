const weights = {
  complexity: { low: 12, medium: 28, high: 45 },
  sensitivity: { low: 8, medium: 25, high: 42 },
  evidence: { weak: -18, partial: 0, strong: 16 },
  modeDifficulty: { idea: 10, process: 14, code: 22 }
};

function currency(value) {
  return new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP', maximumFractionDigits: 0 }).format(value);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function classifyRisk(score) {
  if (score >= 72) return { label: 'High', action: 'Hold full automation. Run a controlled pilot with human approval.' };
  if (score >= 42) return { label: 'Medium', action: 'Pilot with guardrails, evidence capture, and human approval for edge cases.' };
  return { label: 'Low', action: 'Safe candidate for rapid prototype and limited rollout.' };
}

function classifyDecision(value, risk, confidence) {
  if (risk >= 72) return 'Pilot only';
  if (value >= 50000 && confidence >= 62) return 'Fund first';
  if (value >= 20000) return 'Prototype';
  return 'Hold / clarify evidence';
}

function detectCodeSignals(text) {
  const lower = text.toLowerCase();
  const signals = [];
  const risks = [];

  if (lower.includes('package.json') || lower.includes('react') || lower.includes('next') || lower.includes('vite')) signals.push('Frontend application detected. Good candidate for AI-assisted workflow UI.');
  if (lower.includes('requirements.txt') || lower.includes('fastapi') || lower.includes('flask') || lower.includes('python')) signals.push('Python/API layer detected. Good candidate for backend AI orchestration.');
  if (lower.includes('supabase') || lower.includes('postgres') || lower.includes('sql')) signals.push('Database layer detected. Value ledger can be implemented cleanly.');
  if (!lower.includes('test') && !lower.includes('pytest') && !lower.includes('spec')) risks.push('No obvious test layer in submitted summary. Treat delivery risk as elevated.');
  if (lower.includes('secret') || lower.includes('api_key') || lower.includes('password')) risks.push('Possible secret handling risk. Rotate exposed keys and move credentials to environment variables.');
  if (lower.includes('todo') || lower.includes('wip')) risks.push('Incomplete implementation markers detected. Keep first release narrow.');

  return { signals, risks };
}

function analyse(payload) {
  const weeklyCost = payload.hours * payload.hourlyCost;
  const annualWaste = weeklyCost * 52;
  const automationCapture = payload.mode === 'code' ? 0.28 : payload.mode === 'process' ? 0.36 : 0.32;
  const forecastAnnualValue = annualWaste * automationCapture;

  const riskScore = clamp(
    weights.complexity[payload.complexity] + weights.sensitivity[payload.sensitivity] + weights.modeDifficulty[payload.mode] + (payload.evidence === 'weak' ? 12 : 0),
    0,
    100
  );

  const confidence = clamp(58 + weights.evidence[payload.evidence] - riskScore * 0.12 + Math.min(payload.people, 20) * 0.6, 15, 92);
  const paybackMonths = clamp((riskScore / 18) + (payload.complexity === 'high' ? 3.5 : payload.complexity === 'medium' ? 2.2 : 1.2), 1, 12);
  const risk = classifyRisk(riskScore);
  const decision = classifyDecision(forecastAnnualValue, riskScore, confidence);
  const code = payload.mode === 'code' ? detectCodeSignals(payload.problem) : { signals: [], risks: [] };

  const firstBuild = payload.mode === 'code'
    ? 'Shadow CodeLens scan with architecture summary, test-risk check, and one safe AI integration recommendation.'
    : payload.sensitivity === 'high'
      ? 'Human-approved assistant that drafts, classifies, and recommends but does not make final decisions.'
      : 'Narrow workflow copilot focused on the highest-volume repetitive task.';

  const successMetrics = payload.mode === 'code'
    ? ['Time to understand repo', 'Defects found before build', 'Test coverage uplift', 'Implementation cycle time']
    : ['Hours saved per month', 'Average handling time', 'Error reduction', 'User adoption', 'Escalation rate'];

  return {
    forecastAnnualValue,
    riskScore,
    risk,
    confidence,
    paybackMonths,
    decision,
    firstBuild,
    successMetrics,
    code
  };
}

function buildReport(payload, result) {
  const codeSection = payload.mode === 'code'
    ? `<div class="section"><h3>CodeLens signals</h3><ul>${[...result.code.signals, ...result.code.risks].map(item => `<li>${item}</li>`).join('') || '<li>No specific code signals detected from the supplied text.</li>'}</ul></div>`
    : '';

  return `
    <div class="report-header">
      <p class="eyebrow">ValueLens verdict</p>
      <h2>${payload.projectName}</h2>
      <span class="badge">${result.decision}</span>
    </div>

    <div class="cards">
      <div class="card"><span>Forecast annual value</span><strong>${currency(result.forecastAnnualValue)}</strong></div>
      <div class="card"><span>Risk level</span><strong>${result.risk.label}</strong><small>${result.riskScore}/100</small></div>
      <div class="card"><span>Confidence</span><strong>${Math.round(result.confidence)}%</strong></div>
      <div class="card"><span>Payback estimate</span><strong>${result.paybackMonths.toFixed(1)} months</strong></div>
    </div>

    <div class="section"><h3>Recommended first build</h3><p>${result.firstBuild}</p></div>
    <div class="section"><h3>Risk posture</h3><p>${result.risk.action}</p></div>
    ${codeSection}
    <div class="section"><h3>30 / 60 / 90 roadmap</h3><ol><li><strong>30 days:</strong> prototype the narrow workflow and capture baseline metrics.</li><li><strong>60 days:</strong> pilot with real users, human approval, and incident logging.</li><li><strong>90 days:</strong> compare forecast against actual value and decide scale, hold, or kill.</li></ol></div>
    <div class="section"><h3>Success metrics</h3><ul>${result.successMetrics.map(metric => `<li>${metric}</li>`).join('')}</ul></div>
    <div class="ledger"><h3>Outcome ledger</h3><p><strong>Forecast:</strong> ${currency(result.forecastAnnualValue)} annual value.</p><p><strong>Actual:</strong> Pending.</p><p><strong>Status:</strong> Not proven yet.</p><p><strong>Next loop:</strong> Review after pilot and classify the gap between forecast and reality.</p></div>
  `;
}

function collectPayload() {
  return {
    mode: document.getElementById('mode').value,
    projectName: document.getElementById('projectName').value.trim() || 'Untitled project',
    problem: document.getElementById('problem').value.trim(),
    people: Number(document.getElementById('people').value || 1),
    hours: Number(document.getElementById('hours').value || 0),
    hourlyCost: Number(document.getElementById('hourlyCost').value || 0),
    complexity: document.getElementById('complexity').value,
    sensitivity: document.getElementById('sensitivity').value,
    evidence: document.getElementById('evidence').value
  };
}

function render() {
  const payload = collectPayload();
  const result = analyse(payload);
  document.getElementById('report').innerHTML = buildReport(payload, result);
}

document.getElementById('lens-form').addEventListener('submit', event => {
  event.preventDefault();
  render();
});

render();
