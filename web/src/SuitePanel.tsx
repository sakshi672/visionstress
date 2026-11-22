import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';

type Case = { image_id: number; score: number };
type Job = { id: number; status: string; spec: { stress: string; severity: number }; result?: { failure_examples: Case[] } | null };
type Model = { id: number; name: string };
type Comparison = { status: string; metric: string; reference: number | null; candidate: number | null; absolute_change: number | null; max_absolute_drop: number; image_count: number };
type Suite = { id: number; name: string; snapshot: { image_ids: number[]; stress: string; severity: number; gate: { metric: string; max_absolute_drop: number }; reference: { stressed: Record<string, number | null> } }; runs: { id: number; job_id: number; status: string; comparison: Comparison | null }[] };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { ...options, headers: { 'Content-Type': 'application/json' } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function value(number: number | null) { return number == null ? 'undefined' : number.toFixed(4); }

export default function SuitePanel({ projectId, sourceJob, models, jobs, onQueued }: { projectId: number | null; sourceJob: Job | null; models: Model[]; jobs: Job[]; onQueued: () => Promise<void> }) {
  const [suites, setSuites] = useState<Suite[]>([]);
  const [name, setName] = useState('');
  const [selected, setSelected] = useState<number[]>([]);
  const [metric, setMetric] = useState('map50_95');
  const [maxDrop, setMaxDrop] = useState('0.02');
  const [modelId, setModelId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const cases = sourceJob?.status === 'completed' ? sourceJob.result?.failure_examples || [] : [];

  useEffect(() => { setSelected(cases.filter(item => item.score > 0).slice(0, 5).map(item => item.image_id)); }, [sourceJob?.id, sourceJob?.status]);
  useEffect(() => { setModelId(current => models.some(model => String(model.id) === current) ? current : String(models[0]?.id || '')); }, [models]);
  useEffect(() => {
    if (!projectId) { setSuites([]); return; }
    request<Suite[]>(`/suites?project_id=${projectId}`).then(setSuites).catch(e => setError(e.message));
  }, [projectId, jobs]);

  async function create(event: FormEvent) {
    event.preventDefault();
    if (!projectId || !sourceJob) return;
    setBusy(true); setError('');
    try {
      await request('/suites', { method: 'POST', body: JSON.stringify({ project_id: projectId, source_job_id: sourceJob.id, name, image_ids: selected, metric, max_absolute_drop: Number(maxDrop) }) });
      setName('');
      setSuites(await request<Suite[]>(`/suites?project_id=${projectId}`));
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not save suite'); }
    finally { setBusy(false); }
  }

  async function run(suiteId: number) {
    setBusy(true); setError('');
    try {
      await request(`/suites/${suiteId}/runs`, { method: 'POST', body: JSON.stringify({ model_id: Number(modelId) }) });
      await onQueued();
      setSuites(await request<Suite[]>(`/suites?project_id=${projectId}`));
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not queue suite'); }
    finally { setBusy(false); }
  }

  return <section id="suites" className="section"><div className="section-heading"><div><span className="eyebrow">REGRESSION</span><h2>Failure-case suites</h2></div></div>
    {error && <p role="alert" className="error-text">{error}</p>}
    <div className="two-col"><div className="panel"><h3>Save selected cases</h3><p>Select a completed evaluation in Results, then choose cases to preserve. The reference metric is recalculated on exactly these images.</p>
      {cases.length ? <form onSubmit={create}><label htmlFor="suite-name">Suite name</label><input id="suite-name" value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Low-light misses" required/>
        <div className="case-list" role="group" aria-label="Failure cases">{cases.map(item => <label key={item.image_id}><input type="checkbox" checked={selected.includes(item.image_id)} onChange={e => setSelected(current => e.target.checked ? [...current, item.image_id] : current.filter(id => id !== item.image_id))}/> Image {item.image_id} <small>impact {item.score}</small></label>)}</div>
        <div className="suite-gate-fields"><div><label htmlFor="gate-metric">Gate metric</label><select id="gate-metric" value={metric} onChange={e => setMetric(e.target.value)}><option value="map50_95">mAP@0.5:0.95</option><option value="map50">mAP@0.5</option><option value="precision">Precision</option><option value="recall">Recall</option></select></div><div><label htmlFor="gate-drop">Maximum absolute drop</label><input id="gate-drop" type="number" min="0" max="1" step="0.001" value={maxDrop} onChange={e => setMaxDrop(e.target.value)} required/></div></div>
        <button disabled={busy || selected.length === 0}>Save suite</button></form> : <p>Select a completed run to choose cases.</p>}
    </div><div className="panel"><h3>Rerun and compare</h3><p>Each suite keeps the dataset hash, image IDs, stress, severity, seed, evaluation settings, source model, and measured reference. Choose an approved model for a new run.</p>
      <label htmlFor="suite-model">Candidate model</label><select id="suite-model" value={modelId} onChange={e => setModelId(e.target.value)}><option value="" disabled>Select model</option>{models.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}</select>
      <div className="suite-list">{suites.map(suite => <div className="suite-card" key={suite.id}><div className="panel-head"><strong>{suite.name}</strong><span className="tag">{suite.snapshot.image_ids.length} images</span></div><p>{suite.snapshot.stress} severity {suite.snapshot.severity} · reference {suite.snapshot.gate.metric}: {value(suite.snapshot.reference.stressed[suite.snapshot.gate.metric])} · allowed drop {suite.snapshot.gate.max_absolute_drop.toFixed(3)}</p><button className="secondary" disabled={busy || !modelId} onClick={() => run(suite.id)}>Run on candidate</button>{suite.runs.map(run => <div className="suite-run" key={run.id}><span>Run #{run.job_id}</span> <span className={`status ${run.comparison?.status || run.status}`}>{run.comparison?.status || run.status}</span>{run.comparison && <p>{value(run.comparison.reference)} to {value(run.comparison.candidate)} (change {value(run.comparison.absolute_change)})</p>}</div>)}</div>)}{!suites.length && <p>No suites saved yet.</p>}</div>
    </div></div><p className="form-footnote">A gate is inconclusive when its metric is undefined or the source uses exploratory occlusion. Small samples are diagnostic.</p>
  </section>;
}
