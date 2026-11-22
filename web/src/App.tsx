import { useCallback, useEffect, useState } from 'react';
import SuitePanel from './SuitePanel';

type Project = { id: number; name: string; description: string };
type Dataset = { id: number; name: string; digest: string; manifest: { image_count: number; box_count: number; classes: string[]; class_distribution: Record<string, number>; images: {id: number; file_name: string}[]; split: {validation: number[]; test: number[]} } };
type Model = { id: number; name: string; digest: string; classes: string[] };
type Metrics = { map50: number | null; map50_95: number | null; precision: number | null; recall: number | null; average_inference_ms: number; image_count: number };
type Failure = { image_id: number; score: number; baseline: {false_negatives: number; false_positives: number}; stressed: {false_negatives: number; false_positives: number}; original_artifact?: string; stressed_artifact?: string };
type Job = { id: number; status: string; progress: number; error: string | null; spec: { stress: string; severity: number; seed: number }; result?: { baseline: Metrics; stressed: Metrics; degradation: Record<string, {absolute: number | null}>; image_count: number; warnings: string[]; failure_examples: Failure[] } | null };

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options?.headers } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? undefined as T : response.json();
}

function Metric({label, value, difference}: {label: string; value: number | null; difference?: number | null}) {
  return <div className="metric"><span>{label}</span><strong>{value == null ? '—' : value.toFixed(3)}</strong>{difference != null && <small className={difference < 0 ? 'negative' : 'positive'}>{difference > 0 ? '+' : ''}{difference.toFixed(3)} vs baseline</small>}</div>;
}

function FailureGallery({jobId, examples}: {jobId: number; examples: Failure[]}) {
  const available = examples.filter(item => item.original_artifact && item.stressed_artifact);
  if (!available.length) return null;
  return <div className="failure-gallery"><h4>Failure examples</h4><p>Green boxes are ground truth; red boxes are predictions at confidence ≥ 0.25. Ranked by new misses and false detections. This simple matcher is separate from COCOeval.</p>{available.map(item => <div className="failure-example" key={item.image_id}><div className="failure-caption"><strong>Image {item.image_id}</strong><span>Impact score {item.score} · stressed FN {item.stressed.false_negatives} · FP {item.stressed.false_positives}</span></div><div className="image-pair"><div><span>Original</span><img src={`/api/jobs/${jobId}/artifacts/${item.original_artifact}`} alt={`Original image ${item.image_id} with ground truth and baseline predictions`}/></div><div><span>Stressed</span><img src={`/api/jobs/${jobId}/artifacts/${item.stressed_artifact}`} alt={`Stressed image ${item.image_id} with ground truth and predictions`}/></div></div></div>)}</div>;
}

function DatasetCard({dataset}: {dataset: Dataset}) {
  const topClasses = Object.entries(dataset.manifest.class_distribution).sort((a, b) => b[1] - a[1]).slice(0, 5);
  return <div className="dataset-card"><strong>{dataset.name}</strong><span>{dataset.manifest.image_count} images · {dataset.manifest.box_count} boxes · {dataset.manifest.split.validation.length} validation / {dataset.manifest.split.test.length} test</span><span>Top classes: {topClasses.map(([name, count]) => `${name} ${count}`).join(' · ') || 'No boxes'}</span><div className="dataset-samples">{dataset.manifest.images.slice(0, 3).map(image => <img key={image.id} src={`/api/datasets/${dataset.id}/samples/${image.id}`} alt={`Dataset sample ${image.file_name}`}/>)}</div></div>;
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [projectName, setProjectName] = useState('');
  const [datasetName, setDatasetName] = useState('Demo COCO sample');
  const [datasetDirectory, setDatasetDirectory] = useState('demo_coco8');
  const [checkpoint, setCheckpoint] = useState('yolo11m.pt');
  const [datasetId, setDatasetId] = useState('');
  const [modelId, setModelId] = useState('');
  const [stress, setStress] = useState('brightness');
  const [severity, setSeverity] = useState(2);

  const refreshProjects = useCallback(async () => { const value = await api<Project[]>('/projects'); setProjects(value); setProjectId(current => current ?? value[0]?.id ?? null); }, []);
  const refresh = useCallback(async () => {
    if (!projectId) return;
    const [d, m, j] = await Promise.all([api<Dataset[]>(`/datasets?project_id=${projectId}`), api<Model[]>(`/models?project_id=${projectId}`), api<Job[]>(`/jobs?project_id=${projectId}`)]);
    setDatasets(d); setModels(m); setJobs(j);
    setDatasetId(current => current || String(d[0]?.id || ''));
    setModelId(current => current || String(m[0]?.id || ''));
    if (j.length) setSelectedJob(await api<Job>(`/jobs/${j[0].id}`));
  }, [projectId]);
  useEffect(() => { refreshProjects().catch(e => setError(e.message)); }, [refreshProjects]);
  useEffect(() => { refresh().catch(e => setError(e.message)); }, [refresh]);
  useEffect(() => { if (!jobs.some(j => j.status === 'queued' || j.status === 'running')) return; const timer = setInterval(() => refresh().catch(e => setError(e.message)), 2500); return () => clearInterval(timer); }, [jobs, refresh]);

  async function act(task: () => Promise<void>) { setBusy(true); setError(''); try { await task(); } catch (e) { setError(e instanceof Error ? e.message : 'Unknown error'); } finally { setBusy(false); } }
  const selectedDataset = datasets.find(d => d.id === Number(datasetId));
  const latest = selectedJob?.result;
  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><span className="brand-mark">V</span><div><strong>VisionStress</strong><small>Robustness workspace</small></div></div><nav aria-label="Main navigation"><a href="#overview">Overview</a><a href="#assets">Datasets & models</a><a href="#evaluation">New evaluation</a><a href="#results">Results</a><a href="#suites">Regression suites</a></nav><div className="sidebar-note">Object detection MVP<br/><span>Synthetic stress is diagnostic, not safety certification.</span></div></aside>
    <main><header className="topbar"><div><span className="eyebrow">DEVELOPER WORKSPACE</span><h1>Know where your detector breaks.</h1><p>Run repeatable visual stress tests against a trusted YOLO model and a labeled dataset.</p></div><a className="docs-link" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API documentation</a></header>
      {error && <div role="alert" className="alert">{error}<button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
      <section id="overview" className="section"><div className="section-heading"><div><span className="eyebrow">WORKSPACE</span><h2>Project overview</h2></div><button className="text-button" onClick={() => act(refresh)}>Refresh data</button></div>
        <div className="overview-grid"><div className="panel project-panel"><label htmlFor="project">Active project</label><select id="project" value={projectId ?? ''} onChange={e => { setProjectId(Number(e.target.value)); setDatasetId(''); setModelId(''); setSelectedJob(null); }}><option value="" disabled>Select project</option>{projects.map(p => <option value={p.id} key={p.id}>{p.name}</option>)}</select><form onSubmit={e => {e.preventDefault(); act(async () => { const p = await api<Project>('/projects', {method:'POST', body: JSON.stringify({name:projectName, description:''})}); setProjectName(''); await refreshProjects(); setProjectId(p.id); });}}><label htmlFor="project-name">Create project</label><div className="inline-form"><input id="project-name" value={projectName} onChange={e => setProjectName(e.target.value)} placeholder="e.g. Drone detector" required/><button disabled={busy}>Create</button></div></form></div><div className="stat-card"><span>Datasets</span><strong>{datasets.length}</strong><small>Validated YOLO sets</small></div><div className="stat-card"><span>Trusted models</span><strong>{models.length}</strong><small>Hash-verified checkpoints</small></div><div className="stat-card"><span>Evaluations</span><strong>{jobs.length}</strong><small>Persistent worker jobs</small></div></div></section>
      <section id="assets" className="section"><div className="section-heading"><div><span className="eyebrow">INPUTS</span><h2>Datasets & models</h2></div></div><div className="two-col"><div className="panel"><div className="panel-head"><h3>YOLO dataset</h3><span className="tag">Local registration</span></div><p>Place images/, labels/, and classes.json in a folder under data/. Registration validates every file.</p><form onSubmit={e => { e.preventDefault(); if (!projectId) return; act(async () => { await api('/datasets', {method:'POST', body:JSON.stringify({project_id:projectId, name:datasetName, directory:datasetDirectory})}); await refresh(); }); }}><label htmlFor="dataset-name">Display name</label><input id="dataset-name" value={datasetName} onChange={e => setDatasetName(e.target.value)} required/><label htmlFor="dataset-dir">Folder under data/</label><input id="dataset-dir" value={datasetDirectory} onChange={e => setDatasetDirectory(e.target.value)} required/><button disabled={busy || !projectId}>Register dataset</button></form><div className="asset-list">{datasets.map(d => <DatasetCard key={d.id} dataset={d}/>)}{!datasets.length && <p>No datasets registered.</p>}</div></div>
        <div className="panel"><div className="panel-head"><h3>Approved model</h3><span className="tag">SHA-256 verified</span></div><p>Only pinned official YOLO11 checkpoints in models/ can be loaded. Model uploads are disabled.</p><form onSubmit={e => {e.preventDefault(); if (!projectId) return; act(async () => {await api('/models', {method:'POST', body:JSON.stringify({project_id:projectId, checkpoint})}); await refresh();});}}><label htmlFor="checkpoint">Checkpoint</label><select id="checkpoint" value={checkpoint} onChange={e => setCheckpoint(e.target.value)}><option>yolo11n.pt</option><option>yolo11s.pt</option><option>yolo11m.pt</option></select><button disabled={busy || !projectId}>Register model</button></form><div className="asset-list">{models.map(m => <div key={m.id}><strong>{m.name}</strong><span>{m.classes.length} classes · {m.digest.slice(0, 12)}…</span></div>)}{!models.length && <p>No trusted models registered.</p>}</div></div></div></section>
      <section id="evaluation" className="section"><div className="section-heading"><div><span className="eyebrow">RUN</span><h2>New evaluation</h2></div></div><div className="panel"><form className="evaluation-form" onSubmit={e => { e.preventDefault(); if (!projectId) return; act(async () => {const j = await api<Job>('/evaluations', {method:'POST', body:JSON.stringify({project_id:projectId, dataset_id:Number(datasetId), model_id:Number(modelId), stress, severity, seed:42, max_images:12})}); await refresh(); setSelectedJob(j); document.getElementById('results')?.scrollIntoView({behavior:'smooth'});}); }}><div><label htmlFor="eval-dataset">Dataset</label><select id="eval-dataset" value={datasetId} onChange={e => setDatasetId(e.target.value)} required><option value="" disabled>Select dataset</option>{datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}</select></div><div><label htmlFor="eval-model">Model</label><select id="eval-model" value={modelId} onChange={e => setModelId(e.target.value)} required><option value="" disabled>Select model</option>{models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select></div><div><label htmlFor="stress">Stress condition</label><select id="stress" value={stress} onChange={e => setStress(e.target.value)}><option value="brightness">Low light</option><option value="brightness_contrast">Brightness & contrast</option><option value="gaussian_blur">Gaussian blur</option><option value="motion_blur">Motion blur</option><option value="gaussian_noise">Gaussian noise</option><option value="jpeg">JPEG compression</option><option value="fog">Fog approximation</option><option value="rain">Rain approximation</option><option value="partial_occlusion">Partial occlusion (exploratory)</option><option value="color_shift">Color shift</option><option value="defocus">Camera defocus</option></select></div><div><label htmlFor="severity">Severity · {severity}</label><input id="severity" type="range" min="1" max="5" value={severity} onChange={e => setSeverity(Number(e.target.value))}/></div><button disabled={busy || !projectId || !datasetId || !modelId}>Queue evaluation</button></form><p className="form-footnote">Baseline and stress use identical inference settings. Up to 12 validation images are selected in stored manifest order. Seed: 42. Occlusion results are exploratory because labels may be hidden.</p>{selectedDataset && <p className="form-footnote">Selected dataset: {selectedDataset.manifest.classes.length} classes. Model class order must match exactly.</p>}</div></section>
      <section id="results" className="section"><div className="section-heading"><div><span className="eyebrow">EVIDENCE</span><h2>Evaluation results</h2></div></div><div className="two-col result-layout"><div className="panel"><h3>Jobs</h3><div className="job-list">{jobs.map(j => <button key={j.id} className={selectedJob?.id === j.id ? 'job active' : 'job'} onClick={() => act(async () => setSelectedJob(await api<Job>(`/jobs/${j.id}`)))}><span><strong>Run #{j.id}</strong><small>{j.spec.stress} · severity {j.spec.severity}</small></span><span className={`status ${j.status}`}>{j.status}</span></button>)}{!jobs.length && <p>No evaluations yet. Register inputs, then queue a run.</p>}</div></div><div className="panel result-panel">{selectedJob ? <><div className="panel-head"><h3>Run #{selectedJob.id}</h3><span className={`status ${selectedJob.status}`}>{selectedJob.status}</span></div>{selectedJob.status === 'running' || selectedJob.status === 'queued' ? <div><div className="progress-track"><div style={{width:`${selectedJob.progress}%`}}/></div><p>{selectedJob.progress}% complete</p><button className="secondary" onClick={() => act(async () => {await api(`/jobs/${selectedJob.id}/cancel`, {method:'POST'}); await refresh();})}>Cancel</button></div> : null}{selectedJob.error && <p role="alert" className="error-text">{selectedJob.error}</p>}{latest && <><p>{latest.image_count} validation images · {selectedJob.spec.stress} severity {selectedJob.spec.severity}</p>{latest.warnings?.map(w => <p role="alert" className="error-text" key={w}>{w}</p>)}<div className="metric-grid"><Metric label="mAP@0.5:0.95" value={latest.stressed.map50_95} difference={latest.degradation.map50_95.absolute}/><Metric label="mAP@0.5" value={latest.stressed.map50} difference={latest.degradation.map50.absolute}/><Metric label="Precision" value={latest.stressed.precision} difference={latest.degradation.precision.absolute}/><Metric label="Recall" value={latest.stressed.recall} difference={latest.degradation.recall.absolute}/></div><p className="latency">Inference latency: {latest.baseline.average_inference_ms.toFixed(1)} ms baseline · {latest.stressed.average_inference_ms.toFixed(1)} ms stressed</p><a href={`/api/jobs/${selectedJob.id}/artifacts/result.json`} target="_blank" rel="noreferrer">Download result JSON</a><FailureGallery jobId={selectedJob.id} examples={latest.failure_examples || []}/></> }</> : <p>Select a job to inspect measured results.</p>}</div></div></section>
      <SuitePanel projectId={projectId} sourceJob={selectedJob} models={models} jobs={jobs} onQueued={refresh}/>
      <footer>VisionStress · Reproducible synthetic robustness testing for object detection.</footer>
    </main>
  </div>;
}
