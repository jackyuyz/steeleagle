import { useEffect, useRef, useState } from 'react';
import { Button } from 'primereact/button';
import './inferredMission.css';

const API = '/api/inferred-mission';
const ACTIVE = new Set(['observing', 'analyzing', 'pausing', 'stopping']);

function label(value) {
    return (value || 'pending').replaceAll('_', ' ');
}

function timestamp(seconds) {
    const absolute = Math.round(Math.abs(seconds || 0) * 100) / 100;
    const minutes = String(Math.floor(absolute / 60)).padStart(2, '0');
    const remainder = absolute % 60;
    const duration = Number.isInteger(remainder)
        ? String(remainder).padStart(2, '0') : remainder.toFixed(2).padStart(5, '0');
    return `${seconds < 0 ? '−' : ''}${minutes}:${duration}`;
}

async function jsonRequest(path, options) {
    const response = await fetch(path, options);
    const body = await response.json();
    if (!response.ok) {
        throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}).`);
    }
    return body;
}

function uploadVideo(file, onProgress) {
    return new Promise((resolve, reject) => {
        const form = new FormData();
        form.append('file', file);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API}/videos`);
        xhr.upload.onprogress = event => {
            if (event.lengthComputable) onProgress(Math.round(100 * event.loaded / event.total));
        };
        xhr.onerror = () => reject(new Error('The video upload was interrupted.'));
        xhr.onload = () => {
            try {
                const body = JSON.parse(xhr.responseText);
                if (xhr.status >= 200 && xhr.status < 300) resolve(body);
                else reject(new Error(body.detail || 'The video could not be uploaded.'));
            } catch {
                reject(new Error(`The upload returned an unexpected response (${xhr.status}).`));
            }
        };
        xhr.send(form);
    });
}

function ModelStages({ label, models, showLabel }) {
    return (
        <div className="im-models" role="group" aria-label={label}>
            {showLabel && <span>{label}</span>}
            <span>Vision analysis <strong>{models.vision_model || 'Unknown'}</strong></span>
            <span>DSL generation <strong>{models.dsl_model || 'Unknown'}</strong></span>
        </div>
    );
}

function DownloadLink({ href, children }) {
    return (
        <a className="p-button p-component p-button-sm p-button-outlined p-button-secondary im-download" href={href}>
            <span className="p-button-icon p-button-icon-left pi pi-download" aria-hidden="true" />
            <span className="p-button-label">{children}</span>
        </a>
    );
}

function SectionDetail({ row, session, onApply }) {
    const [frameIndex, setFrameIndex] = useState(0);
    const [applying, setApplying] = useState(false);
    const [applyError, setApplyError] = useState('');
    const videoRef = useRef(null);
    const mediaUrl = session.video ? `${API}/videos/${session.video.id}/media` : '';
    const path = `${API}/sessions/${session.id}/sections/${row.index}`;
    const shownFrame = Math.min(frameIndex, Math.max(0, (row.frame_count || 1) - 1));
    const offset = session.video_takeoff_offset || 0;

    useEffect(() => {
        const video = videoRef.current;
        if (video && video.readyState > 0) video.currentTime = row.start;
    }, [row.start, mediaUrl]);

    async function apply() {
        setApplying(true);
        setApplyError('');
        try {
            const graph = await jsonRequest('/api/parse_dsl', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dsl: row.dsl }),
            });
            onApply('fsm-builder', { payload: graph });
        } catch (error) {
            setApplyError(error.message);
        } finally {
            setApplying(false);
        }
    }

    return (
        <article className="im-detail">
            <div className="im-detail-title">
                <h3>Section {String(row.index).padStart(3, '0')}</h3>
                <span>{timestamp(row.start - offset)}–{timestamp(row.end - offset)} · {label(row.input_quality)}</span>
            </div>
            {mediaUrl && <video ref={videoRef} className="im-media" src={mediaUrl} controls preload="metadata"
                onLoadedMetadata={event => { event.currentTarget.currentTime = row.start; }}
                onTimeUpdate={event => {
                    if (event.currentTarget.currentTime >= row.end) event.currentTarget.pause();
                }} />}
            {!!row.frame_count && (
                <details className="im-fold" open={!mediaUrl}>
                    <summary>Model input · {row.frame_count} sampled frames</summary>
                    <img className="im-media" src={`${path}/frames/${shownFrame}`} alt={`Sampled frame ${shownFrame + 1}`} />
                    <label className="im-frame-control">
                        Frame {shownFrame + 1} · +{row.frame_times[shownFrame]} s
                        <input aria-label="Sampled frame" type="range" min="0" max={row.frame_count - 1}
                            value={shownFrame} onChange={event => setFrameIndex(Number(event.target.value))} />
                    </label>
                </details>
            )}
            <div className="im-content-block">
                <h4>Inferred mission</h4>
                <p className="im-mission">{row.inference_text || `Inference ${label(row.inference_status)}.`}</p>
                {row.adaptation && <p>{row.adaptation} Generation input: {row.instruction}</p>}
                {row.error && <p role="alert" className="im-error">{row.error}</p>}
                {row.unresolved?.map(message => <p key={message} className="im-note">{message}</p>)}
            </div>
            <div className="im-content-block">
                <div className="im-detail-title">
                    <h4>SteelEagle DSL</h4>
                    <span className="im-badge">{label(row.compile_status)}</span>
                </div>
                {row.dsl ? <pre><code>{row.dsl}</code></pre> : <p className="im-muted">
                    {row.status === 'failed' || row.status === 'skipped' || row.status === 'gap'
                        ? 'No compiled DSL candidate for this section.' : 'Waiting for a validated DSL candidate.'}
                </p>}
                <div className="im-actions im-artifact-actions">
                    {row.dsl && <DownloadLink href={`${path}/files/mission.dsl`}>Download DSL</DownloadLink>}
                    {row.mission_json && <DownloadLink href={`${path}/files/mission.json`}>Download JSON</DownloadLink>}
                    <DownloadLink href={`${path}/files/inference.json`}>Download record</DownloadLink>
                    {row.dsl && <Button type="button" className="im-apply" label="Apply to FSM Builder"
                        icon="pi pi-arrow-right" size="small" disabled={applying} onClick={apply} />}
                </div>
                {applyError && <p role="alert" className="im-error">{applyError}</p>}
            </div>
            {row.mission_json && (
                <details className="im-fold">
                    <summary>Compiled Mission JSON and transitions</summary>
                    <ul className="im-transitions">
                        {Object.entries(row.mission_json.transitions).flatMap(([action, edges]) =>
                            Object.entries(edges).map(([event, target]) =>
                                <li key={`${action}/${event}`}>{action} → {target} <span>({event})</span></li>))}
                    </ul>
                    <pre><code>{JSON.stringify(row.mission_json, null, 2)}</code></pre>
                </details>
            )}
            <details className="im-fold">
                <summary>History and generation attempts</summary>
                {row.history?.length ? <ol>
                    {row.history.map(item => <li key={item.section}>Section {item.section} · weight {item.weight.toFixed(3)} · {item.text}</li>)}
                </ol> : <p>No prior inferred missions used.</p>}
                {(row.attempts || []).map((attempt, index) => (
                    <details key={index} className="im-fold">
                        <summary>Attempt {index + 1} · {label(attempt.status)}</summary>
                        {attempt.errors.map(error => <p key={error} className="im-error">{error}</p>)}
                        <pre><code>{attempt.normalized_dsl}</code></pre>
                    </details>
                ))}
                {row.elapsed != null && <p>Total {row.elapsed}s · inference {row.inference_elapsed ?? '—'}s · generation {row.generation_elapsed ?? '—'}s</p>}
                {row.telemetry && <pre>{JSON.stringify(row.telemetry, null, 2)}</pre>}
                {row.vision_prompt && <details><summary>Vision prompt</summary><pre>{row.vision_prompt}</pre></details>}
                {row.dsl_reference && <details><summary>DSL reference</summary><pre>{JSON.stringify(row.dsl_reference, null, 2)}</pre></details>}
            </details>
        </article>
    );
}

export default function InferredMissionPage({ vehicles, onApplyArtifact }) {
    const [config, setConfig] = useState(null);
    const [runs, setRuns] = useState([]);
    const [source, setSource] = useState('live');
    const [feed, setFeed] = useState('local');
    const [vehicle, setVehicle] = useState('');
    const [takeoff, setTakeoff] = useState('');
    const [videoTakeoff, setVideoTakeoff] = useState('');
    const [historyMode, setHistoryMode] = useState('');
    const [asset, setAsset] = useState(null);
    const [progress, setProgress] = useState(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [snapshot, setSnapshot] = useState(null);
    const [selectedIndex, setSelectedIndex] = useState(null);
    const [detail, setDetail] = useState(null);
    const [now, setNow] = useState(() => Date.now() / 1000);
    const session = snapshot?.id === sessionId ? snapshot : null;
    const active = Boolean(session && ACTIVE.has(session.status));
    const index = selectedIndex ?? session?.sections?.at(-1)?.index;
    const summary = session?.sections?.find(row => row.index === index);
    const detailStatus = summary?.status;

    useEffect(() => {
        const controller = new AbortController();
        const options = { signal: controller.signal };
        Promise.all([jsonRequest(`${API}/status`, options), jsonRequest(`${API}/sessions`, options)])
            .then(([settings, saved]) => {
                setConfig(settings);
                setRuns(saved);
                if (!settings.local_vehicles.length && settings.remote_available) setFeed('remote');
                if (saved.length) setSessionId(saved[0].id);
            }).catch(failure => { if (!controller.signal.aborted) setError(failure.message); });
        return () => controller.abort();
    }, []);

    useEffect(() => {
        if (!sessionId) return undefined;
        const controller = new AbortController();
        jsonRequest(`${API}/sessions/${sessionId}`, { signal: controller.signal })
            .then(setSnapshot).catch(failure => { if (!controller.signal.aborted) setError(failure.message); });
        return () => controller.abort();
    }, [sessionId]);

    useEffect(() => {
        if (!sessionId || !active) return undefined;
        const events = new EventSource(`${API}/sessions/${sessionId}/events`);
        events.addEventListener('update', event => setSnapshot(JSON.parse(event.data)));
        return () => events.close();
    }, [sessionId, active]);

    useEffect(() => {
        if (!sessionId || index == null || !detailStatus) return undefined;
        const controller = new AbortController();
        jsonRequest(`${API}/sessions/${sessionId}/sections/${index}`, { signal: controller.signal })
            .then(row => setDetail({ sessionId, row }))
            .catch(failure => { if (!controller.signal.aborted) setError(failure.message); });
        return () => controller.abort();
    }, [sessionId, index, detailStatus]);

    useEffect(() => {
        if (!active) return undefined;
        const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
        return () => clearInterval(timer);
    }, [active]);

    async function chooseFile(file) {
        if (!file || busy) return;
        setBusy(true);
        setError('');
        setAsset(null);
        setVideoTakeoff('');
        setProgress(0);
        try {
            setAsset(await uploadVideo(file, setProgress));
        } catch (failure) {
            setError(failure.message);
        } finally {
            setProgress(null);
            setBusy(false);
        }
    }

    async function start() {
        setBusy(true);
        setError('');
        try {
            const body = { source_type: source };
            if (historyMode) body.history_mode = historyMode;
            if (source === 'uploaded_video') {
                body.video_asset_id = asset.id;
                if (videoTakeoff !== '') body.video_takeoff_offset = Number(videoTakeoff);
            }
            else {
                body.vehicle_id = vehicle;
                body.feed = feed;
                if (takeoff) body.flight_started_at = new Date(takeoff).getTime() / 1000;
            }
            const run = await jsonRequest(`${API}/sessions`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
            });
            setRuns(previous => [run, ...previous]);
            setSnapshot(run);
            setSessionId(run.id);
            setSelectedIndex(null);
        } catch (failure) {
            setError(failure.message);
        } finally {
            setBusy(false);
        }
    }

    async function action(name) {
        setBusy(true);
        setError('');
        try {
            setSnapshot(await jsonRequest(`${API}/sessions/${sessionId}/${name}`, { method: 'POST' }));
        } catch (failure) {
            setError(failure.message);
        } finally {
            setBusy(false);
        }
    }

    const row = detail?.sessionId === sessionId && detail?.row.index === index ? detail.row : null;
    const frameAge = session?.last_frame_received ? Math.max(0, Math.floor(now - session.last_frame_received)) : null;
    const videoMode = source === 'uploaded_video';
    return (
        <section className="im-page" aria-label="Inferred Mission">
            <div className="im-input">
                <div className="im-input-heading">
                    <div className="im-source" aria-label="Footage source">
                        <button type="button" aria-pressed={!videoMode} onClick={() => setSource('live')}>Live observation</button>
                        <button type="button" aria-pressed={videoMode} onClick={() => setSource('uploaded_video')}>Upload MP4</button>
                    </div>
                    <div className="im-actions">
                        <label>History <select value={historyMode} onChange={event => setHistoryMode(event.target.value)}>
                            <option value="">Default ({config?.settings.history_mode || 'recency'})</option>
                            <option value="recency">Recency</option><option value="uniform">Uniform</option>
                            <option value="no_memory">No memory</option>
                        </select></label>
                        <span className="im-badge">Analysis only · prompt {config?.prompt_version || 'v5'}</span>
                    </div>
                </div>
                {videoMode ? (
                    <div className="im-upload" onDragOver={event => event.preventDefault()}
                        onDrop={event => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]); }}>
                        <label>Choose or drop an MP4 <input type="file" accept=".mp4,video/mp4" disabled={busy}
                            onChange={event => { chooseFile(event.target.files[0]); event.target.value = ''; }} /></label>
                        {progress != null && <p role="status">{progress < 100 ? `Uploading ${progress}%` : 'Checking video…'}</p>}
                        {asset && <p>{asset.name} · {asset.duration.toFixed(1)}s · {Math.ceil(asset.duration / (config?.settings.section_seconds || 15))} sections</p>}
                        {asset && <label>Takeoff offset in seconds (optional) <input type="number" min="0" max={asset.duration} step="0.1"
                            value={videoTakeoff} onChange={event => setVideoTakeoff(event.target.value)} /></label>}
                        {config && !config.media_available && <p className="im-note">Install FFmpeg and ffprobe on the GCS backend.</p>}
                    </div>
                ) : (
                    <div className="im-live-fields">
                        <label>Feed <select value={feed} onChange={event => { setFeed(event.target.value); setVehicle(''); }}>
                            <option value="local">Local</option><option value="remote">Remote</option>
                        </select></label>
                        {feed === 'local' ? <label>Vehicle <select value={vehicle} onChange={event => setVehicle(event.target.value)}>
                            <option value="">Select a vehicle</option>
                            {config?.local_vehicles.map(name => <option key={name}>{name}</option>)}
                        </select></label> : <label>Vehicle <input value={vehicle} list="im-vehicles" placeholder="Vehicle name"
                            onChange={event => setVehicle(event.target.value)} />
                            <datalist id="im-vehicles">{vehicles?.map(item => <option key={item.name} value={item.name} />)}</datalist>
                        </label>}
                        <label>Takeoff time (optional) <input type="datetime-local" value={takeoff} onChange={event => setTakeoff(event.target.value)} /></label>
                    </div>
                )}
                <div className="im-input-footer">
                    {config ? <ModelStages label="New run models" models={config.models} /> : <span>Loading model settings…</span>}
                    <button type="button" className="im-primary" onClick={start}
                        disabled={busy || !config?.configured || (videoMode ? !asset || !config?.media_available : !vehicle || (feed === 'remote' && !config?.remote_available))}>
                        {videoMode ? 'Start analysis' : 'Start observing'}
                    </button>
                </div>
                {config && !config.configured && <p className="im-note">Configure the OpenAI vision key and GCS Chat model on the backend.</p>}
            </div>
            {error && <p role="alert" className="im-error">{error}</p>}
            <div className="im-run-bar">
                <label>Run <select value={sessionId} onChange={event => { setSessionId(event.target.value); setSelectedIndex(null); }}>
                    <option value="">Select an analysis run</option>
                    {runs.map(run => <option value={run.id} key={run.id}>
                        {run.video ? `Video · ${run.video.name}` : `Live · ${run.vehicle_id}`} · {new Date(run.created_at * 1000).toLocaleString()}
                    </option>)}
                </select></label>
                {session && <div className="im-actions">
                    <span className="im-badge">{label(session.status)}</span>
                    {session.status === 'analyzing' && <button type="button" disabled={busy} onClick={() => action('pause')}>Pause</button>}
                    {session.status === 'paused' && <button type="button" disabled={busy} onClick={() => action('resume')}>Resume</button>}
                    {(active || session.status === 'paused') && <button type="button" disabled={busy || session.status === 'stopping'} onClick={() => action('stop')}>Stop</button>}
                </div>}
            </div>
            {session && <>
                <p className="im-run-summary" role="status">
                    {session.video_takeoff_offset != null ? 'Time relative to supplied takeoff offset' :
                        session.video ? 'Video time' : session.time_basis === 'takeoff' ? 'Time since takeoff' : 'Time since observation began'}
                    {' · '}{session.sections.length}{session.total_sections ? ` / ${session.total_sections}` : ''} sections
                    {active && session.source_type === 'live' && ` · ${session.source_notice}`}
                    {active && session.collecting_section != null && ` Section ${session.collecting_section} collecting (${session.collecting_frames} frames).`}
                    {active && frameAge != null && ` Last frame ${frameAge}s ago${frameAge > 5 ? ' (stale)' : ''}.`}
                </p>
                {session.models && <ModelStages label="Selected run models" models={session.models} showLabel />}
                {session.error && <p className="im-note">{session.error}</p>}
                <div className="im-report">
                    <aside className="im-timeline" aria-label="Section timeline">
                        <div className="im-detail-title"><h3>Timeline</h3>
                            {selectedIndex != null && <button type="button" onClick={() => setSelectedIndex(null)}>Follow latest</button>}
                        </div>
                        {session.sections.map(item => <button type="button" key={item.index} aria-current={item.index === index ? 'true' : undefined}
                            onClick={() => setSelectedIndex(item.index)} className="im-section">
                            <span><strong>#{String(item.index).padStart(3, '0')}</strong> {timestamp(item.start - (session.video_takeoff_offset || 0))}–{timestamp(item.end - (session.video_takeoff_offset || 0))}</span>
                            <span className="im-muted">{label(item.status)} · {label(item.compile_status)}</span>
                            <span className="im-section-nl">{item.inference_text || item.error || 'Preparing section'}</span>
                        </button>)}
                        {!session.sections.length && <p className="im-muted">Waiting for the first section.</p>}
                    </aside>
                    {row ? <SectionDetail key={`${sessionId}/${row.index}`} row={row} session={session} onApply={onApplyArtifact} />
                        : <div className="im-empty">{summary ? 'Loading section…' : 'Section details will appear here.'}</div>}
                </div>
            </>}
        </section>
    );
}
