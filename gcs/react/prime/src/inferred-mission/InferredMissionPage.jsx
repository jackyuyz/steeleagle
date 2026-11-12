import { useEffect, useId, useRef, useState } from 'react';
import { Button } from 'primereact/button';
import { Menu } from 'primereact/menu';
import { TabPanel, TabView } from 'primereact/tabview';
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

function ModelStages({ label, models }) {
    return (
        <div className="im-models" role="group" aria-label={label}>
            <span>Vision analysis <strong>{models.vision_model || 'Unknown'}</strong></span>
            <span>DSL generation <strong>{models.dsl_model || 'Unknown'}</strong></span>
        </div>
    );
}

function SectionDetail({ row, session, onApply }) {
    const [frameIndex, setFrameIndex] = useState(0);
    const [applying, setApplying] = useState(false);
    const [applyError, setApplyError] = useState('');
    const videoRef = useRef(null);
    const downloadMenu = useRef(null);
    const mediaUrl = session.video ? `${API}/videos/${session.video.id}/media` : '';
    const path = `${API}/sessions/${session.id}/sections/${row.index}`;
    const shownFrame = Math.min(frameIndex, Math.max(0, (row.frame_count || 1) - 1));
    const frameTime = row.frame_times?.[shownFrame];
    const frameLabel = `Frame ${shownFrame + 1} of ${row.frame_count}`
        + (Number.isFinite(frameTime) ? ` · +${frameTime.toFixed(2)} s` : '');
    const offset = session.video_takeoff_offset || 0;
    const downloads = [
        { label: 'Download DSL', url: `${path}/files/mission.dsl`, visible: Boolean(row.dsl) },
        { label: 'Download JSON', url: `${path}/files/mission.json`, visible: Boolean(row.mission_json) },
        { label: 'Download record', url: `${path}/files/inference.json` },
    ];

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
            <div className="im-inference">
                <p className="im-mission">{row.inference_text || `Inference ${label(row.inference_status)}.`}</p>
                {row.adaptation && <p>{row.adaptation} Generation input: {row.instruction}</p>}
                {row.error && <p role="alert" className="im-error">{row.error}</p>}
                {row.unresolved?.map(message => <p key={message} className="im-note">{message}</p>)}
            </div>
            <div className="im-detail-grid">
                <div className="im-evidence">
                    {mediaUrl && <video ref={videoRef} className="im-media" src={mediaUrl} controls preload="metadata"
                        onLoadedMetadata={event => { event.currentTarget.currentTime = row.start; }}
                        onTimeUpdate={event => {
                            if (event.currentTarget.currentTime >= row.end) event.currentTarget.pause();
                        }} />}
                    {row.frame_count ? (
                        <details className="im-sampled-frames" open={!mediaUrl}>
                            <summary>Sampled frames <span className="im-muted">{row.frame_count} frames</span></summary>
                            <img className="im-media" src={`${path}/frames/${shownFrame}`} alt={`Sampled frame ${shownFrame + 1}`} />
                            <label className="im-frame-control">
                                <input aria-label="Sampled frame" aria-valuetext={frameLabel} type="range"
                                    min="0" max={row.frame_count - 1} value={shownFrame}
                                    onChange={event => setFrameIndex(Number(event.target.value))} />
                                <span>{frameLabel}</span>
                            </label>
                        </details>
                    ) : !mediaUrl && <p className="im-muted">No sampled frames for this section.</p>}
                </div>
                <div className="im-artifacts">
                    <div className="im-artifact-toolbar">
                        <span className="im-badge">{label(row.compile_status)}</span>
                        <Button type="button" label="Download" icon="pi pi-download" size="small"
                            outlined severity="secondary" aria-haspopup="menu"
                            onClick={event => downloadMenu.current.toggle(event)} />
                        <Menu ref={downloadMenu} model={downloads} popup />
                    </div>
                    <TabView className="im-artifact-tabs">
                        <TabPanel header="SteelEagle DSL">
                            {row.dsl ? <pre><code>{row.dsl}</code></pre> : <p className="im-muted">
                                {row.status === 'failed' || row.status === 'skipped' || row.status === 'gap'
                                    ? 'No compiled DSL candidate for this section.' : 'Waiting for a validated DSL candidate.'}
                            </p>}
                        </TabPanel>
                        <TabPanel header="JSON" disabled={!row.mission_json}>
                            {row.mission_json && <>
                                <ul className="im-transitions">
                                    {Object.entries(row.mission_json.transitions).flatMap(([action, edges]) =>
                                        Object.entries(edges).map(([event, target]) =>
                                            <li key={`${action}/${event}`}>{action} → {target} <span>({event})</span></li>))}
                                </ul>
                                <pre><code>{JSON.stringify(row.mission_json, null, 2)}</code></pre>
                            </>}
                        </TabPanel>
                        <TabPanel header="History">
                            {session.models && <ModelStages label="Selected run models" models={session.models} />}
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
                        </TabPanel>
                    </TabView>
                    {row.dsl && <div className="im-artifact-actions">
                        <Button type="button" label="Apply to FSM Builder" icon="pi pi-arrow-right"
                            size="small" disabled={applying} onClick={apply} />
                        {applyError && <p role="alert" className="im-error">{applyError}</p>}
                    </div>}
                </div>
            </div>
        </article>
    );
}

export default function InferredMissionPage({ vehicles, onApplyArtifact }) {
    const [config, setConfig] = useState(null);
    const [runs, setRuns] = useState([]);
    const [newRunOpen, setNewRunOpen] = useState(null);
    const newRunPanelId = useId();
    const videoInput = useRef(null);
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
    const showNewRun = newRunOpen ?? runs.length === 0;
    const active = Boolean(session && ACTIVE.has(session.status));
    const queuedCount = session?.sections?.filter(row => row.status === 'queued').length ?? 0;
    const index = selectedIndex ?? session?.sections?.findLast(row => row.status !== 'queued')?.index
        ?? session?.sections?.at(-1)?.index;
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
            setNewRunOpen(false);
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
    const sourceReady = videoMode ? asset && config?.media_available
        : vehicle && (feed === 'local' || config?.remote_available);
    const canStart = Boolean(!busy && config?.configured && sourceReady);
    const liveTarget = vehicle ? `Feed: ${feed === 'local' ? 'Local' : 'Remote'} · Vehicle: ${vehicle}`
        : 'Select a vehicle to observe.';
    const videoTarget = asset?.name || (progress == null ? 'Choose an MP4 to analyze.' : 'Preparing video…');
    const startTarget = videoMode ? videoTarget : liveTarget;
    const startTargetId = `${newRunPanelId}-start-target`;
    return (
        <section className="im-page" aria-label="Inferred Mission">
            <div className="im-run-bar">
                <div className="im-actions im-run-selection">
                    <label>Run <select value={sessionId} onChange={event => {
                        const runId = event.target.value;
                        setSessionId(runId);
                        setSelectedIndex(null);
                        setNewRunOpen(!runId);
                    }}>
                        <option value="">Select an analysis run</option>
                        {runs.map(run => <option value={run.id} key={run.id}>
                            {run.video ? `Video · ${run.video.name}` : `Live · ${run.vehicle_id}`} · {new Date(run.created_at * 1000).toLocaleString()}
                        </option>)}
                    </select></label>
                    {session && <>
                        <span className="im-badge">{label(session.status)}</span>
                        <span className="im-muted">
                            {session.sections.length}{session.total_sections ? ` / ${session.total_sections}` : ''} sections
                            {queuedCount > 0 && ` · ${queuedCount} queued`}
                        </span>
                    </>}
                </div>
                <div className="im-actions">
                    {session?.status === 'analyzing' && <button type="button" disabled={busy} onClick={() => action('pause')}>Pause</button>}
                    {session?.status === 'paused' && <button type="button" disabled={busy} onClick={() => action('resume')}>Resume</button>}
                    {(active || session?.status === 'paused') && <button type="button" disabled={busy || session.status === 'stopping'} onClick={() => action('stop')}>Stop</button>}
                    <Button type="button" label="New run" icon={showNewRun ? 'pi pi-chevron-up' : 'pi pi-chevron-down'}
                        size="small" outlined aria-expanded={showNewRun} aria-controls={newRunPanelId}
                        onClick={() => setNewRunOpen(!showNewRun)} />
                </div>
            </div>
            <div className="im-input" id={newRunPanelId} hidden={!showNewRun}>
                <div className="im-run-settings">
                    <div className="im-source" aria-label="Footage source">
                        <button type="button" aria-pressed={!videoMode} onClick={() => setSource('live')}>Live observation</button>
                        <button type="button" aria-pressed={videoMode} onClick={() => setSource('uploaded_video')}>Upload MP4</button>
                    </div>
                    {videoMode ? (
                        <div className={asset ? 'im-upload im-upload-ready' : 'im-upload'} onDragOver={event => event.preventDefault()}
                            onDrop={event => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]); }}>
                            <input ref={videoInput} type="file" accept=".mp4,video/mp4" disabled={busy} hidden
                                onChange={event => { chooseFile(event.target.files[0]); event.target.value = ''; }} />
                            <div className="im-upload-row">
                                <div className="im-upload-info">
                                    {asset ? <>
                                        <strong>{asset.name}</strong>
                                        <p className="im-muted">
                                            {asset.duration.toFixed(1)} s · {Math.ceil(asset.duration / (config?.settings.section_seconds || 15))} sections
                                        </p>
                                    </> : progress != null ? (
                                        <p role="status">{progress < 100 ? `Uploading ${progress}%` : 'Checking video…'}</p>
                                    ) : <p className="im-muted">Drop an MP4 here</p>}
                                </div>
                                <Button type="button" label={asset ? 'Change file' : 'Choose MP4'} icon="pi pi-folder-open"
                                    size="small" outlined severity="secondary" disabled={busy}
                                    onClick={() => videoInput.current?.click()} />
                            </div>
                            {config && !config.media_available && <p className="im-note">Install FFmpeg and ffprobe on the GCS backend.</p>}
                        </div>
                    ) : (
                        <div className="im-live-fields">
                            <label className="im-feed-field">Feed <select value={feed} onChange={event => { setFeed(event.target.value); setVehicle(''); }}>
                                <option value="local">Local</option><option value="remote">Remote</option>
                            </select></label>
                            {feed === 'local' ? <label className="im-vehicle-field">Vehicle <select value={vehicle} onChange={event => setVehicle(event.target.value)}>
                                <option value="">Select a vehicle</option>
                                {config?.local_vehicles.map(name => <option key={name}>{name}</option>)}
                            </select></label> : <label className="im-vehicle-field">Vehicle <input value={vehicle} list="im-vehicles" placeholder="Vehicle name"
                                onChange={event => setVehicle(event.target.value)} />
                                <datalist id="im-vehicles">{vehicles?.map(item => <option key={item.name} value={item.name} />)}</datalist>
                            </label>}
                        </div>
                    )}
                    <details className="im-advanced">
                        <summary>Advanced options</summary>
                        <div className="im-actions im-advanced-fields">
                            <label>History <select value={historyMode} onChange={event => setHistoryMode(event.target.value)}>
                                <option value="">Default ({config?.settings.history_mode || 'recency'})</option>
                                <option value="recency">Recency</option><option value="uniform">Uniform</option>
                                <option value="no_memory">No memory</option>
                            </select></label>
                            {videoMode ? asset && <label>Takeoff offset in seconds (optional) <input type="number" min="0" max={asset.duration} step="0.1"
                                value={videoTakeoff} onChange={event => setVideoTakeoff(event.target.value)} /></label> :
                                <label>Takeoff time (optional) <input type="datetime-local" value={takeoff}
                                    aria-describedby="im-takeoff-help" onChange={event => setTakeoff(event.target.value)} /></label>}
                        </div>
                        {!videoMode && <p className="im-note" id="im-takeoff-help">Leave blank to start the timeline at the first valid frame after you start observing.</p>}
                    </details>
                    {config ? <ModelStages label="New run models" models={config.models} /> : <span className="im-muted">Loading model settings…</span>}
                    {config && !config.configured && <p className="im-note">Configure the OpenAI vision key and GCS Chat model on the backend.</p>}
                </div>
                <div className="im-start-panel">
                    <Button type="button" label={videoMode ? 'Start analysis' : 'Start observing'}
                        icon="pi pi-play" onClick={start} disabled={!canStart} aria-describedby={startTargetId} />
                    <p className="im-muted" id={startTargetId}>{startTarget}</p>
                </div>
            </div>
            {error && <p role="alert" className="im-error">{error}</p>}
            {session && <>
                {(!session.video || session.video_takeoff_offset != null) && (
                    <p className="im-run-summary" role="status">
                        {session.video_takeoff_offset != null ? 'Time relative to supplied takeoff offset' :
                            session.time_basis === 'takeoff' ? 'Time since takeoff' : 'Time since observation began'}
                        {active && session.source_type === 'live' && ` · ${session.source_notice}`}
                        {active && session.collecting_section != null && ` Section ${session.collecting_section} collecting (${session.collecting_frames} frames).`}
                        {active && frameAge != null && ` Last frame ${frameAge}s ago${frameAge > 5 ? ' (stale)' : ''}.`}
                    </p>
                )}
                {session.error && <p className="im-note">{session.error}</p>}
                <div className="im-report">
                    <aside className="im-timeline" aria-label="Section timeline">
                        <div className="im-detail-title"><h3>Timeline</h3>
                            {selectedIndex != null && <button type="button" onClick={() => setSelectedIndex(null)}>Follow latest</button>}
                        </div>
                        {session.sections.map(item => <button type="button" key={item.index} aria-current={item.index === index ? 'true' : undefined}
                            onClick={() => setSelectedIndex(item.index)} className="im-section">
                            <span><strong>#{String(item.index).padStart(3, '0')}</strong> {timestamp(item.start - (session.video_takeoff_offset || 0))}–{timestamp(item.end - (session.video_takeoff_offset || 0))}</span>
                            {item.status !== 'completed' && <span className="im-muted">{label(item.status)}</span>}
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
