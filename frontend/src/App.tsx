import { useEffect, useRef, useState } from "react";
import {
  createJob,
  getJob,
  scoreUrl,
  separateSection,
  transcribe,
  type Instrument,
  type Job,
  type TranscriptionMode,
} from "./api";
import { ScoreView } from "./ScoreView";
import { SectionPicker } from "./SectionPicker";

const INSTRUMENTS: { value: Instrument; label: string }[] = [
  { value: "alto", label: "Alto sax (Eb)" },
  { value: "tenor", label: "Tenor sax (Bb)" },
  { value: "bari", label: "Baritone sax (Eb)" },
  { value: "soprano", label: "Soprano sax (Bb)" },
  { value: "concert", label: "Concert pitch" },
];

const SAX_OPTIONS: { value: SaxType; label: string }[] = [
  { value: "alto", label: "Alto sax (Eb)" },
  { value: "tenor", label: "Tenor sax (Bb)" },
  { value: "bari", label: "Baritone sax (Eb)" },
  { value: "soprano", label: "Soprano sax (Bb)" },
];

const TIME_SIGNATURES = ["4/4", "3/4", "2/4", "6/8"];

type SaxType = Exclude<Instrument, "concert">;

function instrumentLabel(value: string): string {
  return INSTRUMENTS.find((item) => item.value === value)?.label ?? value;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function clock(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [editingSection, setEditingSection] = useState(false);
  const [instrument, setInstrument] = useState<Instrument>("alto");
  const [saxType, setSaxType] = useState<SaxType>("alto");
  const [method, setMethod] = useState<TranscriptionMode>("melody");
  const [timeSignature, setTimeSignature] = useState("4/4");
  const [bpm, setBpm] = useState<number | null>(null);
  const [bpmDraft, setBpmDraft] = useState("");
  const [scoreXml, setScoreXml] = useState("");
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [scoreLoading, setScoreLoading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (
      !job ||
      job.status === "done" ||
      job.status === "error" ||
      job.status === "awaiting_stem" ||
      job.status === "awaiting_section"
    ) {
      return;
    }
    const timer = window.setInterval(() => {
      getJob(job.id)
        .then(setJob)
        .catch((err) => setError(message(err)));
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (job?.transcription && bpm === null) {
      setBpm(job.transcription.bpm);
      setBpmDraft(job.transcription.bpm.toFixed(1));
    }
  }, [job?.transcription, bpm]);

  useEffect(() => {
    if (!job || job.status !== "done" || bpm === null) return;
    const controller = new AbortController();
    setScoreLoading(true);
    setScoreError(null);
    fetch(scoreUrl(job.id, instrument, timeSignature, bpm, "musicxml"), {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Could not build the score (${response.status})`);
        return response.text();
      })
      .then(setScoreXml)
      .catch((err) => {
        if ((err as Error).name !== "AbortError") setScoreError(message(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setScoreLoading(false);
      });
    return () => controller.abort();
  }, [job?.id, job?.status, instrument, timeSignature, bpm]);

  async function handleFile(file: File) {
    setUploading(true);
    setError(null);
    setJob(null);
    setScoreXml("");
    setScoreError(null);
    setBpm(null);
    setBpmDraft("");
    setEditingSection(false);
    try {
      setJob(await createJob(file));
    } catch (err) {
      setError(message(err));
    } finally {
      setUploading(false);
    }
  }

  async function handleStem(stem: string) {
    if (!job) return;
    setError(null);
    setScoreXml("");
    setScoreError(null);
    setBpm(null);
    setBpmDraft("");
    setInstrument(saxType);
    try {
      setJob(await transcribe(job.id, stem, saxType, method));
    } catch (err) {
      setError(message(err));
    }
  }

  async function handleSection(start: number, end: number) {
    if (!job) return;
    setError(null);
    setScoreXml("");
    setScoreError(null);
    setBpm(null);
    setBpmDraft("");
    try {
      setJob(await separateSection(job.id, start, end));
      setEditingSection(false);
    } catch (err) {
      setError(message(err));
      throw err;
    }
  }

  function adjustBpm(factor: number) {
    const base = bpm ?? job?.transcription?.bpm;
    if (!base) return;
    const next = Math.min(400, Math.max(20, base * factor));
    setBpm(next);
    setBpmDraft(next.toFixed(1));
  }

  function commitBpm() {
    const value = Number(bpmDraft);
    if (Number.isFinite(value) && value >= 20 && value <= 400) {
      setBpm(value);
    } else if (bpm !== null) {
      setBpmDraft(bpm.toFixed(1));
    }
  }

  function reset() {
    setJob(null);
    setError(null);
    setScoreXml("");
    setScoreError(null);
    setBpm(null);
    setBpmDraft("");
    setUploading(false);
    setEditingSection(false);
  }

  const busy =
    uploading ||
    job?.status === "preparing" ||
    job?.status === "separating" ||
    job?.status === "transcribing";
  const showPicker = Boolean(
    job &&
      job.duration > 0 &&
      (job.status === "awaiting_section" || editingSection),
  );
  const sectionLabel = job?.section
    ? `${clock(job.section.start)} - ${clock(job.section.end)}`
    : null;

  return (
    <div className="page">
      <header className="hero">
        <div className="eyebrow">Solo Transcriber</div>
        <h1>Play along with the record.</h1>
        <p>
          Drop in a track, pick the stem that carries the sax, and get a transposed
          part you can read in MuseScore or any notation app.
        </p>
      </header>

      {!job && (
        <section
          className={`dropzone ${dragging ? "dragging" : ""} ${busy ? "busy" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files[0];
            if (file) void handleFile(file);
          }}
        >
          <input
            ref={fileInput}
            type="file"
            accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg,.aiff"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void handleFile(file);
            }}
          />
          <div className="drop-icon">♫</div>
          <h2>{uploading ? "Uploading..." : "Drop an audio file"}</h2>
          <p>MP3, WAV, M4A, FLAC or OGG. You will pick the solo section next.</p>
          <button
            className="primary"
            type="button"
            disabled={busy}
            onClick={() => fileInput.current?.click()}
          >
            Choose a track
          </button>
        </section>
      )}

      {error && (
        <div className="banner error">
          <span>{error}</span>
          <button type="button" onClick={reset}>
            Start over
          </button>
        </div>
      )}

      {job && (
        <section className="panel">
          <div className="panel-head">
            <div>
              <div className="muted">Now working on</div>
              <h2>{job.filename}</h2>
            </div>
            <div className="head-actions">
              {job.mock && <span className="badge">Demo mode</span>}
              {job.duration > 0 &&
                job.status !== "awaiting_section" &&
                !editingSection && (
                <button
                  type="button"
                  className="ghost"
                  disabled={busy}
                  onClick={() => setEditingSection(true)}
                >
                  Change section
                </button>
              )}
              <button type="button" className="ghost" onClick={reset}>
                New track
              </button>
            </div>
          </div>

          <div className="status">
            <div className="status-row">
              <span>{job.stage}</span>
              <span className="muted">
                {sectionLabel
                  ? `Section ${sectionLabel}`
                  : job.duration
                    ? `${Math.round(job.duration)}s`
                    : ""}
              </span>
            </div>
            <div className="track">
              <div className="fill" style={{ width: `${job.progress}%` }} />
            </div>
          </div>

          {showPicker && job && (
            <SectionPicker
              url={job.preview_url ?? `/media/jobs/${job.id}/source.wav`}
              duration={job.duration}
              initialStart={job.section?.start ?? 0}
              initialEnd={job.section?.end ?? job.duration}
              canCancel={job.status !== "awaiting_section"}
              onConfirm={handleSection}
              onCancel={() => setEditingSection(false)}
            />
          )}

          {job.status === "awaiting_stem" && (
            <div className="notice">
              The sax is usually in <strong>other</strong>. Audition the stems below
              and pick the closest one.
            </div>
          )}

          {job.stems.length > 0 && (
            <div className="transcribe-options">
              <label>
                Your sax
                <select
                  value={saxType}
                  onChange={(event) => {
                    const value = event.target.value as SaxType;
                    setSaxType(value);
                    setInstrument(value);
                  }}
                >
                  {SAX_OPTIONS.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Transcription
                <select
                  value={method}
                  onChange={(event) =>
                    setMethod(event.target.value as TranscriptionMode)
                  }
                >
                  <option value="melody">Melody (follow one line)</option>
                  <option value="basic_pitch">Basic Pitch (polyphonic)</option>
                </select>
              </label>
              <p className="muted">
                Melody tracks the dominant single line inside your sax range.
                Basic Pitch transcribes every pitched note it hears.
              </p>
            </div>
          )}

          {job.stems.length > 0 && (
            <div className="stems">
              {job.stems.map((stem) => (
                <article
                  key={stem.name}
                  className={`stem ${stem.suggested ? "suggested" : ""} ${
                    job.selected_stem === stem.name ? "selected" : ""
                  }`}
                >
                  <div className="stem-head">
                    <strong>{stem.name}</strong>
                    {stem.suggested && <span className="badge">Try this first</span>}
                  </div>
                  <audio controls preload="none" src={stem.url} />
                  <button
                    type="button"
                    className="secondary"
                    disabled={job.status === "transcribing"}
                    onClick={() => void handleStem(stem.name)}
                  >
                    Transcribe this stem
                  </button>
                </article>
              ))}

              <article className="stem">
                <div className="stem-head">
                  <strong>original section</strong>
                  <span className="badge">Solo recordings</span>
                </div>
                <audio
                  controls
                  preload="none"
                  src={job.original_url ?? `/media/jobs/${job.id}/source.wav`}
                />
                <button
                  type="button"
                  className="secondary"
                  disabled={job.status === "transcribing"}
                  onClick={() => void handleStem("original")}
                >
                  Transcribe the mix
                </button>
              </article>
            </div>
          )}

          {job.status === "done" && job.transcription && (
            <div className="result">
              <div className="result-head">
                <div>
                  <div className="muted">Transcription</div>
                  <h2>
                    {job.transcription.note_count} notes at{" "}
                    {job.transcription.bpm.toFixed(1)} BPM
                  </h2>
                  <div className="muted">
                    {job.transcription.mode === "melody"
                      ? "Melody mode"
                      : "Basic Pitch mode"}{" "}
                    · {instrumentLabel(job.transcription.instrument)}
                  </div>
                  <div className="muted">
                    {job.transcription.first_note > 0.5
                      ? `Sax enters ${clock(job.transcription.first_note)} into the section; the score starts there`
                      : "The score starts at the first note in the section"}
                  </div>
                </div>
                <div className="downloads">
                  <a
                    className="primary"
                    href={scoreUrl(job.id, instrument, timeSignature, bpm ?? job.transcription.bpm, "musicxml")}
                  >
                    MusicXML
                  </a>
                  <a
                    className="secondary"
                    href={scoreUrl(job.id, instrument, timeSignature, bpm ?? job.transcription.bpm, "midi")}
                  >
                    MIDI
                  </a>
                </div>
              </div>

              <div className="controls">
                <label>
                  Instrument
                  <select
                    value={instrument}
                    onChange={(event) => setInstrument(event.target.value as Instrument)}
                  >
                    {INSTRUMENTS.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  Time signature
                  <select
                    value={timeSignature}
                    onChange={(event) => setTimeSignature(event.target.value)}
                  >
                    {TIME_SIGNATURES.map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  Tempo (BPM)
                  <div className="bpm-row">
                    <input
                      type="number"
                      min={20}
                      max={400}
                      step={0.1}
                      value={bpmDraft}
                      onChange={(event) => setBpmDraft(event.target.value)}
                      onBlur={commitBpm}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") commitBpm();
                      }}
                    />
                    <button
                      type="button"
                      className="ghost"
                      title="Use half tempo"
                      onClick={() => adjustBpm(0.5)}
                    >
                      /2
                    </button>
                    <button
                      type="button"
                      className="ghost"
                      title="Use double tempo"
                      onClick={() => adjustBpm(2)}
                    >
                      x2
                    </button>
                  </div>
                </label>

                <label>
                  Reference
                  <audio
                    controls
                    preload="none"
                    src={
                      job.selected_stem === "original"
                        ? job.original_url ?? `/media/jobs/${job.id}/source.wav`
                        : `/media/jobs/${job.id}/stems/${job.selected_stem}.wav`
                    }
                  />
                </label>
              </div>

              <p className="hint">
                Written for {instrumentLabel(instrument)}. Transcribed with{" "}
                {job.transcription.mode === "melody" ? "melody" : "Basic Pitch"}{" "}
                mode for {instrumentLabel(job.transcription.instrument)}. Change
                the displayed instrument or tempo without re-transcribing. Export
                to MusicXML for MuseScore, or MIDI for a DAW.
              </p>

              {scoreError && <div className="banner error">{scoreError}</div>}
              {scoreLoading && <div className="score-loading">Engraving the part...</div>}
              {scoreXml && (
                <div className="score-shell">
                  <ScoreView xml={scoreXml} onError={setScoreError} />
                </div>
              )}
            </div>
          )}

          {job.status === "error" && (
            <div className="banner error">
              <span>{job.error ?? "The pipeline failed."}</span>
              <button type="button" onClick={reset}>
                Start over
              </button>
            </div>
          )}
        </section>
      )}

      <footer>
        Stem separation by Demucs, melody tracking by CREPE, notation by
        music21. Sheet music renders with OSMD.
      </footer>
    </div>
  );
}
