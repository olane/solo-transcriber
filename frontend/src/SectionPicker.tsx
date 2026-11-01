import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { type Region } from "wavesurfer.js/plugins/regions";

type Props = {
  url: string;
  duration: number;
  initialStart: number;
  initialEnd: number;
  canCancel: boolean;
  onConfirm: (start: number, end: number) => Promise<void>;
  onCancel: () => void;
};

const MIN_SECTION = 2;

function clock(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

export function SectionPicker({
  url,
  duration,
  initialStart,
  initialEnd,
  canCancel,
  onConfirm,
  onCancel,
}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const wave = useRef<WaveSurfer | null>(null);
  const region = useRef<Region | null>(null);
  const initial = useRef({ start: initialStart, end: initialEnd });
  const [start, setStart] = useState(initialStart);
  const [end, setEnd] = useState(Math.max(initialEnd, initialStart + MIN_SECTION));
  const [playhead, setPlayhead] = useState(0);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const target = container.current;
    if (!target) return;
    target.innerHTML = "";
    setReady(false);
    setPlaying(false);
    setError(null);

    const ws = WaveSurfer.create({
      container: target,
      url,
      height: 104,
      waveColor: "#39465e",
      progressColor: "#f0a35e",
      cursorColor: "#f5e6d2",
      cursorWidth: 1,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
    });
    wave.current = ws;
    const regions = ws.registerPlugin(RegionsPlugin.create());

    ws.on("ready", () => {
      const total = ws.getDuration() || duration;
      const nextStart = Math.max(0, Math.min(initial.current.start, total - 0.1));
      let nextEnd = Math.min(initial.current.end || total, total);
      if (nextEnd - nextStart < MIN_SECTION) {
        nextEnd = Math.min(total, nextStart + MIN_SECTION);
      }
      region.current = regions.addRegion({
        start: nextStart,
        end: nextEnd,
        color: "rgba(240, 163, 94, 0.18)",
        drag: true,
        resize: true,
      });
      setStart(nextStart);
      setEnd(nextEnd);
      setReady(true);
      regions.enableDragSelection(
        { color: "rgba(240, 163, 94, 0.18)", drag: true },
        4,
      );
    });

    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    ws.on("timeupdate", (time) => setPlayhead(time));
    ws.on("error", (err) =>
      setError(err instanceof Error ? err.message : String(err)),
    );

    const created = (createdRegion: Region) => {
      for (const other of regions.getRegions()) {
        if (other.id !== createdRegion.id) other.remove();
      }
      region.current = createdRegion;
      setStart(createdRegion.start);
      setEnd(createdRegion.end);
    };
    const updated = (updatedRegion: Region) => {
      region.current = updatedRegion;
      setStart(updatedRegion.start);
      setEnd(updatedRegion.end);
    };
    regions.on("region-created", created);
    regions.on("region-updated", updated);

    return () => {
      regions.un("region-created", created);
      regions.un("region-updated", updated);
      ws.destroy();
      wave.current = null;
      region.current = null;
    };
  }, [url, duration]);

  function applyRange(nextStart: number, nextEnd: number) {
    const total = wave.current?.getDuration() || duration;
    let a = Math.max(0, Math.min(nextStart, total));
    let b = Math.max(0, Math.min(nextEnd, total));
    if (b < a) [a, b] = [b, a];
    setStart(a);
    setEnd(b);
    region.current?.setOptions({ start: a, end: b });
  }

  function markIn() {
    const total = wave.current?.getDuration() || duration;
    const time = Math.min(wave.current?.getCurrentTime() ?? start, total);
    const nextEnd = end <= time ? Math.min(total, time + MIN_SECTION) : end;
    applyRange(time, nextEnd);
  }

  function markOut() {
    const time = wave.current?.getCurrentTime() ?? end;
    const nextStart = start >= time ? Math.max(0, time - MIN_SECTION) : start;
    applyRange(nextStart, time);
  }

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      await onConfirm(start, end);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  const length = Math.max(0, end - start);
  const valid = ready && length >= MIN_SECTION && !submitting;

  return (
    <div className="section-picker">
      <div className="picker-head">
        <div>
          <h3>Choose the solo section</h3>
          <p>
            Drag across the waveform to select the part with the solo. Only this
            section gets separated and transcribed.
          </p>
        </div>
        {canCancel && (
          <button type="button" className="ghost" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>

      <div className="waveform-shell">
        <div className="waveform" ref={container} />
        {!ready && <div className="waveform-loading">Loading waveform...</div>}
      </div>

      <div className="picker-row">
        <div className="picker-buttons">
          <button
            type="button"
            className="secondary"
            onClick={() => wave.current?.playPause()}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => void wave.current?.play(start, end)}
          >
            Play selection
          </button>
          <span className="muted">{clock(playhead)}</span>
        </div>

        <div className="picker-buttons">
          <button type="button" className="ghost" onClick={markIn}>
            Mark in
          </button>
          <button type="button" className="ghost" onClick={markOut}>
            Mark out
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => applyRange(0, wave.current?.getDuration() || duration)}
          >
            Full track
          </button>
        </div>
      </div>

      <div className="selection-row">
        <div>
          <span className="muted">Selection</span>{" "}
          <strong>
            {clock(start)} - {clock(end)}
          </strong>{" "}
          <span className="muted">({length.toFixed(1)}s)</span>
        </div>
        <button
          type="button"
          className="primary"
          disabled={!valid}
          onClick={() => void submit()}
        >
          {submitting ? "Starting..." : "Separate this section"}
        </button>
      </div>

      {error && <div className="banner error">{error}</div>}
    </div>
  );
}
