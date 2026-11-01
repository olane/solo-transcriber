export type JobStatus =
  | "queued"
  | "preparing"
  | "awaiting_section"
  | "separating"
  | "awaiting_stem"
  | "transcribing"
  | "done"
  | "error";

export type Instrument = "concert" | "soprano" | "alto" | "tenor" | "bari";
export type ScoreFormat = "musicxml" | "midi";

export type Stem = {
  name: string;
  url: string;
  suggested: boolean;
};

export type Section = {
  start: number;
  end: number;
};

export type Transcription = {
  bpm: number;
  note_count: number;
  duration: number;
  first_note: number;
  mock: boolean;
};

export type Job = {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  error: string | null;
  filename: string;
  duration: number;
  stems: Stem[];
  selected_stem: string | null;
  section: Section | null;
  preview_url: string | null;
  transcription: Transcription | null;
  mock: boolean;
};

async function failure(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // fall through
  }
  return `${response.status} ${response.statusText}`;
}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error(await failure(response));
  return (await response.json()) as T;
}

export async function createJob(file: File): Promise<Job> {
  const body = new FormData();
  body.append("file", file);
  return parse<Job>(await fetch("/api/jobs", { method: "POST", body }));
}

export async function getJob(id: string): Promise<Job> {
  return parse<Job>(await fetch(`/api/jobs/${id}`));
}

export async function separateSection(
  id: string,
  start: number,
  end: number,
): Promise<Job> {
  return parse<Job>(
    await fetch(`/api/jobs/${id}/separate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start, end }),
    }),
  );
}

export async function transcribe(id: string, stem: string): Promise<Job> {
  return parse<Job>(
    await fetch(`/api/jobs/${id}/transcribe`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stem }),
    }),
  );
}

export function scoreUrl(
  id: string,
  instrument: Instrument,
  timeSignature: string,
  bpm: number,
  format: ScoreFormat,
): string {
  const params = new URLSearchParams({
    instrument,
    time_signature: timeSignature,
    bpm: bpm.toFixed(1),
    format,
  });
  return `/api/jobs/${id}/score?${params.toString()}`;
}
