export type Preset = "edit" | "style" | "controlnet" | "inpaint" | "edit_premium" | "ltx_video";

export interface UploadedFile {
  name: string;
  bytes: number;
}

export interface GenerateRequest {
  preset: Preset;
  prompt: string;
  negative_prompt?: string;
  input_image_name: string;
  mask_image_name?: string | null;
  reference_image_name?: string | null;
  parameters?: Record<string, unknown>;
  timeout_sec?: number;
}

export interface JobState {
  job_id: string;
  out_name: string;
  preset: string;
  status: string;
  started_at: number;
  finished_at: number | null;
  duration_ms: number | null;
  stages_ms: Record<string, number>;
  error: { code?: string; message?: string } | null;
  output_url: string | null;
  output_kind: string | null;
}

async function jsonOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status}: ${text.slice(0, 500)}`);
  }
  return (await r.json()) as T;
}

export async function uploadFile(file: File): Promise<UploadedFile> {
  const fd = new FormData();
  fd.append("file", file);
  return jsonOrThrow<UploadedFile>(await fetch("/api/upload", { method: "POST", body: fd }));
}

export async function submitGenerate(body: GenerateRequest): Promise<{ job_id: string; status: string }> {
  return jsonOrThrow(
    await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function getJob(job_id: string): Promise<JobState> {
  return jsonOrThrow<JobState>(await fetch(`/api/jobs/${job_id}`));
}
