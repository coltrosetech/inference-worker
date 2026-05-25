export type Preset = "tryon" | "wan_i2v" | "wan_flf2v";

export interface UploadedFile {
  name: string;
  bytes: number;
  bucket?: [number, number] | null;
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

export interface HealthState {
  ok: boolean;
  worker_url: string;
  jobs_active: number;
}

export interface WorkerHealth {
  ok: boolean;
  ready: boolean;
  worker_id?: string;
  gpu?: string;
  vram_used_gb?: number;
  vram_total_gb?: number;
  gpu_util_ratio?: number;
  queue_depth?: number;
  current_job?: string | null;
  comfyui_alive?: boolean;
  uptime_sec?: number;
  version?: string;
  error?: string;
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

export async function submitGenerate(
  body: GenerateRequest,
): Promise<{ job_id: string; status: string }> {
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

export async function listJobs(limit = 20): Promise<JobState[]> {
  return jsonOrThrow<JobState[]>(await fetch(`/api/jobs?limit=${limit}`));
}

export interface SamplingProgress {
  running: number;
  step: number | null;
  total: number | null;
  detail: string;
}

export async function getProgress(): Promise<SamplingProgress> {
  return jsonOrThrow<SamplingProgress>(await fetch(`/api/progress`));
}

export async function getHealth(): Promise<HealthState> {
  return jsonOrThrow<HealthState>(await fetch(`/api/health`));
}

export async function getWorkerHealth(): Promise<WorkerHealth> {
  const r = await fetch(`/api/worker-health`);
  return (await r.json()) as WorkerHealth;
}

/** Worker direct health probe (proxied through webapp's WORKER_URL setting is
 *  internal; we use webapp /api/health which reflects worker state coarsely). */
export const presetUrl = (name: string) => `/u/${name}`;
export const outputUrl = (name: string) => `/o/${name}`;
