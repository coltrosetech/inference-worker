import { useEffect, useState } from "react";
import {
  CheckCircle2,
  Copy,
  Download,
  ExternalLink,
  Repeat,
  XCircle,
  Loader2,
  Maximize2,
} from "lucide-react";
import { getJob, getProgress, type JobState, type SamplingProgress, presetUrl } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  jobId: string | null;
  inputName: string | null;
  onRetry?: () => void;
  onLightbox?: (job: JobState, inputName: string | null) => void;
}

export function OutputViewer({ jobId, inputName, onRetry, onLightbox }: Props) {
  const [job, setJob] = useState<JobState | null>(null);
  const [prog, setProg] = useState<SamplingProgress | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setProg(null);
      return;
    }
    let stopped = false;
    const fetchJob = async () => {
      try {
        const j = await getJob(jobId);
        if (!stopped) setJob(j);
        if (j.status === "success" || j.status === "failed") {
          if (!stopped) setProg(null);
          return;
        }
      } catch {
        /* ignore */
      }
      try {
        const p = await getProgress();
        if (!stopped) setProg(p);
      } catch {
        /* ignore */
      }
      if (!stopped) setTimeout(fetchJob, 1000);
    };
    fetchJob();
    return () => {
      stopped = true;
    };
  }, [jobId]);

  if (!jobId || !job) {
    return (
      <section className="reveal reveal-delay-2 flex h-full flex-col items-center justify-center text-center px-8 py-16">
        <div className="grid h-12 w-12 place-items-center rounded-full bg-surface-2 ring-1 ring-border">
          <Maximize2 className="h-4 w-4 text-muted-foreground" />
        </div>
        <h2 className="mt-4 font-display italic text-lg">no output yet</h2>
        <p className="mt-1 text-[12px] text-muted-foreground max-w-xs">
          Submit a job from the panel on the left, or pick one from the queue to inspect.
        </p>
      </section>
    );
  }

  const isDone = job.status === "success";
  const isFailed = job.status === "failed";
  const isRunning = job.status === "running";

  const fullUrl = job.output_url ? `${window.location.origin}${job.output_url}` : null;

  const copy = () => {
    if (!fullUrl) return;
    void navigator.clipboard.writeText(fullUrl).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <section className="reveal reveal-delay-2 flex h-full flex-col">
      <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-3">
        <div className="flex items-baseline gap-2 min-w-0">
          <h2 className="font-display text-[14px] italic">output</h2>
          <span className="font-mono text-[11px] text-foreground/85 truncate">{job.preset}</span>
          <span className="num text-[10px] text-muted-foreground truncate" title={job.job_id}>
            · {job.job_id}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {job.duration_ms != null && (
            <span className="num text-[11px] text-muted-foreground">
              {(job.duration_ms / 1000).toFixed(1)}s
            </span>
          )}
          {isDone && (
            <StatusPill tone="success" label="done" />
          )}
          {isFailed && <StatusPill tone="danger" label="failed" />}
          {isRunning && <StatusPill tone="warn" label="running" pulse />}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto bg-black/30 p-5">
        {isFailed && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/5 p-4">
            <div className="flex items-center gap-2 text-destructive">
              <XCircle className="h-4 w-4" />
              <span className="font-mono text-sm">
                {job.error?.code ?? "FAILED"}
              </span>
            </div>
            <p className="mt-2 text-[12px] text-foreground/80">
              {job.error?.message ?? "unknown error"}
            </p>
          </div>
        )}

        {isRunning && (
          <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
            <Loader2 className="h-5 w-5 animate-spin text-warning" />
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-warning">
              inference in progress
            </div>
            {prog && prog.step != null && prog.total != null ? (
              <div className="w-64 space-y-1">
                <div className="flex items-baseline justify-between font-mono text-[11px]">
                  <span className="text-foreground/85">step {prog.step}/{prog.total}</span>
                  {prog.detail && (
                    <span className="text-muted-foreground tabular-nums">{prog.detail}</span>
                  )}
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
                  <div
                    className="h-full rounded-full bg-primary transition-all"
                    style={{ width: `${Math.round((prog.step / prog.total) * 100)}%` }}
                  />
                </div>
              </div>
            ) : (
              <ShimmerBar />
            )}
            {Object.keys(job.stages_ms || {}).length > 0 && (
              <div className="flex flex-wrap items-center justify-center gap-1.5 mt-2">
                {Object.entries(job.stages_ms).map(([k, ms]) => (
                  <span
                    key={k}
                    className="rounded border border-border bg-surface px-2 py-0.5 font-mono text-[10px] text-muted-foreground"
                  >
                    {k}:{ms}ms
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {isDone && job.output_url && (
          <div className="flex flex-col items-center gap-4">
            <button
              onClick={() => onLightbox?.(job, inputName)}
              className="group relative inline-block max-h-[68vh] overflow-hidden rounded-md border border-border bg-black"
              title="Click to compare with input"
            >
              {job.output_kind === "video/mp4" ? (
                <video
                  src={job.output_url}
                  controls
                  className="block max-h-[68vh] max-w-full"
                />
              ) : (
                <img
                  src={job.output_url}
                  alt="output"
                  className="block max-h-[68vh] max-w-full object-contain"
                  loading="lazy"
                />
              )}
              <div className="absolute right-2 top-2 flex items-center gap-1 rounded-md bg-background/85 px-2 py-1 text-[10px] text-muted-foreground opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
                <Maximize2 className="h-3 w-3" /> compare
              </div>
            </button>

            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                onClick={copy}
                className={cn(
                  "flex items-center gap-1.5 rounded-md border border-border bg-surface-2 px-3 py-1.5 font-mono text-[11px] transition-colors",
                  "hover:bg-surface-3",
                )}
              >
                {copied ? <CheckCircle2 className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
                {copied ? "copied" : "copy link"}
              </button>
              <a
                href={job.output_url}
                download
                className="flex items-center gap-1.5 rounded-md border border-border bg-surface-2 px-3 py-1.5 font-mono text-[11px] transition-colors hover:bg-surface-3"
              >
                <Download className="h-3.5 w-3.5" /> download
              </a>
              <a
                href={job.output_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 rounded-md border border-border bg-surface-2 px-3 py-1.5 font-mono text-[11px] transition-colors hover:bg-surface-3"
              >
                <ExternalLink className="h-3.5 w-3.5" /> open
              </a>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="flex items-center gap-1.5 rounded-md border border-amber-500/30 bg-primary/10 px-3 py-1.5 font-mono text-[11px] text-primary transition-colors hover:bg-primary/20"
                  style={{ borderColor: "rgba(245,185,66,0.32)" }}
                >
                  <Repeat className="h-3.5 w-3.5" /> retry · new seed
                </button>
              )}
            </div>

            {Object.keys(job.stages_ms || {}).length > 0 && (
              <div className="flex flex-wrap justify-center gap-1.5 text-[10px]">
                {Object.entries(job.stages_ms).map(([k, ms]) => (
                  <span
                    key={k}
                    className="rounded border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-muted-foreground"
                  >
                    {k}:{ms}ms
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Mini-strip showing the input image when both are available */}
      {inputName && isDone && job.output_url && (
        <div className="border-t border-border px-5 py-2.5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              input source
            </span>
            <span className="num text-[10px] text-muted-foreground truncate max-w-[40%]">
              {inputName}
            </span>
          </div>
          <div className="mt-2 flex gap-2">
            <img
              src={presetUrl(inputName)}
              alt="input"
              className="h-16 w-16 rounded border border-border object-cover"
            />
            <div className="flex flex-1 items-center justify-center text-[10px] text-muted-foreground">
              — click image above to slide-compare —
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function StatusPill({
  tone,
  label,
  pulse,
}: {
  tone: "success" | "danger" | "warn";
  label: string;
  pulse?: boolean;
}) {
  const cls =
    tone === "success"
      ? "text-success"
      : tone === "danger"
      ? "text-destructive"
      : "text-warning";
  return (
    <span className="flex items-center gap-1.5 rounded-md border border-border bg-surface-2 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.18em]">
      <span
        className={cn("status-dot", pulse && "status-dot-pulse", cls)}
        style={{ backgroundColor: "currentColor" }}
      />
      <span className={cls}>{label}</span>
    </span>
  );
}

function ShimmerBar() {
  return (
    <div className="relative h-1 w-48 overflow-hidden rounded-full bg-surface-3">
      <div
        className="absolute inset-y-0 -left-1/3 w-1/3 animate-shimmer rounded-full"
        style={{
          background: "linear-gradient(90deg, transparent, rgba(245,185,66,0.6), transparent)",
          backgroundSize: "200% 100%",
        }}
      />
    </div>
  );
}
