import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Clock, Image as ImgIcon, Loader2, XCircle, Repeat } from "lucide-react";
import { listJobs, type JobState } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  onSelect: (job: JobState) => void;
  selectedId: string | null;
  onRetry?: (job: JobState) => void;
}

function relTime(epochSec: number, now: number): string {
  const dt = Math.max(0, now - epochSec);
  if (dt < 5) return "now";
  if (dt < 60) return `${Math.floor(dt)}s`;
  if (dt < 3600) return `${Math.floor(dt / 60)}m`;
  return `${Math.floor(dt / 3600)}h`;
}

export function JobQueue({ onSelect, selectedId, onRetry }: Props) {
  const [jobs, setJobs] = useState<JobState[]>([]);
  const [now, setNow] = useState(Date.now() / 1000);
  const wasRunning = useRef<Set<string>>(new Set());

  useEffect(() => {
    let stopped = false;
    const fetchOnce = async () => {
      try {
        const list = await listJobs(20);
        if (!stopped) setJobs(list);
      } catch {
        /* ignore */
      }
    };
    fetchOnce();
    const id = setInterval(fetchOnce, 2000);
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => {
      stopped = true;
      clearInterval(id);
      clearInterval(tick);
    };
  }, []);

  // Track which were running so we can highlight new arrivals (slide-in)
  useEffect(() => {
    const cur = new Set(jobs.filter((j) => j.status === "running").map((j) => j.job_id));
    wasRunning.current = cur;
  }, [jobs]);

  const empty = jobs.length === 0;

  return (
    <section className="reveal reveal-delay-3 flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <div className="flex items-baseline gap-2">
          <h2 className="font-display text-[14px] italic">queue</h2>
          <span className="num text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            live · 2s poll
          </span>
        </div>
        <span className="num text-[10px] text-muted-foreground">
          {jobs.length}
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {empty && (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center">
            <ImgIcon className="h-5 w-5 text-muted-foreground" />
            <div className="text-[11px] text-muted-foreground">
              no jobs yet — submit one
            </div>
          </div>
        )}

        <ul className="space-y-1">
          {jobs.map((j) => {
            const status = j.status;
            const selected = j.job_id === selectedId;
            const isDone = status === "success";
            const isFailed = status === "failed";
            const isRunning = status === "running";

            return (
              <li key={j.job_id}>
                <button
                  onClick={() => onSelect(j)}
                  className={cn(
                    "group flex w-full items-center gap-2.5 rounded-md border border-transparent px-2.5 py-2 text-left transition-all",
                    "hover:bg-surface-2",
                    selected && "bg-surface-2 border-amber-500/20",
                  )}
                  style={selected ? { borderColor: "rgba(245,185,66,0.22)" } : undefined}
                >
                  <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-border bg-surface-3">
                    {isDone && <CheckCircle2 className="h-3.5 w-3.5 text-success" />}
                    {isFailed && <XCircle className="h-3.5 w-3.5 text-destructive" />}
                    {isRunning && <Loader2 className="h-3.5 w-3.5 animate-spin text-warning" />}
                    {!isDone && !isFailed && !isRunning && (
                      <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                    )}
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <span className="font-mono text-[11px] truncate text-foreground/95">
                        {j.preset}
                      </span>
                      {isRunning && (
                        <span className="num text-[10px] text-warning uppercase tracking-[0.18em]">
                          running
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 flex items-center gap-2 text-[10px] text-muted-foreground">
                      <span className="num truncate" title={j.job_id}>
                        {j.job_id.replace("web_", "")}
                      </span>
                      <span>·</span>
                      <span className="num">
                        {j.duration_ms != null
                          ? `${(j.duration_ms / 1000).toFixed(1)}s`
                          : relTime(j.started_at, now)}
                      </span>
                    </div>
                  </div>

                  {onRetry && isDone && (
                    <span
                      role="button"
                      tabIndex={0}
                      title="Retry with new seed"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRetry(j);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.stopPropagation();
                          onRetry(j);
                        }
                      }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity rounded p-1 text-muted-foreground hover:bg-surface-3 hover:text-foreground"
                    >
                      <Repeat className="h-3 w-3" />
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
