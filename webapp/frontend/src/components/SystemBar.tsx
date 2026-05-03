import { useEffect, useState } from "react";
import { CircleDot, Cpu, Layers } from "lucide-react";
import { getWorkerHealth, type WorkerHealth } from "@/lib/api";
import { cn } from "@/lib/utils";

function formatUptime(s: number): string {
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
}

export function SystemBar() {
  const [h, setH] = useState<WorkerHealth | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let stopped = false;
    const fetchOnce = async () => {
      try {
        const r = await getWorkerHealth();
        if (!stopped) setH(r);
      } catch {
        if (!stopped) setH({ ok: false, ready: false });
      }
    };
    fetchOnce();
    const id = setInterval(fetchOnce, 4000);
    const t = setInterval(() => setTick((x) => x + 1), 1000);
    return () => {
      stopped = true;
      clearInterval(id);
      clearInterval(t);
    };
  }, []);

  const ready = h?.ready === true;
  const busy = (h?.queue_depth ?? 0) > 0 || h?.current_job;
  const offline = !h?.ok;

  const dotClass = offline
    ? "text-destructive"
    : busy
    ? "text-warning"
    : ready
    ? "text-success"
    : "text-muted-foreground";

  const stateLabel = offline
    ? "offline"
    : busy
    ? "busy"
    : ready
    ? "ready"
    : "warming";

  const vramPct = h?.vram_total_gb && h?.vram_used_gb != null ? (h.vram_used_gb / h.vram_total_gb) * 100 : null;

  return (
    <header className="sticky top-0 z-30 border-b border-border/70 bg-background/85 backdrop-blur-xl">
      <div className="flex h-12 items-center gap-6 px-6">
        {/* Brand mark */}
        <div className="flex items-center gap-2.5">
          <div className="relative grid h-6 w-6 place-items-center rounded-md bg-surface-2 shadow-inset">
            <span className="absolute inset-0 rounded-md ring-1 ring-inset ring-amber-500/20" style={{ "--tw-ring-color": "rgba(245,185,66,0.22)" } as React.CSSProperties} />
            <span className="num text-[10px] font-medium text-amber">A</span>
          </div>
          <div className="flex items-baseline gap-2 leading-none">
            <span className="font-display italic text-[15px] tracking-tight">atelier</span>
            <span className="num text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              inference console
            </span>
          </div>
        </div>

        {/* Worker status */}
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "status-dot",
              !offline && "status-dot-pulse",
              dotClass,
            )}
            style={{ backgroundColor: "currentColor" }}
            aria-hidden
          />
          <span className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            worker
          </span>
          <span className="num text-xs font-medium">
            {stateLabel}
          </span>
        </div>

        <div className="hidden items-center gap-1 text-[11px] text-muted-foreground md:flex">
          <Layers className="h-3.5 w-3.5" />
          <span className="num">queue {h?.queue_depth ?? 0}</span>
        </div>

        {h?.gpu && (
          <div className="hidden items-center gap-1.5 text-[11px] text-muted-foreground lg:flex">
            <Cpu className="h-3.5 w-3.5" />
            <span className="num truncate max-w-[180px]" title={h.gpu}>
              {h.gpu.replace("NVIDIA GeForce ", "")}
            </span>
            {vramPct != null && (
              <span className="num ml-1 tabular-nums">
                {h?.vram_used_gb?.toFixed(1)}/{h?.vram_total_gb?.toFixed(0)}GB
              </span>
            )}
          </div>
        )}

        <div className="flex-1" />

        <div className="hidden items-center gap-2 text-[11px] text-muted-foreground sm:flex">
          <CircleDot className="h-3 w-3" />
          <span className="num">{h?.uptime_sec ? formatUptime(h.uptime_sec + tick) : "--"}</span>
        </div>

        {h?.version && (
          <span className="num text-[10px] text-muted-foreground">v{h.version}</span>
        )}
      </div>
    </header>
  );
}
