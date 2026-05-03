import { useEffect, useRef, useState } from "react";
import { ChevronsLeftRight, X, Download, Copy, CheckCircle2 } from "lucide-react";
import type { JobState } from "@/lib/api";
import { presetUrl } from "@/lib/api";

interface Props {
  open: boolean;
  job: JobState | null;
  inputName: string | null;
  onClose: () => void;
}

export function CompareLightbox({ open, job, inputName, onClose }: Props) {
  const [pos, setPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef(false);
  const [copied, setCopied] = useState(false);

  // Lock body scroll
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  // Keyboard
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 4));
      if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 4));
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open || !job || !job.output_url) return null;

  const inputUrl = inputName ? presetUrl(inputName) : null;
  const fullOutput = `${window.location.origin}${job.output_url}`;

  const isVideo = job.output_kind === "video/mp4";

  const updatePos = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = clientX - rect.left;
    const next = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPos(next);
  };

  const copy = () => {
    void navigator.clipboard.writeText(fullOutput).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-black/85 backdrop-blur-md"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="relative flex h-full w-full max-w-[1400px] flex-col py-8 px-8">
        {/* Top bar */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-baseline gap-3">
            <h2 className="font-display text-xl italic">compare</h2>
            <span className="font-mono text-[12px] text-foreground/85">{job.preset}</span>
            <span className="num text-[11px] text-muted-foreground" title={job.job_id}>
              · {job.job_id}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={copy}
              className="flex items-center gap-1.5 rounded-md border border-border bg-surface-2 px-3 py-1.5 font-mono text-[11px] hover:bg-surface-3"
            >
              {copied ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-success" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
              {copied ? "copied" : "copy link"}
            </button>
            <a
              href={job.output_url}
              download
              className="flex items-center gap-1.5 rounded-md border border-border bg-surface-2 px-3 py-1.5 font-mono text-[11px] hover:bg-surface-3"
            >
              <Download className="h-3.5 w-3.5" /> download
            </a>
            <button
              onClick={onClose}
              className="rounded-md border border-border bg-surface-2 p-1.5 hover:bg-surface-3"
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Comparison container */}
        <div className="relative flex-1 min-h-0 overflow-hidden rounded-lg border border-border bg-black">
          {isVideo || !inputUrl ? (
            // Video or no input → show output only
            isVideo ? (
              <video
                src={job.output_url}
                controls
                className="h-full w-full object-contain"
              />
            ) : (
              <img
                src={job.output_url}
                alt="output"
                className="h-full w-full object-contain"
              />
            )
          ) : (
            // Image side-by-side comparator
            <div
              ref={containerRef}
              className="relative h-full w-full select-none"
              onMouseDown={(e) => {
                dragRef.current = true;
                updatePos(e.clientX);
              }}
              onMouseMove={(e) => {
                if (dragRef.current) updatePos(e.clientX);
              }}
              onMouseUp={() => (dragRef.current = false)}
              onMouseLeave={() => (dragRef.current = false)}
              onTouchStart={(e) => {
                dragRef.current = true;
                updatePos(e.touches[0].clientX);
              }}
              onTouchMove={(e) => {
                if (dragRef.current && e.touches[0]) updatePos(e.touches[0].clientX);
              }}
              onTouchEnd={() => (dragRef.current = false)}
            >
              {/* Output (full underneath) */}
              <img
                src={job.output_url}
                alt="output"
                className="absolute inset-0 h-full w-full object-contain pointer-events-none"
                draggable={false}
              />
              {/* Input (clipped on top) */}
              <div
                className="absolute inset-0"
                style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
              >
                <img
                  src={inputUrl}
                  alt="input"
                  className="h-full w-full object-contain pointer-events-none"
                  draggable={false}
                />
              </div>

              {/* Slider line + handle */}
              <div
                className="absolute inset-y-0 w-px bg-primary shadow-amber-soft pointer-events-none"
                style={{ left: `${pos}%` }}
              >
                <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 grid h-9 w-9 place-items-center rounded-full bg-primary text-primary-foreground shadow-amber">
                  <ChevronsLeftRight className="h-4 w-4" />
                </div>
              </div>

              {/* Labels */}
              <div className="absolute top-3 left-3 rounded-md border border-border bg-background/80 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/85 backdrop-blur">
                input
              </div>
              <div className="absolute top-3 right-3 rounded-md border border-amber-500/30 bg-primary/15 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-primary backdrop-blur"
                style={{ borderColor: "rgba(245,185,66,0.32)" }}>
                output
              </div>
            </div>
          )}
        </div>

        {/* Bottom hint */}
        <div className="mt-3 flex items-center justify-between text-[10px] text-muted-foreground">
          <div className="flex items-center gap-2">
            <span>drag the handle · or use</span>
            <span className="kbd">←</span>
            <span className="kbd">→</span>
            <span>· press</span>
            <span className="kbd">esc</span>
            <span>to close</span>
          </div>
          <div className="num">
            split · {pos.toFixed(0)}%
          </div>
        </div>
      </div>
    </div>
  );
}
