import { Loader2, Play } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  busy: boolean;
  disabled: boolean;
  onClick: () => void;
  hint: string | null;
}

export function GenerateButton({ busy, disabled, onClick, hint }: Props) {
  return (
    <div className="flex flex-col items-stretch gap-1">
      <button
        onClick={onClick}
        disabled={disabled}
        className={cn(
          "group relative flex h-11 items-center justify-center gap-2 overflow-hidden rounded-lg",
          "border border-amber-500/30 bg-primary/15 px-5 font-mono text-[12px] uppercase tracking-[0.22em] text-primary",
          "transition-all hover:bg-primary/25 hover:shadow-amber",
          "active:scale-[0.99]",
          "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:shadow-none",
          "focus-visible:outline-none focus-visible:shadow-amber",
        )}
        style={{ borderColor: "rgba(245,185,66,0.32)" }}
      >
        {busy ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Play className="h-4 w-4" />
        )}
        <span>{busy ? "running…" : "generate"}</span>
        <span className="num text-[10px] opacity-60 ml-1">⌘↵</span>
        {!disabled && !busy && (
          <span className="pointer-events-none absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-amber-300/15 to-transparent transition-transform duration-700 ease-out group-hover:translate-x-full" />
        )}
      </button>
      {hint && (
        <span className="text-[11px] text-muted-foreground font-mono">
          {hint}
        </span>
      )}
    </div>
  );
}
