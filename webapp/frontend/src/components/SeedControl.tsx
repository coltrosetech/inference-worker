import { Dice5, Lock, RotateCw, Unlock } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  seed: number | null;
  onChange: (seed: number | null) => void;
}

function randSeed() {
  // 53-bit safe int range; ample variety
  return Math.floor(Math.random() * 9_007_199_254_740_991);
}

export function SeedControl({ seed, onChange }: Props) {
  const locked = seed !== null && seed !== undefined && Number.isFinite(seed);

  return (
    <div className="flex items-stretch gap-1 rounded-md border border-border bg-surface-2 p-1">
      <button
        onClick={() => onChange(locked ? null : randSeed())}
        className={cn(
          "flex h-7 items-center gap-1.5 rounded px-2 text-[11px] uppercase tracking-[0.18em] transition-colors",
          locked
            ? "bg-primary/15 text-primary"
            : "text-muted-foreground hover:text-foreground hover:bg-surface-3",
        )}
        title={locked ? "Lock seed for reproducibility" : "Random each run"}
      >
        {locked ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
        <span>{locked ? "lock" : "random"}</span>
      </button>

      {locked && (
        <input
          type="number"
          value={seed ?? ""}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "") {
              onChange(null);
            } else {
              const n = Number(v);
              if (Number.isFinite(n)) onChange(n);
            }
          }}
          className={cn(
            "num w-32 bg-transparent px-2 text-[12px] tabular-nums",
            "outline-none placeholder:text-muted-foreground",
            "border-l border-border",
          )}
          placeholder="0"
        />
      )}

      <button
        onClick={() => onChange(randSeed())}
        title="Roll a new seed"
        className="flex h-7 items-center gap-1 rounded px-2 text-muted-foreground hover:bg-surface-3 hover:text-foreground transition-colors"
      >
        <Dice5 className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => onChange(null)}
        title="Reset to random"
        disabled={!locked}
        className={cn(
          "flex h-7 items-center gap-1 rounded px-2 transition-colors",
          locked
            ? "text-muted-foreground hover:bg-surface-3 hover:text-foreground"
            : "text-muted-foreground/40 cursor-not-allowed",
        )}
      >
        <RotateCw className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
