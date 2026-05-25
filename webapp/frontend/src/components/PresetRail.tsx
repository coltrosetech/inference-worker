import { cn } from "@/lib/utils";
import type { Preset } from "@/lib/api";

type PresetMeta = {
  value: Preset;
  label: string;
  desc: string;
  group: "image" | "video";
  premium?: boolean;
};

export const PRESETS: PresetMeta[] = [
  { value: "tryon", label: "Try-on (gelinlik)", desc: "virtual try-on · IP-Adapter · hires-fix", group: "image" },
  { value: "ltx_video", label: "Video (animasyon)", desc: "image→video · LTX-Video 13B", group: "video" },
  { value: "wan_flf2v", label: "Video (giyme geçişi)", desc: "start→end · Wan 2.1 FLF2V 14B", group: "video" },
];

export function presetMeta(p: Preset): PresetMeta {
  return PRESETS.find((x) => x.value === p)!;
}

interface Props {
  selected: Preset;
  onSelect: (p: Preset) => void;
}

export function PresetRail({ selected, onSelect }: Props) {
  const sectionStyle = "px-3 pt-5 pb-1 text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80";

  return (
    <aside className="reveal reveal-delay-1 flex h-full w-full flex-col gap-0 border-r border-border bg-surface/60">
      <div className="px-5 pt-5">
        <span className="font-display text-xs italic text-muted-foreground">presets</span>
        <div className="mt-1 flex items-baseline gap-2">
          <span className="num text-[28px] leading-none tabular-nums tracking-tight">
            {String(PRESETS.length).padStart(2, "0")}
          </span>
          <span className="num text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            pipelines
          </span>
        </div>
      </div>

      <nav className="mt-3 flex-1 overflow-y-auto pb-6">
        <h3 className={sectionStyle}>image</h3>
        <ul className="space-y-px px-2">
          {PRESETS.filter((p) => p.group === "image").map((p) => (
            <PresetItem
              key={p.value}
              meta={p}
              selected={selected === p.value}
              onSelect={onSelect}
            />
          ))}
        </ul>

        <h3 className={sectionStyle}>video</h3>
        <ul className="space-y-px px-2">
          {PRESETS.filter((p) => p.group === "video").map((p) => (
            <PresetItem
              key={p.value}
              meta={p}
              selected={selected === p.value}
              onSelect={onSelect}
            />
          ))}
        </ul>
      </nav>

      <div className="border-t border-border px-5 py-3">
        <div className="font-display text-[11px] italic text-muted-foreground">
          private console · single user
        </div>
      </div>
    </aside>
  );
}

function PresetItem({
  meta,
  selected,
  onSelect,
}: {
  meta: PresetMeta;
  selected: boolean;
  onSelect: (p: Preset) => void;
}) {
  return (
    <li>
      <button
        onClick={() => onSelect(meta.value)}
        className={cn(
          "group relative flex w-full items-start gap-3 rounded-md border border-transparent px-3 py-2 text-left transition-colors duration-150",
          "hover:bg-surface-2",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/60",
          selected && "border-amber-500/20 bg-surface-2",
        )}
        style={selected ? { borderColor: "rgba(245,185,66,0.22)" } : undefined}
      >
        <span
          className={cn(
            "mt-1.5 inline-block h-1.5 w-1.5 shrink-0 rounded-full",
            selected ? "bg-primary shadow-amber-soft" : "bg-border-strong group-hover:bg-muted-foreground",
          )}
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "font-mono text-[12px] tracking-tight",
                selected ? "text-foreground" : "text-foreground/85",
              )}
            >
              {meta.label}
            </span>
            {meta.premium && (
              <span className="rounded-sm border border-border bg-surface-3 px-1 text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
                premium
              </span>
            )}
          </div>
          <div className="truncate text-[11px] text-muted-foreground">{meta.desc}</div>
        </div>
      </button>
    </li>
  );
}
