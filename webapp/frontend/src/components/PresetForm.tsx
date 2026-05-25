import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import type { Preset } from "@/lib/api";

export interface PresetParams {
  prompt: string;
  negative_prompt: string;
  steps?: number;
  cfg?: number;
  strength?: number;
  seed?: number | null;
  grow_mask_px?: number;
  reference_weight?: number;
  auto_mask?: boolean;
  auto_mask_categories?: string[];
  hires_fix?: boolean;
  hires_strength?: number;
  hires_scale?: number;
  // video presets (wan_i2v image→video, wan_flf2v start→end frame)
  fps?: number;
  width?: number;
  height?: number;
  length?: number; // 4n+1
  shift?: number;
  lora_strength?: number;
}

export const DEFAULTS_BY_PRESET: Record<Preset, PresetParams> = {
  tryon: {
    prompt: "wearing an elegant white lace wedding gown, natural lighting, photorealistic",
    negative_prompt: "",
    steps: 32,
    cfg: 7.0,
    strength: 0.92,
    grow_mask_px: 20,
    reference_weight: 0.9,
    auto_mask: true,
    auto_mask_categories: ["upper_clothes", "pants", "skirt", "dress", "belt"],
    hires_fix: true,
    hires_strength: 0.25,
    hires_scale: 1.5,
  },
  wan_i2v: {
    prompt: "the woman in the wedding gown turns gently, fabric flows, soft cinematic motion, photorealistic, consistent face",
    negative_prompt: "",
    steps: 4,
    cfg: 1.0,
    shift: 5.0,
    length: 81, // 4n+1; ≈5s @ 16fps
    fps: 16,
    width: 720,
    height: 1280,
  },
  wan_flf2v: {
    prompt: "the woman puts on the jacket with a slow, gentle, smooth natural motion, cinematic, photorealistic, consistent face",
    negative_prompt: "blurry, distorted, deformed face, morphing face, extra limbs, flickering, jitter, low quality, fast chaotic motion, nudity",
    length: 49, // 4n+1; ≈4s @ 12fps (soft)
    fps: 12,
    steps: 6, // LightX2V holds at few steps; ≈32s @480x832 (4≈22s, 8≈42s)
    cfg: 1.0,
    shift: 5.0, // lower shift = identity-stable morph for near-identical pairs
    width: 480,
    height: 832,
  },
};

// ------------------- UI primitives -------------------

function Group({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="overflow-hidden rounded-md border border-border bg-surface/30">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-foreground/85 transition-colors hover:bg-surface-2"
      >
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          {title}
        </span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      {open && (
        <div className="space-y-3 border-t border-border bg-surface/40 px-3 py-3">
          {children}
        </div>
      )}
    </section>
  );
}

function SliderRow({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.01,
  hint,
  format = (v) => v.toFixed(2),
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  hint?: string;
  format?: (v: number) => string;
}) {
  const v = value ?? 0;
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[11px] text-foreground/85">{label}</span>
        <span className="num text-[11px] text-amber tabular-nums">{format(v)}</span>
      </div>
      <Slider
        value={[v]}
        min={min}
        max={max}
        step={step}
        onValueChange={(arr) => onChange(arr[0])}
      />
      {hint && <p className="text-[10px] text-muted-foreground">{hint}</p>}
    </div>
  );
}

function NumberRow({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[11px] text-foreground/85">{label}</span>
      </div>
      <input
        type="number"
        value={value ?? ""}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const n = e.target.value === "" ? NaN : Number(e.target.value);
          if (Number.isFinite(n)) onChange(n);
        }}
        className="w-full rounded-md border border-border bg-surface-2 px-2 py-1.5 font-mono text-[12px] text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
      />
      {hint && <p className="text-[10px] text-muted-foreground">{hint}</p>}
    </div>
  );
}

function ToggleRow({
  label,
  desc,
  checked,
  onChange,
}: {
  label: string;
  desc?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-3 rounded-md border border-border bg-surface-2/40 px-3 py-2">
      <div className="min-w-0">
        <div className="font-mono text-[11.5px] text-foreground/90">{label}</div>
        {desc && <div className="text-[10.5px] text-muted-foreground mt-0.5 leading-snug">{desc}</div>}
      </div>
      <Switch checked={checked} onCheckedChange={onChange} />
    </div>
  );
}

const ALL_MASK_CATEGORIES = [
  "upper_clothes",
  "pants",
  "skirt",
  "dress",
  "belt",
  "hat",
  "shoe",
  "scarf",
  "bag",
  "sunglasses",
] as const;

function MaskCategoryChips({
  selected,
  onChange,
}: {
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {ALL_MASK_CATEGORIES.map((cat) => {
        const on = selected.includes(cat);
        return (
          <button
            key={cat}
            onClick={() => {
              const next = new Set(selected);
              if (on) next.delete(cat);
              else next.add(cat);
              onChange(Array.from(next));
            }}
            className={cn(
              "rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] transition-all",
              on
                ? "border-amber-500/40 bg-primary/15 text-primary"
                : "border-border bg-surface-2 text-muted-foreground hover:bg-surface-3 hover:text-foreground",
            )}
            style={on ? { borderColor: "rgba(245,185,66,0.42)" } : undefined}
          >
            {cat.replace("_", " ")}
          </button>
        );
      })}
    </div>
  );
}

// ------------------- Main form -------------------

export function PresetForm({
  preset,
  params,
  setParams,
}: {
  preset: Preset;
  params: PresetParams;
  setParams: (p: PresetParams) => void;
}) {
  const u = <K extends keyof PresetParams>(k: K, v: PresetParams[K]) =>
    setParams({ ...params, [k]: v });

  if (preset === "wan_flf2v") {
    return (
      <div className="space-y-2.5">
        <Group title="video · start→end" defaultOpen>
          <p className="text-[10px] text-muted-foreground">
            <b>input</b> = başlangıç karesi (ceketsiz orijinal), <b>reference</b> = bitiş karesi
            (giyinik try-on çıktısı). Wan ikisi arasını canlandırır.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <NumberRow
              label="length (4n+1)"
              value={params.length}
              onChange={(v) => u("length", v)}
              min={5}
              max={205}
              step={4}
              hint={`≈${((params.length || 49) / (params.fps || 12)).toFixed(1)}s @ ${params.fps || 12}fps`}
            />
            <NumberRow label="fps" value={params.fps} onChange={(v) => u("fps", v)} min={8} max={30} />
            <NumberRow label="width (÷16)" value={params.width} onChange={(v) => u("width", v)} min={256} max={1280} step={16} />
            <NumberRow label="height (÷16)" value={params.height} onChange={(v) => u("height", v)} min={256} max={1280} step={16} />
          </div>
        </Group>

        <Group title="sampling">
          <div className="grid grid-cols-2 gap-3">
            <NumberRow label="steps" value={params.steps} onChange={(v) => u("steps", v)} min={1} max={60} />
            <NumberRow label="cfg" value={params.cfg} onChange={(v) => u("cfg", v)} min={0} max={15} step={0.1} />
          </div>
          <SliderRow
            label="shift"
            value={params.shift}
            onChange={(v) => u("shift", v)}
            min={1}
            max={12}
            step={0.5}
            hint="motion/temporal shift"
          />
        </Group>
      </div>
    );
  }

  if (preset === "wan_i2v") {
    return (
      <div className="space-y-2.5">
        <Group title="video · image→video" defaultOpen>
          <div className="grid grid-cols-2 gap-3">
            <NumberRow
              label="length (4n+1)"
              value={params.length}
              onChange={(v) => u("length", v)}
              min={5}
              max={205}
              step={4}
              hint={`≈${((params.length || 81) / (params.fps || 16)).toFixed(1)}s @ ${params.fps || 16}fps`}
            />
            <NumberRow label="fps" value={params.fps} onChange={(v) => u("fps", v)} min={8} max={30} />
            <NumberRow label="width (÷16)" value={params.width} onChange={(v) => u("width", v)} min={256} max={1280} step={16} />
            <NumberRow label="height (÷16)" value={params.height} onChange={(v) => u("height", v)} min={256} max={1280} step={16} />
          </div>
          <p className="text-[10px] text-muted-foreground">
            Girdi olarak try-on çıktısını yükle — Wan 2.2 I2V onu canlandırır.
          </p>
        </Group>

        <Group title="sampling">
          <div className="grid grid-cols-2 gap-3">
            <NumberRow label="steps" value={params.steps} onChange={(v) => u("steps", v)} min={2} max={60} />
            <NumberRow label="cfg" value={params.cfg} onChange={(v) => u("cfg", v)} min={0} max={15} step={0.1} />
          </div>
          <SliderRow
            label="shift"
            value={params.shift}
            onChange={(v) => u("shift", v)}
            min={1}
            max={12}
            step={0.5}
            hint="temporal shift (4-step LightX2V → cfg≈1)"
          />
        </Group>
      </div>
    );
  }

  // tryon
  return (
    <div className="space-y-2.5">
      <Group title="sampling" defaultOpen>
        <div className="grid grid-cols-2 gap-3">
          <NumberRow label="steps" value={params.steps} onChange={(v) => u("steps", v)} min={1} max={60} />
          <NumberRow label="cfg" value={params.cfg} onChange={(v) => u("cfg", v)} min={0} max={20} step={0.1} />
          <SliderRow label="strength" value={params.strength} onChange={(v) => u("strength", v)} min={0} max={1} />
          <SliderRow
            label="reference_weight"
            value={params.reference_weight}
            onChange={(v) => u("reference_weight", v)}
            min={0}
            max={2}
            step={0.05}
            hint="how strongly the result follows the garment reference (IP-Adapter)"
          />
        </div>
      </Group>

      <Group title="mask" defaultOpen>
        <ToggleRow
          label="auto-mask clothing"
          desc="SegFormer-B2 detects the current outfit — no manual mask required."
          checked={!!params.auto_mask}
          onChange={(v) => u("auto_mask", v)}
        />
        {params.auto_mask && (
          <div className="space-y-1.5">
            <div className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-muted-foreground">
              regions to replace
            </div>
            <MaskCategoryChips
              selected={params.auto_mask_categories ?? []}
              onChange={(next) => u("auto_mask_categories", next)}
            />
          </div>
        )}
        <SliderRow
          label="grow_mask_px"
          value={params.grow_mask_px}
          onChange={(v) => u("grow_mask_px", Math.round(v))}
          min={0}
          max={64}
          step={1}
          format={(v) => `${Math.round(v)}px`}
        />
      </Group>

      <Group title="quality · hires-fix" defaultOpen>
        <ToggleRow
          label="hires_fix"
          desc="4×-UltraSharp upscale + low-denoise resample — sharpens fabric/detail, keeps identity."
          checked={!!params.hires_fix}
          onChange={(v) => u("hires_fix", v)}
        />
        {params.hires_fix && (
          <>
            <SliderRow
              label="hires_strength"
              value={params.hires_strength}
              onChange={(v) => u("hires_strength", v)}
              min={0}
              max={0.6}
            />
            <SliderRow
              label="hires_scale"
              value={params.hires_scale}
              onChange={(v) => u("hires_scale", v)}
              min={1.0}
              max={2.0}
              step={0.05}
              format={(v) => `${v.toFixed(2)}×`}
            />
          </>
        )}
      </Group>
    </div>
  );
}
