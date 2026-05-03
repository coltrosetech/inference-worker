import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { Preset } from "@/lib/api";

export interface PresetParams {
  prompt: string;
  negative_prompt: string;
  steps?: number;
  cfg?: number;
  strength?: number;
  seed?: number | null;
  width?: number;
  height?: number;
  // style
  style_strength?: number;
  // controlnet
  controlnet_type?: "canny" | "depth" | "pose" | "lineart" | "scribble";
  controlnet_strength?: number;
  // inpaint family
  grow_mask_px?: number;
  feather_mask_px?: number;
  two_pass?: boolean;
  skin_prompt?: string;
  structural_refiner?: boolean;
  refiner_strength?: number;
  auto_mask?: boolean;
  auto_mask_categories?: string[];
  // inpaint_realvis quality stack
  lora_detail_weight?: number;
  lora_skin_weight?: number;
  lora_anatomy_weight?: number;
  bust_emphasis?: number;
  face_detailer?: boolean;
  hand_detailer?: boolean;
  person_detailer?: boolean;
  person_detailer_strength?: number;
  hires_fix?: boolean;
  hires_strength?: number;
  hires_scale?: number;
  preserve_face?: boolean;
  // edit_premium
  guidance?: number;
  // inpaint_premium pose
  use_pose_guide?: boolean;
  pose_strength?: number;
  // tryon
  reference_weight?: number;
  // ltx_video
  num_frames?: 25 | 49 | 97 | 121;
  fps?: number;
}

const SHARED_REALVIS_DEFAULTS = {
  steps: 32,
  cfg: 5.5,
  strength: 0.92,
  grow_mask_px: 16,
  feather_mask_px: 16,
  two_pass: false,
  skin_prompt: "bare natural skin, torso, arms, body, soft even lighting, anatomy",
  structural_refiner: false,
  refiner_strength: 0.3,
  auto_mask: true,
  auto_mask_categories: ["upper_clothes", "pants", "skirt", "dress", "belt"],
  lora_detail_weight: 0.3,
  lora_skin_weight: 0.25,
  lora_anatomy_weight: 0.3,
  bust_emphasis: 0.0,
  face_detailer: false,
  hand_detailer: true,
  person_detailer: true,
  person_detailer_strength: 0.32,
  hires_fix: true,
  hires_strength: 0.22,
  hires_scale: 1.5,
  preserve_face: true,
};

export const DEFAULTS_BY_PRESET: Record<Preset, PresetParams> = {
  edit: { prompt: "", negative_prompt: "", steps: 6, cfg: 1.8, strength: 0.7, width: 1024, height: 1024 },
  style: { prompt: "", negative_prompt: "", steps: 6, cfg: 1.8, strength: 0.6, style_strength: 0.7 },
  controlnet: {
    prompt: "",
    negative_prompt: "",
    steps: 6,
    cfg: 1.8,
    strength: 0.7,
    controlnet_type: "canny",
    controlnet_strength: 0.8,
  },
  inpaint: {
    prompt: "",
    negative_prompt: "",
    steps: 6,
    cfg: 1.8,
    strength: 0.9,
    grow_mask_px: 8,
    two_pass: false,
    skin_prompt: "bare natural skin, torso, arms, body, soft even lighting, anatomy",
    structural_refiner: false,
    refiner_strength: 0.3,
    auto_mask: false,
    auto_mask_categories: ["upper_clothes", "pants", "skirt", "dress", "belt"],
  },
  inpaint_sdxl: {
    prompt: "",
    negative_prompt: "",
    steps: 25,
    cfg: 7.0,
    strength: 0.9,
    grow_mask_px: 10,
    two_pass: false,
    skin_prompt: "bare natural skin, torso, arms, body, soft even lighting, anatomy",
    structural_refiner: false,
    refiner_strength: 0.3,
    auto_mask: true,
    auto_mask_categories: ["upper_clothes", "pants", "skirt", "dress", "belt"],
  },
  inpaint_realvis: {
    prompt: "",
    negative_prompt: "",
    ...SHARED_REALVIS_DEFAULTS,
  },
  tryon: {
    prompt: "wearing the reference garment, natural lighting, photorealistic",
    negative_prompt: "",
    steps: 25,
    cfg: 7.0,
    strength: 0.9,
    grow_mask_px: 12,
    reference_weight: 0.9,
    auto_mask: true,
    auto_mask_categories: ["upper_clothes", "pants", "skirt", "dress", "belt"],
  },
  edit_premium: { prompt: "", negative_prompt: "", steps: 20, cfg: 1.0, guidance: 2.5 },
  inpaint_premium: {
    prompt: "",
    negative_prompt: "",
    steps: 20,
    cfg: 1.0,
    guidance: 30.0,
    grow_mask_px: 12,
    auto_mask: true,
    auto_mask_categories: ["upper_clothes", "pants", "skirt", "dress", "belt"],
    use_pose_guide: false,
    pose_strength: 0.5,
  },
  ltx_video: {
    prompt: "",
    negative_prompt: "",
    steps: 8,
    cfg: 3.0,
    num_frames: 97,
    fps: 24,
    width: 768,
    height: 512,
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
      {hint && (
        <p className="text-[10px] text-muted-foreground">{hint}</p>
      )}
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

  const isInpaint =
    preset === "inpaint" ||
    preset === "inpaint_sdxl" ||
    preset === "inpaint_realvis" ||
    preset === "inpaint_premium" ||
    preset === "tryon";

  const isRealvis = preset === "inpaint_realvis";

  return (
    <div className="space-y-2.5">
      {/* Sampling */}
      <Group title="sampling" defaultOpen>
        <div className="grid grid-cols-2 gap-3">
          <NumberRow
            label="steps"
            value={params.steps}
            onChange={(v) => u("steps", v)}
            min={1}
            max={preset === "ltx_video" ? 100 : 60}
          />
          <NumberRow
            label={preset === "edit_premium" || preset === "inpaint_premium" || preset === "ltx_video" ? "cfg" : "cfg"}
            value={params.cfg}
            onChange={(v) => u("cfg", v)}
            min={0}
            max={20}
            step={0.1}
          />
          {(preset === "edit" ||
            preset === "style" ||
            preset === "controlnet" ||
            preset === "inpaint" ||
            preset === "inpaint_sdxl" ||
            preset === "inpaint_realvis" ||
            preset === "tryon") && (
            <SliderRow
              label="strength"
              value={params.strength}
              onChange={(v) => u("strength", v)}
              min={0}
              max={1}
            />
          )}
          {preset === "edit_premium" && (
            <SliderRow
              label="guidance"
              value={params.guidance}
              onChange={(v) => u("guidance", v)}
              min={0}
              max={10}
              step={0.1}
            />
          )}
        </div>

        {preset === "inpaint_premium" && (
          <SliderRow
            label="guidance (FLUX-Fill)"
            value={params.guidance}
            onChange={(v) => u("guidance", v)}
            min={0}
            max={100}
            step={0.5}
            format={(v) => v.toFixed(1)}
          />
        )}

        {preset === "style" && (
          <SliderRow
            label="style_strength"
            value={params.style_strength}
            onChange={(v) => u("style_strength", v)}
            min={0}
            max={1.5}
          />
        )}

        {preset === "tryon" && (
          <SliderRow
            label="reference_weight"
            value={params.reference_weight}
            onChange={(v) => u("reference_weight", v)}
            min={0}
            max={2}
            step={0.05}
          />
        )}

        {preset === "controlnet" && (
          <>
            <div className="space-y-1">
              <span className="font-mono text-[11px] text-foreground/85">controlnet_type</span>
              <Select
                value={params.controlnet_type}
                onValueChange={(v) =>
                  u("controlnet_type", v as NonNullable<PresetParams["controlnet_type"]>)
                }
              >
                <SelectTrigger className="bg-surface-2 font-mono text-[12px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(["canny", "depth", "pose", "lineart", "scribble"] as const).map((t) => (
                    <SelectItem key={t} value={t} className="font-mono text-[12px]">
                      {t}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <SliderRow
              label="controlnet_strength"
              value={params.controlnet_strength}
              onChange={(v) => u("controlnet_strength", v)}
              min={0}
              max={2}
            />
          </>
        )}
      </Group>

      {/* Mask + auto-mask */}
      {isInpaint && (
        <Group title="mask" defaultOpen={isInpaint}>
          <ToggleRow
            label="auto-mask clothing"
            desc="SegFormer-B2 detects garments — no manual mask required."
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
          {isRealvis && (
            <SliderRow
              label="feather_mask_px"
              value={params.feather_mask_px}
              onChange={(v) => u("feather_mask_px", Math.round(v))}
              min={0}
              max={64}
              step={1}
              hint="soft gradient at the mask edge"
              format={(v) => `${Math.round(v)}px`}
            />
          )}
        </Group>
      )}

      {/* RealVis LoRA stack */}
      {isRealvis && (
        <Group title="lora stack">
          <SliderRow
            label="detail"
            value={params.lora_detail_weight}
            onChange={(v) => u("lora_detail_weight", v)}
            min={0}
            max={1.5}
            hint="add-detail-xl — fabric/skin texture"
          />
          <SliderRow
            label="skin"
            value={params.lora_skin_weight}
            onChange={(v) => u("lora_skin_weight", v)}
            min={0}
            max={1.5}
            hint="realistic-skin-v5 — pores & natural tone"
          />
          <SliderRow
            label="anatomy"
            value={params.lora_anatomy_weight}
            onChange={(v) => u("lora_anatomy_weight", v)}
            min={0}
            max={1.5}
            hint="body-details-xl — proportions"
          />
          <SliderRow
            label="bust_emphasis"
            value={params.bust_emphasis}
            onChange={(v) => u("bust_emphasis", v)}
            min={-1.5}
            max={1.5}
            hint="curvy-body-xl · negative reduces, positive enhances"
            format={(v) => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
          />
        </Group>
      )}

      {/* Detailers + hires */}
      {isRealvis && (
        <Group title="detailers · hires">
          <ToggleRow
            label="person_detailer"
            desc="YOLO26n detects body, re-render at 1024px for distant subjects."
            checked={!!params.person_detailer}
            onChange={(v) => u("person_detailer", v)}
          />
          {params.person_detailer && (
            <SliderRow
              label="person_detailer_strength"
              value={params.person_detailer_strength}
              onChange={(v) => u("person_detailer_strength", v)}
              min={0}
              max={0.8}
            />
          )}
          <ToggleRow
            label="face_detailer"
            desc="Re-renders face — turn off if you want the original face preserved."
            checked={!!params.face_detailer}
            onChange={(v) => u("face_detailer", v)}
          />
          <ToggleRow
            label="hand_detailer"
            desc="Detects hands and fixes finger anatomy."
            checked={!!params.hand_detailer}
            onChange={(v) => u("hand_detailer", v)}
          />
          <ToggleRow
            label="hires_fix"
            desc="4× upscale + img2img refine — boosts final resolution."
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
          <ToggleRow
            label="preserve_face"
            desc="Composites original face/skin/background back onto the hires output."
            checked={!!params.preserve_face}
            onChange={(v) => u("preserve_face", v)}
          />
        </Group>
      )}

      {/* Two-pass + structural refiner */}
      {(preset === "inpaint" || preset === "inpaint_sdxl" || preset === "inpaint_realvis") && (
        <Group title="advanced">
          <ToggleRow
            label="two_pass (undress → redress)"
            desc="Pass 1 fills skin, pass 2 dresses. Removes original-outfit residual bias."
            checked={!!params.two_pass}
            onChange={(v) => u("two_pass", v)}
          />
          {params.two_pass && (
            <div className="space-y-1.5">
              <span className="font-mono text-[11px] text-foreground/85">skin_prompt (pass 1)</span>
              <Textarea
                value={params.skin_prompt ?? ""}
                onChange={(e) => u("skin_prompt", e.target.value)}
                rows={2}
                className="bg-surface-2 font-mono text-[12px]"
              />
            </div>
          )}
          <ToggleRow
            label="structural_refiner"
            desc="Final unsharp pass — sharpens edges, may amplify noise."
            checked={!!params.structural_refiner}
            onChange={(v) => u("structural_refiner", v)}
          />
          {params.structural_refiner && (
            <SliderRow
              label="refiner_strength"
              value={params.refiner_strength}
              onChange={(v) => u("refiner_strength", v)}
              min={0}
              max={1}
            />
          )}
        </Group>
      )}

      {preset === "inpaint_premium" && (
        <Group title="pose guard">
          <ToggleRow
            label="use_pose_guide (ControlNet)"
            desc="Locks body proportions using OpenPose, recommended for radical swaps."
            checked={!!params.use_pose_guide}
            onChange={(v) => u("use_pose_guide", v)}
          />
          {params.use_pose_guide && (
            <SliderRow
              label="pose_strength"
              value={params.pose_strength}
              onChange={(v) => u("pose_strength", v)}
              min={0}
              max={1.5}
              step={0.05}
            />
          )}
        </Group>
      )}

      {preset === "ltx_video" && (
        <Group title="video" defaultOpen>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <span className="font-mono text-[11px] text-foreground/85">num_frames</span>
              <Select
                value={String(params.num_frames)}
                onValueChange={(v) =>
                  u("num_frames", Number(v) as NonNullable<PresetParams["num_frames"]>)
                }
              >
                <SelectTrigger className="bg-surface-2 font-mono text-[12px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[25, 49, 97, 121].map((n) => (
                    <SelectItem key={n} value={String(n)} className="font-mono">
                      {n}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <NumberRow label="fps" value={params.fps} onChange={(v) => u("fps", v)} min={8} max={60} />
            <NumberRow
              label="width"
              value={params.width}
              onChange={(v) => u("width", v)}
              min={256}
              max={1216}
              step={16}
            />
            <NumberRow
              label="height"
              value={params.height}
              onChange={(v) => u("height", v)}
              min={256}
              max={704}
              step={16}
            />
          </div>
        </Group>
      )}

      {(preset === "edit" || preset === "style" || preset === "controlnet") && (
        <Group title="size">
          <div className="grid grid-cols-2 gap-3">
            <NumberRow
              label="width"
              value={params.width}
              onChange={(v) => u("width", v)}
              min={64}
              max={4096}
              step={16}
            />
            <NumberRow
              label="height"
              value={params.height}
              onChange={(v) => u("height", v)}
              min={64}
              max={4096}
              step={16}
            />
          </div>
        </Group>
      )}
    </div>
  );
}
