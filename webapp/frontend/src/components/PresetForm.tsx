import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Preset } from "@/lib/api";

export interface PresetParams {
  prompt: string;
  negative_prompt: string;
  // shared
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
  // inpaint
  grow_mask_px?: number;
  two_pass?: boolean;
  skin_prompt?: string;
  structural_refiner?: boolean;
  refiner_strength?: number;
  auto_mask?: boolean;
  auto_mask_categories?: string[];
  // edit_premium
  guidance?: number;
  // ltx_video
  num_frames?: 25 | 49 | 97 | 121;
  fps?: number;
}

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
    steps: 30,
    cfg: 6.5,
    strength: 0.9,
    grow_mask_px: 10,
    two_pass: false,
    skin_prompt: "bare natural skin, torso, arms, body, soft even lighting, anatomy",
    structural_refiner: false,
    refiner_strength: 0.3,
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

function Num({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="flex justify-between">
        <span>{label}</span>
        <span className="text-muted-foreground font-mono text-xs">{value}</span>
      </Label>
      <Input
        type="number"
        value={value ?? ""}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function Range({
  label,
  value,
  onChange,
  step = 0.01,
  min = 0,
  max = 1,
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="flex justify-between">
        <span>{label}</span>
        <span className="text-muted-foreground font-mono text-xs">{value?.toFixed(2)}</span>
      </Label>
      <Slider
        value={[value ?? 0]}
        step={step}
        min={min}
        max={max}
        onValueChange={(v) => onChange(v[0])}
      />
    </div>
  );
}

export function PresetForm({
  preset,
  params,
  setParams,
}: {
  preset: Preset;
  params: PresetParams;
  setParams: (p: PresetParams) => void;
}) {
  const update = <K extends keyof PresetParams>(k: K, v: PresetParams[K]) => setParams({ ...params, [k]: v });

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <Label>prompt</Label>
        <Textarea
          placeholder="describe the image you want..."
          value={params.prompt}
          onChange={(e) => update("prompt", e.target.value)}
          rows={3}
        />
      </div>

      <div className="space-y-1.5">
        <Label>negative prompt (optional)</Label>
        <Textarea
          placeholder="what to avoid..."
          value={params.negative_prompt}
          onChange={(e) => update("negative_prompt", e.target.value)}
          rows={2}
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        {preset !== "ltx_video" && preset !== "edit_premium" && preset !== "inpaint_premium" && (
          <Num label="steps" value={params.steps} onChange={(v) => update("steps", v)} min={1} max={50} />
        )}
        {(preset === "edit_premium" || preset === "inpaint_premium") && (
          <Num label="steps" value={params.steps} onChange={(v) => update("steps", v)} min={4} max={50} />
        )}
        {preset === "ltx_video" && (
          <Num label="steps" value={params.steps} onChange={(v) => update("steps", v)} min={1} max={100} />
        )}

        {preset === "ltx_video" ? (
          <Num
            label="cfg"
            value={params.cfg}
            onChange={(v) => update("cfg", v)}
            step={0.1}
            min={0}
            max={20}
          />
        ) : preset === "edit_premium" || preset === "inpaint_premium" ? (
          <Num
            label="cfg"
            value={params.cfg}
            onChange={(v) => update("cfg", v)}
            step={0.1}
            min={0}
            max={10}
          />
        ) : (
          <Num
            label="cfg"
            value={params.cfg}
            onChange={(v) => update("cfg", v)}
            step={0.1}
            min={0}
            max={15}
          />
        )}

        <Num
          label="seed (blank = random)"
          value={params.seed ?? undefined}
          onChange={(v) => update("seed", Number.isFinite(v) ? v : null)}
        />
      </div>

      {(preset === "edit" || preset === "style" || preset === "controlnet" || preset === "inpaint") && (
        <Range
          label="strength (denoise)"
          value={params.strength}
          onChange={(v) => update("strength", v)}
        />
      )}

      {preset === "edit_premium" && (
        <Range
          label="guidance"
          value={params.guidance}
          onChange={(v) => update("guidance", v)}
          min={0}
          max={10}
          step={0.1}
        />
      )}

      {preset === "inpaint_premium" && (
        <Num
          label="guidance (FLUX-Fill)"
          value={params.guidance}
          onChange={(v) => update("guidance", v)}
          min={0}
          max={100}
          step={0.5}
        />
      )}

      {preset === "style" && (
        <Range
          label="style_strength (IP-Adapter)"
          value={params.style_strength}
          onChange={(v) => update("style_strength", v)}
          max={1.5}
        />
      )}

      {preset === "controlnet" && (
        <>
          <div className="space-y-1.5">
            <Label>controlnet_type</Label>
            <Select
              value={params.controlnet_type}
              onValueChange={(v) =>
                update("controlnet_type", v as NonNullable<PresetParams["controlnet_type"]>)
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(["canny", "depth", "pose", "lineart", "scribble"] as const).map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Range
            label="controlnet_strength"
            value={params.controlnet_strength}
            onChange={(v) => update("controlnet_strength", v)}
            max={2}
          />
        </>
      )}

      {(preset === "inpaint" || preset === "inpaint_sdxl" || preset === "inpaint_realvis" || preset === "inpaint_premium") && (
        <>
          <Num
            label="grow_mask_px"
            value={params.grow_mask_px}
            onChange={(v) => update("grow_mask_px", v)}
            min={0}
            max={128}
          />
          <div className="flex items-center justify-between rounded-lg border p-3">
            <div>
              <Label>auto-mask clothing</Label>
              <p className="text-xs text-muted-foreground">
                SegFormer-B2 detects garment regions — no manual mask upload needed.
              </p>
            </div>
            <Switch
              checked={!!params.auto_mask}
              onCheckedChange={(v) => update("auto_mask", v)}
            />
          </div>
          {params.auto_mask && (
            <div className="space-y-2 rounded-lg border p-3">
              <Label className="text-xs text-muted-foreground">regions to replace</Label>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                {(["upper_clothes", "pants", "skirt", "dress", "belt"] as const).map((cat) => {
                  const on = (params.auto_mask_categories ?? []).includes(cat);
                  return (
                    <div key={cat} className="flex items-center justify-between">
                      <span className="text-sm capitalize">
                        {cat.replace("_", " ")}
                      </span>
                      <Switch
                        checked={on}
                        onCheckedChange={(v) => {
                          const cur = new Set(params.auto_mask_categories ?? []);
                          if (v) cur.add(cat);
                          else cur.delete(cat);
                          update("auto_mask_categories", Array.from(cur));
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}

      {(preset === "inpaint" || preset === "inpaint_sdxl" || preset === "inpaint_realvis") && (
        <>
          <div className="flex items-center justify-between rounded-lg border p-3">
            <div>
              <Label>two-pass (undress → redress)</Label>
              <p className="text-xs text-muted-foreground">
                Fills skin first to avoid the "majority completion" bias.
              </p>
            </div>
            <Switch
              checked={params.two_pass}
              onCheckedChange={(v) => update("two_pass", v)}
            />
          </div>
          {params.two_pass && (
            <div className="space-y-1.5">
              <Label>skin_prompt (pass 1)</Label>
              <Textarea
                value={params.skin_prompt ?? ""}
                onChange={(e) => update("skin_prompt", e.target.value)}
                rows={2}
              />
            </div>
          )}
          <div className="flex items-center justify-between rounded-lg border p-3">
            <div>
              <Label>structural refiner</Label>
              <p className="text-xs text-muted-foreground">
                High-frequency unsharp at final stage.
              </p>
            </div>
            <Switch
              checked={params.structural_refiner}
              onCheckedChange={(v) => update("structural_refiner", v)}
            />
          </div>
          {params.structural_refiner && (
            <Range
              label="refiner_strength"
              value={params.refiner_strength}
              onChange={(v) => update("refiner_strength", v)}
            />
          )}
        </>
      )}

      {preset === "ltx_video" && (
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label>num_frames</Label>
            <Select
              value={String(params.num_frames)}
              onValueChange={(v) =>
                update("num_frames", Number(v) as NonNullable<PresetParams["num_frames"]>)
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[25, 49, 97, 121].map((n) => (
                  <SelectItem key={n} value={String(n)}>
                    {n}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Num
            label="fps"
            value={params.fps}
            onChange={(v) => update("fps", v)}
            min={8}
            max={60}
          />
          <Num
            label="width"
            value={params.width}
            onChange={(v) => update("width", v)}
            min={256}
            max={1216}
            step={16}
          />
          <Num
            label="height"
            value={params.height}
            onChange={(v) => update("height", v)}
            min={256}
            max={704}
            step={16}
          />
        </div>
      )}

      {(preset === "edit" || preset === "style" || preset === "controlnet") && (
        <div className="grid grid-cols-2 gap-3">
          <Num label="width" value={params.width} onChange={(v) => update("width", v)} min={64} max={4096} step={16} />
          <Num label="height" value={params.height} onChange={(v) => update("height", v)} min={64} max={4096} step={16} />
        </div>
      )}
    </div>
  );
}
