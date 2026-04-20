import { useMemo, useState } from "react";
import { Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { ImageDropzone } from "@/components/ImageDropzone";
import { DEFAULTS_BY_PRESET, PresetForm, type PresetParams } from "@/components/PresetForm";
import { JobProgress } from "@/components/JobProgress";
import { ResultViewer } from "@/components/ResultViewer";
import { submitGenerate, type JobState, type Preset } from "@/lib/api";

const PRESETS: { value: Preset; label: string; desc: string; needsRef?: boolean; needsMask?: boolean }[] = [
  { value: "edit", label: "edit", desc: "SDXL Lightning img2img + IP-Adapter (preservation)" },
  { value: "style", label: "style", desc: "IP-Adapter style transfer", needsRef: true },
  { value: "controlnet", label: "controlnet", desc: "Canny / depth / pose / lineart / scribble" },
  { value: "inpaint", label: "inpaint", desc: "Mask-guided regeneration (two-pass opt)", needsMask: true },
  { value: "edit_premium", label: "edit_premium", desc: "FLUX.1-Kontext prompt-driven edit" },
  { value: "ltx_video", label: "ltx_video", desc: "LTX-Video img2video (mp4)" },
];

export default function App() {
  const [preset, setPreset] = useState<Preset>("edit");
  const [params, setParams] = useState<PresetParams>(DEFAULTS_BY_PRESET.edit);
  const [inputName, setInputName] = useState<string | null>(null);
  const [maskName, setMaskName] = useState<string | null>(null);
  const [refName, setRefName] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitErr, setSubmitErr] = useState<string | null>(null);
  const [done, setDone] = useState<JobState | null>(null);

  const presetMeta = useMemo(() => PRESETS.find((p) => p.value === preset)!, [preset]);

  const changePreset = (p: Preset) => {
    setPreset(p);
    setParams(DEFAULTS_BY_PRESET[p]);
    setDone(null);
  };

  const maskRequired = presetMeta.needsMask && !params.auto_mask;
  const canSubmit =
    !!inputName &&
    !!params.prompt.trim() &&
    (presetMeta.needsRef ? !!refName : true) &&
    (maskRequired ? !!maskName : true) &&
    !jobId;

  const submit = async () => {
    setSubmitErr(null);
    setDone(null);
    if (!inputName) return;
    try {
      const {
        prompt,
        negative_prompt,
        seed,
        ...rest
      } = params;
      const parameters: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(rest)) {
        if (v !== undefined && v !== null && v !== "") parameters[k] = v;
      }
      if (seed !== null && seed !== undefined && Number.isFinite(seed)) parameters.seed = seed;

      const r = await submitGenerate({
        preset,
        prompt,
        negative_prompt: negative_prompt ?? "",
        input_image_name: inputName,
        mask_image_name: maskName ?? undefined,
        reference_image_name: refName ?? undefined,
        parameters,
        timeout_sec: preset === "ltx_video" ? 600 : 300,
      });
      if (r.status === "failed") {
        setSubmitErr("worker rejected the request");
        return;
      }
      setJobId(r.job_id);
    } catch (e) {
      setSubmitErr((e as Error).message);
    }
  };

  const reset = () => {
    setJobId(null);
    setDone(null);
    setSubmitErr(null);
  };

  return (
    <div className="min-h-screen">
      <header className="border-b bg-card/30">
        <div className="container flex items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <div>
              <h1 className="text-lg font-semibold leading-tight">inference worker · playground</h1>
              <p className="text-xs text-muted-foreground">
                coltrosetech/inference-worker · 6 presets · HMAC-signed callbacks
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="container grid gap-6 py-8 lg:grid-cols-[1fr_1.3fr]">
        {/* Left column: inputs + form */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>preset</CardTitle>
              <CardDescription>{presetMeta.desc}</CardDescription>
            </CardHeader>
            <CardContent>
              <Select value={preset} onValueChange={(v) => changePreset(v as Preset)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PRESETS.map((p) => (
                    <SelectItem key={p.value} value={p.value}>
                      {p.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>inputs</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <ImageDropzone
                label="input image"
                uploadedName={inputName}
                onUploaded={setInputName}
              />
              {maskRequired && (
                <ImageDropzone
                  label="mask image"
                  hint="white = inpaint, black = keep"
                  uploadedName={maskName}
                  onUploaded={setMaskName}
                />
              )}
              {presetMeta.needsRef && (
                <ImageDropzone
                  label="reference image"
                  hint="style source (IP-Adapter)"
                  uploadedName={refName}
                  onUploaded={setRefName}
                />
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <PresetForm preset={preset} params={params} setParams={setParams} />
            </CardContent>
          </Card>
        </div>

        {/* Right column: actions + progress + result */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>run</CardTitle>
              <CardDescription>
                worker ingests via HTTP callback; HMAC-verified receipt.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Button onClick={submit} disabled={!canSubmit} className="w-full">
                  {jobId ? "running..." : "generate"}
                </Button>
                {(jobId || done) && (
                  <Button variant="outline" onClick={reset}>
                    reset
                  </Button>
                )}
              </div>
              {submitErr && <p className="text-xs text-destructive">{submitErr}</p>}
              {!canSubmit && !jobId && (
                <p className="text-xs text-muted-foreground">
                  {!inputName
                    ? "upload an input image to enable"
                    : !params.prompt.trim()
                    ? "enter a prompt"
                    : presetMeta.needsRef && !refName
                    ? "upload a reference image"
                    : maskRequired && !maskName
                    ? "upload a mask image (or toggle auto-mask)"
                    : ""}
                </p>
              )}

              {jobId && (
                <>
                  <Separator />
                  <JobProgress
                    jobId={jobId}
                    onDone={(s) => {
                      setDone(s);
                    }}
                  />
                </>
              )}
            </CardContent>
          </Card>

          {done && done.status === "success" && (
            <Card>
              <CardHeader>
                <CardTitle>result</CardTitle>
                <CardDescription>
                  {done.preset} · {done.duration_ms != null ? `${(done.duration_ms / 1000).toFixed(1)}s` : ""}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResultViewer state={done} />
              </CardContent>
            </Card>
          )}
        </div>
      </main>

      <footer className="border-t py-4">
        <div className="container text-xs text-muted-foreground">
          vast.ai · ComfyUI · FLUX.1-Kontext / LTX-Video / SDXL Lightning
        </div>
      </footer>
    </div>
  );
}
