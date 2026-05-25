import { useEffect, useMemo, useState } from "react";
import { SystemBar } from "@/components/SystemBar";
import { PresetRail, presetMeta } from "@/components/PresetRail";
import { InputCard } from "@/components/InputCard";
import { PromptEditor } from "@/components/PromptEditor";
import { DEFAULTS_BY_PRESET, PresetForm, type PresetParams } from "@/components/PresetForm";
import { JobQueue } from "@/components/JobQueue";
import { OutputViewer } from "@/components/OutputViewer";
import { CompareLightbox } from "@/components/CompareLightbox";
import { SeedControl } from "@/components/SeedControl";
import { GenerateButton } from "@/components/GenerateButton";
import { submitGenerate, type JobState, type Preset } from "@/lib/api";

export default function App() {
  const [preset, setPreset] = useState<Preset>("tryon");
  const [params, setParams] = useState<PresetParams>(DEFAULTS_BY_PRESET.tryon);

  const [inputName, setInputName] = useState<string | null>(null);
  const [inputBucket, setInputBucket] = useState<[number, number] | null>(null);
  const [maskName, setMaskName] = useState<string | null>(null);
  const [maskBucket, setMaskBucket] = useState<[number, number] | null>(null);
  const [refName, setRefName] = useState<string | null>(null);
  const [refBucket, setRefBucket] = useState<[number, number] | null>(null);

  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitErr, setSubmitErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [lightboxJob, setLightboxJob] = useState<JobState | null>(null);
  const [lightboxInput, setLightboxInput] = useState<string | null>(null);

  const meta = useMemo(() => presetMeta(preset), [preset]);

  // Try-on needs garment reference + clothing mask (auto-mask satisfies it);
  // wan_i2v only animates the input image; wan_flf2v needs an END frame.
  const presetReq =
    preset === "wan_i2v"
      ? { needsRef: false, needsMask: false }
      : preset === "wan_flf2v"
      ? { needsRef: true, needsMask: false }
      : { needsRef: true, needsMask: true };

  const changePreset = (p: Preset) => {
    setPreset(p);
    setParams(DEFAULTS_BY_PRESET[p]);
    setSubmitErr(null);
  };

  // Wan 2.2 I2V: match output aspect to the uploaded image (long side ~1024, /16).
  useEffect(() => {
    if (preset !== "wan_i2v" || !inputBucket) return;
    const [iw, ih] = inputBucket;
    const long = 1024;
    const snap = (n: number) => Math.max(256, Math.min(1280, Math.round(n / 16) * 16));
    const [w, h] = iw >= ih ? [long, (long * ih) / iw] : [(long * iw) / ih, long];
    const nw = snap(w);
    const nh = snap(h);
    if (params.width !== nw || params.height !== nh) {
      setParams((p) => ({ ...p, width: nw, height: nh }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset, inputBucket]);

  const maskRequired = presetReq.needsMask && !params.auto_mask;
  const canSubmit =
    !!inputName &&
    !!params.prompt.trim() &&
    (presetReq.needsRef ? !!refName : true) &&
    (maskRequired ? !!maskName : true) &&
    !busy;

  const submit = async () => {
    setSubmitErr(null);
    if (!inputName || !params.prompt.trim()) return;
    setBusy(true);
    try {
      const { prompt, negative_prompt, seed, ...rest } = params;
      const parameters: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(rest)) {
        if (v !== undefined && v !== null) parameters[k] = v;
      }
      if (seed !== null && seed !== undefined && Number.isFinite(seed)) {
        parameters.seed = seed;
      }
      const r = await submitGenerate({
        preset,
        prompt,
        negative_prompt: negative_prompt ?? "",
        input_image_name: inputName,
        mask_image_name: maskName ?? undefined,
        reference_image_name: refName ?? undefined,
        parameters,
        timeout_sec: preset === "wan_flf2v" ? 1200 : preset === "wan_i2v" ? 600 : 300,
      });
      if (r.status === "failed") {
        setSubmitErr("worker rejected the request");
        setBusy(false);
        return;
      }
      setActiveJobId(r.job_id);
    } catch (e) {
      setSubmitErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // Cmd/Ctrl + Enter to submit
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && canSubmit) {
        e.preventDefault();
        void submit();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  });

  const onRetry = () => {
    setParams({ ...params, seed: Math.floor(Math.random() * 9_007_199_254_740_991) });
    void submit();
  };

  const blockingHint = !canSubmit
    ? !inputName
      ? "→ upload an input image"
      : !params.prompt.trim()
      ? "→ enter a prompt"
      : presetReq.needsRef && !refName
      ? "→ upload a reference image"
      : maskRequired && !maskName
      ? "→ upload a mask (or enable auto-mask)"
      : null
    : null;

  return (
    <div className="grid h-screen min-h-screen w-full grid-rows-[auto_1fr] bg-background text-foreground">
      <SystemBar />

      <div className="grid min-h-0 grid-cols-[260px_minmax(0,1fr)_360px_360px]">
        <PresetRail selected={preset} onSelect={changePreset} />

        {/* Center column: composer */}
        <main className="reveal reveal-delay-2 flex min-h-0 flex-col overflow-hidden">
          <div className="border-b border-border px-6 py-4">
            <div className="flex items-baseline gap-2">
              <span className="font-display italic text-lg">{meta.label}</span>
              <span className="text-[11px] text-muted-foreground">— {meta.desc}</span>
            </div>
          </div>

          <div className="grid min-h-0 flex-1 grid-rows-[auto_1fr_auto]">
            {/* Inputs row — fixed height */}
            <div className="grid gap-3 border-b border-border px-6 py-5 lg:grid-cols-3">
              <InputCard
                label="input"
                uploadedName={inputName}
                bucket={inputBucket}
                onUploaded={(n, b) => {
                  setInputName(n);
                  setInputBucket(b);
                }}
              />
              {presetReq.needsMask && !params.auto_mask && (
                <InputCard
                  label="mask"
                  hint="white = garment area to replace, black = keep"
                  uploadedName={maskName}
                  bucket={maskBucket}
                  onUploaded={(n, b) => {
                    setMaskName(n);
                    setMaskBucket(b);
                  }}
                />
              )}
              {presetReq.needsRef && (
                <InputCard
                  label="reference"
                  hint="style/garment source · IP-Adapter"
                  uploadedName={refName}
                  bucket={refBucket}
                  onUploaded={(n, b) => {
                    setRefName(n);
                    setRefBucket(b);
                  }}
                />
              )}
            </div>

            {/* Composer body — scrolls */}
            <div className="grid min-h-0 grid-cols-[1.4fr_1fr] gap-4 overflow-y-auto px-6 py-5">
              <div className="space-y-4">
                <PromptEditor
                  prompt={params.prompt}
                  negative={params.negative_prompt ?? ""}
                  onPrompt={(s) => setParams({ ...params, prompt: s })}
                  onNegative={(s) => setParams({ ...params, negative_prompt: s })}
                  presetLabel={preset}
                />
              </div>
              <div className="space-y-2">
                <PresetForm preset={preset} params={params} setParams={setParams} />
              </div>
            </div>

            {/* Sticky action bar at bottom of composer */}
            <div className="border-t border-border bg-surface/30 px-6 py-3 backdrop-blur">
              <div className="flex items-center justify-between gap-3">
                <SeedControl
                  seed={params.seed ?? null}
                  onChange={(s) => setParams({ ...params, seed: s })}
                />
                <div className="flex items-center gap-3">
                  {submitErr && (
                    <span className="font-mono text-[11px] text-destructive">{submitErr}</span>
                  )}
                  <GenerateButton
                    busy={busy || (activeJobId ? true : false)}
                    disabled={!canSubmit}
                    onClick={submit}
                    hint={blockingHint}
                  />
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Right column 1: output viewer */}
        <section className="reveal reveal-delay-3 min-h-0 overflow-hidden border-l border-border bg-surface/20">
          <OutputViewer
            jobId={activeJobId}
            inputName={inputName}
            onRetry={onRetry}
            onLightbox={(j, i) => {
              setLightboxJob(j);
              setLightboxInput(i);
            }}
          />
        </section>

        {/* Right column 2: queue */}
        <section className="reveal reveal-delay-4 min-h-0 overflow-hidden border-l border-border bg-surface/40">
          <JobQueue
            selectedId={activeJobId}
            onSelect={(j) => {
              setActiveJobId(j.job_id);
              if (j.status === "success") {
                setLightboxJob(null);
              }
            }}
            onRetry={(j) => {
              setActiveJobId(j.job_id);
              setParams({
                ...params,
                seed: Math.floor(Math.random() * 9_007_199_254_740_991),
              });
              void submit();
            }}
          />
        </section>
      </div>

      <CompareLightbox
        open={!!lightboxJob}
        job={lightboxJob}
        inputName={lightboxInput}
        onClose={() => setLightboxJob(null)}
      />
    </div>
  );
}
