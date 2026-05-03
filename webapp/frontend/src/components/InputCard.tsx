import { useCallback, useRef, useState } from "react";
import { CloudUpload, ImageIcon, Crop, X, Link2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { uploadFile, presetUrl } from "@/lib/api";

interface Props {
  label: string;
  hint?: string;
  uploadedName: string | null;
  bucket: [number, number] | null;
  onUploaded: (name: string | null, bucket: [number, number] | null) => void;
}

export function InputCard({ label, hint, uploadedName, bucket, onUploaded }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [origDim, setOrigDim] = useState<[number, number] | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const doUpload = useCallback(
    async (file: File) => {
      setBusy(true);
      setError(null);
      try {
        // Read original dimensions for "768×1024 → 896×1152" display
        const localUrl = URL.createObjectURL(file);
        const dims = await new Promise<[number, number]>((res, rej) => {
          const img = new Image();
          img.onload = () => res([img.naturalWidth, img.naturalHeight]);
          img.onerror = () => rej(new Error("decode failed"));
          img.src = localUrl;
        });
        setOrigDim(dims);

        const res = await uploadFile(file);
        const b = res.bucket && res.bucket.length === 2 ? (res.bucket as [number, number]) : null;
        onUploaded(res.name, b);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [onUploaded],
  );

  const clear = () => {
    onUploaded(null, null);
    setOrigDim(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void doUpload(f);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <label className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          {label}
        </label>
        {uploadedName && (
          <button
            onClick={clear}
            className="flex items-center gap-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="h-3 w-3" /> clear
          </button>
        )}
      </div>

      <div
        className={cn(
          "group relative overflow-hidden rounded-lg border border-dashed border-border bg-surface/40 transition-all",
          "hover:border-primary/40 hover:bg-surface",
          drag && "border-primary/70 bg-surface shadow-amber-soft",
          busy && "animate-pulse",
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        onClick={() => !uploadedName && inputRef.current?.click()}
      >
        {uploadedName ? (
          <div className="relative">
            <img
              src={presetUrl(uploadedName)}
              alt={label}
              className="block max-h-72 w-full object-contain bg-black"
            />
            <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-3 border-t border-border bg-background/85 px-3 py-2 text-[11px] backdrop-blur">
              <div className="flex items-center gap-2 text-muted-foreground">
                <ImageIcon className="h-3 w-3" />
                <span className="num truncate max-w-[160px]" title={uploadedName}>
                  {uploadedName}
                </span>
              </div>
              {(origDim || bucket) && (
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  {origDim && (
                    <span className="num">
                      {origDim[0]}×{origDim[1]}
                    </span>
                  )}
                  {origDim && bucket && (origDim[0] !== bucket[0] || origDim[1] !== bucket[1]) && (
                    <>
                      <Crop className="h-3 w-3 text-amber" />
                      <span className="num text-amber">
                        {bucket[0]}×{bucket[1]}
                      </span>
                    </>
                  )}
                  {!origDim && bucket && (
                    <span className="num text-amber">{bucket[0]}×{bucket[1]}</span>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
            <div className="grid h-10 w-10 place-items-center rounded-full bg-surface-2 ring-1 ring-border">
              <CloudUpload className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="font-mono text-[12px] text-foreground">
              {busy ? "uploading…" : "drop image · or click"}
            </div>
            {hint && (
              <div className="text-[11px] text-muted-foreground">{hint}</div>
            )}
            <div className="mt-1 flex items-center gap-1 text-[10px] text-muted-foreground/70">
              <Link2 className="h-2.5 w-2.5" />
              <span className="num">PNG · JPG · WEBP · 25MB</span>
            </div>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void doUpload(f);
          }}
        />
      </div>

      {error && (
        <p className="text-[11px] text-destructive font-mono">{error}</p>
      )}
    </div>
  );
}
