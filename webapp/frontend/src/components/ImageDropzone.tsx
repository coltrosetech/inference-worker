import { useCallback, useRef, useState } from "react";
import { ImageIcon, UploadCloud, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { uploadFile } from "@/lib/api";

interface Props {
  label: string;
  hint?: string;
  uploadedName: string | null;
  onUploaded: (name: string | null) => void;
  accept?: string;
}

export function ImageDropzone({ label, hint, uploadedName, onUploaded, accept = "image/*" }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const doUpload = useCallback(
    async (file: File) => {
      setBusy(true);
      setError(null);
      try {
        const res = await uploadFile(file);
        onUploaded(res.name);
        setPreview(URL.createObjectURL(file));
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [onUploaded],
  );

  const clear = () => {
    onUploaded(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        {uploadedName && (
          <Button size="sm" variant="ghost" onClick={clear} className="h-7 text-muted-foreground">
            <X className="h-3 w-3" /> clear
          </Button>
        )}
      </div>
      <div
        className={cn(
          "relative flex h-40 cursor-pointer items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/30 px-4 text-center transition-colors hover:border-primary/50 hover:bg-muted/50",
          drag && "border-primary/70 bg-muted/60",
          busy && "opacity-60",
        )}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files?.[0];
          if (f) doUpload(f);
        }}
      >
        {preview ? (
          <img src={preview} alt="preview" className="max-h-[150px] rounded-md object-contain" />
        ) : uploadedName ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ImageIcon className="h-4 w-4" />
            <span className="truncate">{uploadedName}</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-1.5 text-sm text-muted-foreground">
            <UploadCloud className="h-6 w-6" />
            <span>{busy ? "uploading..." : "drop image or click"}</span>
            {hint && <span className="text-xs">{hint}</span>}
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) doUpload(f);
          }}
        />
      </div>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
