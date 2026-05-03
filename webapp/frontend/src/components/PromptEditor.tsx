import { Sparkles } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";

interface Props {
  prompt: string;
  negative: string;
  onPrompt: (s: string) => void;
  onNegative: (s: string) => void;
  presetLabel?: string;
}

export function PromptEditor({ prompt, negative, onPrompt, onNegative }: Props) {
  return (
    <div className="space-y-3">
      <Field
        label="prompt"
        icon={<Sparkles className="h-3 w-3" />}
        value={prompt}
        onChange={onPrompt}
        rows={3}
        placeholder="describe what you want — clothing, lighting, mood, fabric, photo style"
      />
      <Field
        label="negative"
        value={negative}
        onChange={onNegative}
        rows={2}
        placeholder="what to avoid (artifacts, anatomy bugs, original outfit residue)"
        muted
      />
    </div>
  );
}

function Field({
  label,
  icon,
  value,
  onChange,
  rows,
  placeholder,
  muted,
}: {
  label: string;
  icon?: React.ReactNode;
  value: string;
  onChange: (s: string) => void;
  rows: number;
  placeholder?: string;
  muted?: boolean;
}) {
  const len = value.length;
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          {icon}
          {label}
        </span>
        <span className="num text-[10px] text-muted-foreground/70">
          {len}/4000
        </span>
      </div>
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={rows}
        placeholder={placeholder}
        className={`resize-y bg-surface/60 font-mono text-[12.5px] leading-relaxed placeholder:text-muted-foreground/50 ${muted ? "text-muted-foreground" : ""}`}
      />
    </div>
  );
}
