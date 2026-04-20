import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { JobState } from "@/lib/api";

export function ResultViewer({ state }: { state: JobState }) {
  if (!state.output_url) return null;
  const isVideo = (state.output_kind ?? "").startsWith("video/");

  return (
    <div className="space-y-3">
      <div className="overflow-hidden rounded-lg border bg-muted/30">
        {isVideo ? (
          <video src={state.output_url} controls className="w-full" />
        ) : (
          <img src={state.output_url} alt="result" className="w-full object-contain" />
        )}
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <code className="font-mono">{state.output_url}</code>
        <Button size="sm" variant="outline" asChild>
          <a href={state.output_url} download>
            <Download className="h-3.5 w-3.5" /> download
          </a>
        </Button>
      </div>
    </div>
  );
}
