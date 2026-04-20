import { useEffect, useState } from "react";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { getJob, type JobState } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";

interface Props {
  jobId: string;
  onDone: (state: JobState) => void;
}

export function JobProgress({ jobId, onDone }: Props) {
  const [state, setState] = useState<JobState | null>(null);

  useEffect(() => {
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      try {
        const s = await getJob(jobId);
        if (stopped) return;
        setState(s);
        if (s.status === "success" || s.status === "failed") {
          onDone(s);
          return;
        }
      } catch {
        // keep polling; worker may still be warming up
      }
      setTimeout(tick, 1000);
    };
    tick();
    return () => {
      stopped = true;
    };
  }, [jobId, onDone]);

  if (!state) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        submitting...
      </div>
    );
  }

  const isDone = state.status === "success";
  const isFailed = state.status === "failed";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm">
        {isDone ? (
          <CheckCircle2 className="h-4 w-4 text-green-500" />
        ) : isFailed ? (
          <XCircle className="h-4 w-4 text-destructive" />
        ) : (
          <Loader2 className="h-4 w-4 animate-spin" />
        )}
        <span className="font-medium">status: {state.status}</span>
        {state.duration_ms != null && (
          <span className="text-muted-foreground">· {(state.duration_ms / 1000).toFixed(1)}s</span>
        )}
      </div>

      {Object.keys(state.stages_ms || {}).length > 0 && (
        <div className="flex flex-wrap gap-1.5 text-xs">
          {Object.entries(state.stages_ms).map(([k, ms]) => (
            <span
              key={k}
              className={cn(
                "rounded-md border px-2 py-0.5 font-mono",
                "bg-muted/40 text-muted-foreground",
              )}
            >
              {k}: {ms}ms
            </span>
          ))}
        </div>
      )}

      {isFailed && state.error && (
        <Alert variant="destructive">
          <AlertTitle>{state.error.code ?? "failed"}</AlertTitle>
          <AlertDescription>{state.error.message ?? "unknown error"}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
