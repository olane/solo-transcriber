import { useEffect, useRef } from "react";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";

type Props = {
  xml: string;
  onError: (message: string) => void;
};

export function ScoreView({ xml, onError }: Props) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const target = container.current;
    if (!target || !xml) return;
    let cancelled = false;
    target.innerHTML = "";

    const osmd = new OpenSheetMusicDisplay(target, {
      autoResize: true,
      drawingParameters: "compacttight",
    });

    osmd
      .load(xml)
      .then(() => {
        if (!cancelled) osmd.render();
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          onError(error instanceof Error ? error.message : String(error));
        }
      });

    return () => {
      cancelled = true;
      osmd.clear();
    };
  }, [xml, onError]);

  return <div className="score" ref={container} />;
}
