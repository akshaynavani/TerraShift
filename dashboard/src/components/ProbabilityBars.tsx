import type { ChangeClass } from "../types";

export const CLASS_COLORS: Record<ChangeClass, string> = {
  no_change: "#94a3b8",
  vegetation_loss: "#b45309",
  urban_expansion: "#7c3aed",
  water_change: "#0284c7",
  agricultural_change: "#65a30d",
};

export const prettyLabel = (c: string) => c.replace(/_/g, " ");

export function ProbabilityBars({ probabilities }: { probabilities: Record<ChangeClass, number> }) {
  const entries = (Object.entries(probabilities) as [ChangeClass, number][]).sort((a, b) => b[1] - a[1]);
  return (
    <div className="prob-bars">
      {entries.map(([cls, p]) => (
        <div key={cls} className="prob-row">
          <span className="prob-label">{prettyLabel(cls)}</span>
          <div className="prob-track">
            <div className="prob-fill" style={{ width: `${(p * 100).toFixed(1)}%`, background: CLASS_COLORS[cls] }} />
          </div>
          <span className="prob-value">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
