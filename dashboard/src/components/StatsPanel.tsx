import type { ChangeClass, Stats } from "../types";
import { prettyLabel } from "./ProbabilityBars";

export function StatsPanel({ stats, classes }: { stats: Stats | null; classes: ChangeClass[] }) {
  if (!stats) return <p className="muted">Loading stats…</p>;
  const pct = (n: number, d: number) => (d ? `${((n / d) * 100).toFixed(1)}%` : "–");
  const maxCell = Math.max(1, ...Object.values(stats.confusion).flatMap((r) => Object.values(r)));

  return (
    <section className="stats">
      <div className="kpis">
        <Kpi label="Tiles" value={stats.tiles} />
        <Kpi label="Predicted" value={stats.predicted} sub={pct(stats.predicted, stats.tiles)} />
        <Kpi label="Reviewed" value={stats.reviewed} sub={pct(stats.reviewed, stats.predicted)} />
        <Kpi label="Corrected" value={stats.corrected} sub={pct(stats.corrected, stats.reviewed)} />
        <Kpi
          label="Model–analyst agreement"
          value={stats.agreement_rate === null ? "–" : `${(stats.agreement_rate * 100).toFixed(1)}%`}
        />
      </div>

      <div className="card">
        <h3>Analyst label (rows) vs model prediction (columns)</h3>
        <table className="confusion">
          <thead>
            <tr>
              <th />
              {classes.map((c) => (
                <th key={c}>{prettyLabel(c)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {classes.map((row) => (
              <tr key={row}>
                <th>{prettyLabel(row)}</th>
                {classes.map((col) => {
                  const v = stats.confusion[row]?.[col] ?? 0;
                  const alpha = v / maxCell;
                  const bg = row === col ? `rgba(22,163,74,${alpha})` : `rgba(220,38,38,${alpha})`;
                  return (
                    <td key={col} style={{ background: bg }}>
                      {v || ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Label distribution</h3>
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Predicted</th>
              <th>Analyst-reviewed</th>
            </tr>
          </thead>
          <tbody>
            {classes.map((c) => (
              <tr key={c}>
                <td>{prettyLabel(c)}</td>
                <td>{stats.predicted_label_counts[c] ?? 0}</td>
                <td>{stats.reviewed_label_counts[c] ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Kpi({ label, value, sub }: { label: string; value: number | string; sub?: string }) {
  return (
    <div className="kpi">
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">
        {label}
        {sub && <span className="muted"> · {sub}</span>}
      </div>
    </div>
  );
}
