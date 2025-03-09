import type { QueueStrategy, Tile } from "../types";
import { CLASS_COLORS, prettyLabel } from "./ProbabilityBars";

interface Props {
  tiles: Tile[];
  selectedId: number | null;
  strategy: QueueStrategy;
  loading: boolean;
  onStrategy: (s: QueueStrategy) => void;
  onSelect: (id: number) => void;
  onRefresh: () => void;
}

export function ReviewQueue({ tiles, selectedId, strategy, loading, onStrategy, onSelect, onRefresh }: Props) {
  return (
    <aside className="queue">
      <div className="queue-header">
        <h2>Review queue</h2>
        <button onClick={onRefresh} disabled={loading}>
          {loading ? "…" : "Refresh"}
        </button>
      </div>
      <label className="field">
        Order by
        <select value={strategy} onChange={(e) => onStrategy(e.target.value as QueueStrategy)}>
          <option value="uncertainty">Lowest confidence</option>
          <option value="entropy">Highest entropy</option>
          <option value="changes">Predicted changes first</option>
        </select>
      </label>
      {tiles.length === 0 && !loading && <p className="muted">Nothing left to review.</p>}
      <ul>
        {tiles.map((t) => {
          const p = t.latest_prediction;
          return (
            <li key={t.id} className={t.id === selectedId ? "active" : ""} onClick={() => onSelect(t.id)}>
              <div className="queue-key">{t.tile_key}</div>
              {p && (
                <div className="queue-meta">
                  <span className="chip" style={{ background: CLASS_COLORS[p.label] }}>
                    {prettyLabel(p.label)}
                  </span>
                  <span className="muted">{(p.confidence * 100).toFixed(0)}%</span>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
