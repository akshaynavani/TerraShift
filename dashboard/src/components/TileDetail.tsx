import { useState } from "react";
import { api } from "../api";
import type { ChangeClass, TileDetail as TileDetailT } from "../types";
import { CorrectionForm } from "./CorrectionForm";
import { CLASS_COLORS, ProbabilityBars, prettyLabel } from "./ProbabilityBars";

interface Props {
  tile: TileDetailT;
  classes: ChangeClass[];
  models: string[];
  analyst: string;
  onAnalyst: (a: string) => void;
  onAnnotate: (label: ChangeClass, notes: string, action?: "skipped") => Promise<void>;
  onRepredict: (model: string) => Promise<void>;
}

const OVERLAYS = ["ndvi_diff", "ndwi_diff", "ndbi_diff"];

export function TileDetail({ tile, classes, models, analyst, onAnalyst, onAnnotate, onRepredict }: Props) {
  const [overlay, setOverlay] = useState(OVERLAYS[0]);
  const [model, setModel] = useState("auto");
  const pred = tile.latest_prediction;

  return (
    <section className="detail">
      <header className="detail-header">
        <div>
          <h2>{tile.tile_key}</h2>
          <p className="muted">
            Tile #{tile.id}
            {tile.lat !== null && tile.lon !== null && ` · ${tile.lat.toFixed(3)}, ${tile.lon.toFixed(3)}`}
            {tile.ground_truth && ` · reference label: ${prettyLabel(tile.ground_truth)}`}
          </p>
        </div>
        <div className="repredict">
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="auto">auto</option>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <button onClick={() => void onRepredict(model)}>Re-run model</button>
        </div>
      </header>

      <div className="imagery">
        <figure>
          <img src={api.previewUrl(tile.id, "before")} alt="before" />
          <figcaption>Before (t0)</figcaption>
        </figure>
        <figure>
          <img src={api.previewUrl(tile.id, "after")} alt="after" />
          <figcaption>After (t1)</figcaption>
        </figure>
        <figure>
          <img src={api.previewUrl(tile.id, overlay)} alt={overlay} />
          <figcaption>
            <select value={overlay} onChange={(e) => setOverlay(e.target.value)}>
              {OVERLAYS.map((o) => (
                <option key={o} value={o}>
                  {o.replace("_diff", "").toUpperCase()} change
                </option>
              ))}
            </select>
          </figcaption>
        </figure>
      </div>

      <div className="detail-grid">
        <div className="card">
          <h3>Model prediction</h3>
          {pred ? (
            <>
              <p>
                <span className="chip" style={{ background: CLASS_COLORS[pred.label] }}>
                  {prettyLabel(pred.label)}
                </span>{" "}
                <span className="muted">
                  {pred.model_name} · confidence {(pred.confidence * 100).toFixed(1)}% · entropy {pred.entropy.toFixed(2)}
                </span>
              </p>
              <ProbabilityBars probabilities={pred.probabilities} />
            </>
          ) : (
            <p className="muted">No prediction yet.</p>
          )}
        </div>
        <div className="card">
          <CorrectionForm classes={classes} prediction={pred} analyst={analyst} onAnalyst={onAnalyst} onSubmit={onAnnotate} />
        </div>
      </div>

      {tile.annotations.length > 0 && (
        <div className="card">
          <h3>Annotation history</h3>
          <table>
            <thead>
              <tr>
                <th>When</th>
                <th>Analyst</th>
                <th>Action</th>
                <th>Label</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {tile.annotations.map((a) => (
                <tr key={a.id}>
                  <td>{new Date(a.created_at).toLocaleString()}</td>
                  <td>{a.analyst}</td>
                  <td>{a.action}</td>
                  <td>{prettyLabel(a.label)}</td>
                  <td>{a.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
