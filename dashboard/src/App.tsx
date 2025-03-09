import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { ReviewQueue } from "./components/ReviewQueue";
import { StatsPanel } from "./components/StatsPanel";
import { TileDetail } from "./components/TileDetail";
import type { ChangeClass, QueueStrategy, Scene, Stats, Tile, TileDetail as TileDetailT } from "./types";

type View = "review" | "stats";

const loadAnalyst = () => {
  try {
    return localStorage.getItem("terrashift.analyst") ?? "";
  } catch {
    return "";
  }
};

export default function App() {
  const [view, setView] = useState<View>("review");
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [sceneId, setSceneId] = useState<number | null>(null);
  const [classes, setClasses] = useState<ChangeClass[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [strategy, setStrategy] = useState<QueueStrategy>("uncertainty");
  const [queue, setQueue] = useState<Tile[]>([]);
  const [selected, setSelected] = useState<TileDetailT | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [analyst, setAnalyst] = useState(loadAnalyst);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const guard = useCallback(async (fn: () => Promise<void>) => {
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void guard(async () => {
      const [sc, cl, md] = await Promise.all([api.scenes(), api.classes(), api.models()]);
      setScenes(sc);
      setClasses(cl);
      setModels(md.available);
    });
  }, [guard]);

  useEffect(() => {
    try {
      localStorage.setItem("terrashift.analyst", analyst);
    } catch {
      /* storage unavailable */
    }
  }, [analyst]);

  const refreshQueue = useCallback(async () => {
    setLoading(true);
    await guard(async () => setQueue(await api.queue(sceneId, strategy)));
    setLoading(false);
  }, [guard, sceneId, strategy]);

  useEffect(() => {
    void refreshQueue();
  }, [refreshQueue]);

  useEffect(() => {
    if (view === "stats") void guard(async () => setStats(await api.stats(sceneId)));
  }, [view, sceneId, guard]);

  const selectTile = (id: number) => guard(async () => setSelected(await api.tile(id)));

  const flash = (msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 2500);
  };

  const annotate = async (label: ChangeClass, notes: string, action?: "skipped") => {
    if (!selected) return;
    await guard(async () => {
      const ann = await api.annotate(selected.id, {
        analyst,
        label,
        notes: notes || undefined,
        prediction_id: selected.latest_prediction?.id,
        action,
      });
      flash(`Saved: ${ann.action} as ${label.replace(/_/g, " ")}`);
      const remaining = queue.filter((t) => t.id !== selected.id);
      setQueue(remaining);
      if (remaining.length) setSelected(await api.tile(remaining[0].id));
      else setSelected(null);
    });
  };

  const repredict = async (model: string) => {
    if (!selected) return;
    await guard(async () => {
      await api.predict(selected.id, model);
      setSelected(await api.tile(selected.id));
      flash("Prediction updated");
    });
  };

  const predictScene = async () => {
    if (sceneId === null) return;
    await guard(async () => {
      const r = await api.predictScene(sceneId);
      flash(`Ran ${r.model} on ${r.predicted} tiles`);
      await refreshQueue();
    });
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1>TerraShift</h1>
        <nav>
          <button className={view === "review" ? "active" : ""} onClick={() => setView("review")}>
            Review
          </button>
          <button className={view === "stats" ? "active" : ""} onClick={() => setView("stats")}>
            Stats
          </button>
        </nav>
        <div className="topbar-right">
          <select value={sceneId ?? ""} onChange={(e) => setSceneId(e.target.value ? Number(e.target.value) : null)}>
            <option value="">All scenes</option>
            {scenes.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.tile_count})
              </option>
            ))}
          </select>
          <button onClick={() => void predictScene()} disabled={sceneId === null}>
            Predict scene
          </button>
          <a href={api.exportUrl()} target="_blank" rel="noreferrer">
            Export labels
          </a>
        </div>
      </header>

      {error && (
        <div className="error" onClick={() => setError(null)}>
          {error}
        </div>
      )}
      {toast && <div className="toast">{toast}</div>}

      {view === "review" ? (
        <main className="review-layout">
          <ReviewQueue
            tiles={queue}
            selectedId={selected?.id ?? null}
            strategy={strategy}
            loading={loading}
            onStrategy={setStrategy}
            onSelect={(id) => void selectTile(id)}
            onRefresh={() => void refreshQueue()}
          />
          {selected ? (
            <TileDetail
              tile={selected}
              classes={classes}
              models={models}
              analyst={analyst}
              onAnalyst={setAnalyst}
              onAnnotate={annotate}
              onRepredict={repredict}
            />
          ) : (
            <section className="detail empty">
              <p className="muted">Select a tile from the queue to review it.</p>
            </section>
          )}
        </main>
      ) : (
        <main>
          <StatsPanel stats={stats} classes={classes} />
        </main>
      )}
    </div>
  );
}
