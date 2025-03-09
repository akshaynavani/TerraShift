import { useEffect, useState } from "react";
import type { ChangeClass, Prediction } from "../types";
import { prettyLabel } from "./ProbabilityBars";

interface Props {
  classes: ChangeClass[];
  prediction: Prediction | null;
  analyst: string;
  onAnalyst: (name: string) => void;
  onSubmit: (label: ChangeClass, notes: string, action?: "skipped") => Promise<void>;
}

export function CorrectionForm({ classes, prediction, analyst, onAnalyst, onSubmit }: Props) {
  const [label, setLabel] = useState<ChangeClass>(prediction?.label ?? "no_change");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setLabel(prediction?.label ?? "no_change");
    setNotes("");
  }, [prediction?.id]);

  const submit = async (action?: "skipped") => {
    setBusy(true);
    try {
      await onSubmit(label, notes, action);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT") return;
      const n = Number(e.key);
      if (Number.isInteger(n) && n >= 1 && n <= classes.length) {
        setLabel(classes[n - 1]);
      } else if (e.key === "Enter" && analyst && !busy) {
        e.preventDefault();
        void submit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const isCorrection = prediction !== null && label !== prediction.label;

  return (
    <form
      className="correction"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <h3>Analyst label</h3>
      <label className="field">
        Analyst
        <input value={analyst} onChange={(e) => onAnalyst(e.target.value)} placeholder="your name" required />
      </label>
      <div className="label-grid">
        {classes.map((c) => (
          <label key={c} className={`label-option ${label === c ? "selected" : ""}`}>
            <input type="radio" name="label" value={c} checked={label === c} onChange={() => setLabel(c)} />
            {prettyLabel(c)}
          </label>
        ))}
      </div>
      <label className="field">
        Notes
        <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} />
      </label>
      <div className="actions">
        <button type="submit" disabled={busy || !analyst} className={isCorrection ? "warn" : "primary"}>
          {isCorrection ? "Submit correction" : "Confirm prediction"}
        </button>
        <button type="button" disabled={busy || !analyst} onClick={() => void submit("skipped")}>
          Skip
        </button>
      </div>
      <p className="muted small">Shortcuts: 1–{classes.length} pick class, Enter submits.</p>
    </form>
  );
}
