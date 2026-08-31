import type { BundleIn } from "../types";

export function BundleForm({
  bundle,
  destinations,
  onBundleChange,
  onDestinationsChange,
}: {
  bundle: BundleIn;
  destinations: number[];
  onBundleChange: (b: BundleIn) => void;
  onDestinationsChange: (d: number[]) => void;
}) {
  return (
    <div className="bundle-form">
      <div className="field-row">
        <label className="field-label" htmlFor="bundle_id">Bundle ID</label>
        <input
          id="bundle_id" className="field-input" type="text"
          value={bundle.bundle_id}
          onChange={(e) => onBundleChange({ ...bundle, bundle_id: e.target.value })}
        />
      </div>
      <div className="field-grid">
        <div className="field-row">
          <label className="field-label" htmlFor="source_id">Source node ID</label>
          <input
            id="source_id" className="field-input" type="number"
            value={bundle.source_id}
            onChange={(e) => onBundleChange({ ...bundle, source_id: Number(e.target.value) })}
          />
        </div>
        <div className="field-row">
          <label className="field-label" htmlFor="size_bytes">Size (bytes)</label>
          <input
            id="size_bytes" className="field-input" type="number"
            value={bundle.size_bytes}
            onChange={(e) => onBundleChange({ ...bundle, size_bytes: Number(e.target.value) })}
          />
        </div>
        <div className="field-row">
          <label className="field-label" htmlFor="science_priority">Priority (0–1)</label>
          <input
            id="science_priority" className="field-input" type="number" min={0} max={1} step={0.01}
            value={bundle.science_priority}
            onChange={(e) => onBundleChange({ ...bundle, science_priority: Number(e.target.value) })}
          />
        </div>
        <div className="field-row">
          <label className="field-label" htmlFor="deadline_s">Deadline (s)</label>
          <input
            id="deadline_s" className="field-input" type="number"
            value={bundle.deadline_s ?? ""}
            placeholder="none"
            onChange={(e) =>
              onBundleChange({ ...bundle, deadline_s: e.target.value === "" ? null : Number(e.target.value) })
            }
          />
        </div>
        <div className="field-row">
          <label className="field-label" htmlFor="data_type">Data type</label>
          <select
            id="data_type" className="field-input"
            value={bundle.data_type}
            onChange={(e) => onBundleChange({ ...bundle, data_type: e.target.value })}
          >
            {["TRANSIENT", "STAR_FIELD", "CALIBRATION", "HOUSEKEEPING"].map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        <div className="field-row">
          <label className="field-label" htmlFor="destinations">Destination node IDs</label>
          <input
            id="destinations" className="field-input" type="text"
            value={destinations.join(", ")}
            onChange={(e) =>
              onDestinationsChange(
                e.target.value.split(",").map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n)),
              )
            }
          />
        </div>
      </div>
    </div>
  );
}
