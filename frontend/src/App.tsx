import { useEffect, useState } from "react";
import type { BundleIn, ContactIn, RouteResponse, ScenarioMap } from "./types";
import { computeRoute, fetchScenarios } from "./api";
import { BundleForm } from "./components/BundleForm";
import { ContactTable } from "./components/ContactTable";
import { Timeline } from "./components/Timeline";
import { RouteSummary } from "./components/RouteSummary";

const FALLBACK_NODE_NAMES: Record<string, string> = {
  0: "SCI",
  1: "LEO1", 2: "LEO2", 3: "LEO3", 4: "LEO4",
  5: "LEO5", 6: "LEO6", 7: "LEO7", 8: "LEO8",
  9: "GEO1", 10: "GEO2",
  11: "GNDA", 12: "GNDB", 13: "GNDC",
};

const DEFAULT_BUNDLE: BundleIn = {
  bundle_id: "OBS-004812",
  source_id: 0,
  size_bytes: 90_000_000,
  science_priority: 0.96,
  deadline_s: 600,
  data_type: "TRANSIENT",
};

const DEFAULT_CONTACTS: ContactIn[] = [
  { source_id: 0, destination_id: 9, start_s: 0, end_s: 900, data_rate_bps: 2_000_000, propagation_delay_s: 0.12 },
  { source_id: 9, destination_id: 11, start_s: 0, end_s: 900, data_rate_bps: 2_000_000, propagation_delay_s: 0.12 },
  { source_id: 0, destination_id: 3, start_s: 0, end_s: 180, data_rate_bps: 10_000_000, propagation_delay_s: 0.004 },
  { source_id: 3, destination_id: 5, start_s: 120, end_s: 400, data_rate_bps: 10_000_000, propagation_delay_s: 0.006 },
  { source_id: 5, destination_id: 12, start_s: 300, end_s: 600, data_rate_bps: 10_000_000, propagation_delay_s: 0.003 },
];

export default function App() {
  const [scenarios, setScenarios] = useState<ScenarioMap | null>(null);
  const [scenarioKey, setScenarioKey] = useState<string>("hybrid-leo-mesh");
  const [contacts, setContacts] = useState<ContactIn[]>(DEFAULT_CONTACTS);
  const [bundle, setBundle] = useState<BundleIn>(DEFAULT_BUNDLE);
  const [destinations, setDestinations] = useState<number[]>([11, 12, 13]);
  const [tEnd, setTEnd] = useState(900);
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchScenarios()
      .then((s) => {
        setScenarios(s);
        applyScenario(s, scenarioKey);
      })
      .catch(() => setError("Could not reach the router API at http://localhost:8000. Is it running?"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function applyScenario(map: ScenarioMap, key: string) {
    const s = map[key];
    if (!s) return;
    setContacts(s.contacts);
    setBundle(s.bundle);
    setTEnd(s.t_end);
  }

  function handleScenarioSelect(key: string) {
    setScenarioKey(key);
    if (scenarios) applyScenario(scenarios, key);
    setRoute(null);
  }

  async function runRoute() {
    setLoading(true);
    setError(null);
    try {
      const res = await computeRoute({ contacts, bundle, destinations });
      setRoute(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Route computation failed");
    } finally {
      setLoading(false);
    }
  }

  const nodeNames = route?.node_names ?? FALLBACK_NODE_NAMES;

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">
          <span className="app-title-eyebrow">Multi-Orbit Scientific Data Relay Network</span>
          <h1>Temporal Router</h1>
        </div>
        <div className="scenario-picker">
          {scenarios &&
            Object.entries(scenarios).map(([key, s]) => (
              <button
                key={key}
                className={`scenario-button ${key === scenarioKey ? "scenario-button--active" : ""}`}
                onClick={() => handleScenarioSelect(key)}
                title={s.why}
              >
                {s.title.split("—")[0].trim()}
              </button>
            ))}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}
      {scenarios?.[scenarioKey] && (
        <p className="scenario-why">{scenarios[scenarioKey].why}</p>
      )}

      <main className="app-grid">
        <section className="panel panel--config">
          <h2 className="panel-title">Bundle</h2>
          <BundleForm
            bundle={bundle}
            destinations={destinations}
            onBundleChange={setBundle}
            onDestinationsChange={setDestinations}
          />

          <h2 className="panel-title panel-title--spaced">Contact Plan</h2>
          <ContactTable contacts={contacts} onChange={setContacts} />

          <div className="field-row field-row--inline">
            <label className="field-label" htmlFor="t_end">Timeline horizon (s)</label>
            <input
              id="t_end" className="field-input field-input--narrow" type="number"
              value={tEnd} onChange={(e) => setTEnd(Number(e.target.value))}
            />
          </div>

          <button className="predict-button" onClick={runRoute} disabled={loading}>
            {loading ? "Routing…" : "Compute Route"}
          </button>
        </section>

        <section className="panel panel--result">
          <h2 className="panel-title">Route Result</h2>
          <RouteSummary route={route} />

          <h2 className="panel-title panel-title--spaced">Timeline</h2>
          <Timeline contacts={contacts} route={route} tEnd={tEnd} nodeNames={nodeNames} />
        </section>
      </main>
    </div>
  );
}
