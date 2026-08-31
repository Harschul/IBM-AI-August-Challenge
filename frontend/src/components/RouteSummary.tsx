import type { RouteResponse } from "../types";

const VERDICT_STYLE: Record<RouteResponse["verdict"], { label: string; className: string }> = {
  DELIVERED_ON_TIME: { label: "Delivered On Time", className: "verdict--good" },
  MISSED_DEADLINE: { label: "Missed Deadline", className: "verdict--bad" },
  NO_FEASIBLE_ROUTE: { label: "No Feasible Route", className: "verdict--bad" },
};

export function RouteSummary({ route }: { route: RouteResponse | null }) {
  if (!route) {
    return <div className="route-summary route-summary--empty">Run a route to see the result.</div>;
  }

  const style = VERDICT_STYLE[route.verdict];

  return (
    <div className="route-summary">
      <span className={`verdict-badge ${style.className}`}>{style.label}</span>

      {route.feasible ? (
        <>
          <div className="route-path">
            {route.path_names.map((name, i) => (
              <span key={i} className="route-path-node">
                {name}
                {i < route.path_names.length - 1 && <span className="route-path-arrow">→</span>}
              </span>
            ))}
          </div>
          <dl className="route-stats">
            <div>
              <dt>Arrival</dt>
              <dd>{route.arrival_s?.toFixed(1)}s</dd>
            </div>
            <div>
              <dt>Hops</dt>
              <dd>{route.hops.length}</dd>
            </div>
            {route.slack_s !== null && (
              <div>
                <dt>Deadline slack</dt>
                <dd className={route.slack_s >= 0 ? "slack-positive" : "slack-negative"}>
                  {route.slack_s >= 0 ? "+" : ""}
                  {route.slack_s.toFixed(1)}s
                </dd>
              </div>
            )}
          </dl>
        </>
      ) : (
        <p className="route-summary-note">
          The bundle cannot reach any destination before its deadline. That is a valid
          answer — not an error.
        </p>
      )}
    </div>
  );
}
