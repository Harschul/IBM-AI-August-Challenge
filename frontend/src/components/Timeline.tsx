import type { ContactIn, HopOut, RouteResponse } from "../types";

function fmtRate(bps: number): string {
  if (bps >= 1_000_000) return `${(bps / 1_000_000).toFixed(0)}Mb`;
  if (bps >= 1_000) return `${(bps / 1_000).toFixed(0)}Kb`;
  return `${bps.toFixed(0)}b`;
}

function nodeLabel(id: number, names: Record<string, string>): string {
  return names[id] ?? `N${id}`;
}

export function Timeline({
  contacts,
  route,
  tEnd,
  nodeNames,
}: {
  contacts: ContactIn[];
  route: RouteResponse | null;
  tEnd: number;
  nodeNames: Record<string, string>;
}) {
  const width = 720;
  const rowHeight = 34;
  const labelWidth = 150;
  const chartWidth = width - labelWidth;
  const t0 = 0;
  const t1 = Math.max(tEnd, 1);

  const x = (t: number) => (Math.max(t0, Math.min(t1, t)) / t1) * chartWidth;

  const usedByKey = new Map<string, HopOut>();
  if (route) {
    for (const h of route.hops) {
      usedByKey.set(`${h.source_id}-${h.destination_id}-${h.start_s}-${h.end_s}`, h);
    }
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => t0 + (t1 - t0) * f);

  return (
    <div className="timeline">
      <svg width={width} height={rowHeight * contacts.length + 30} role="img" aria-label="Contact timeline">
        {/* axis */}
        <g transform={`translate(${labelWidth}, 16)`}>
          <line x1={0} y1={0} x2={chartWidth} y2={0} stroke="var(--border)" />
          {ticks.map((t, i) => (
            <g key={i} transform={`translate(${x(t)}, 0)`}>
              <line y1={-4} y2={4} stroke="var(--text-faint)" />
              <text y={-8} textAnchor="middle" className="tl-tick">{t.toFixed(0)}s</text>
            </g>
          ))}
        </g>

        {/* rows */}
        {contacts.map((c, i) => {
          const key = `${c.source_id}-${c.destination_id}-${c.start_s}-${c.end_s}`;
          const used = usedByKey.get(key);
          const y = 30 + i * rowHeight;
          return (
            <g key={key} transform={`translate(0, ${y})`}>
              <text x={labelWidth - 10} y={rowHeight / 2 + 4} textAnchor="end" className="tl-row-label">
                {used ? "→ " : ""}
                {nodeLabel(c.source_id, nodeNames)}{"→"}{nodeLabel(c.destination_id, nodeNames)}
                <tspan className="tl-row-rate"> {fmtRate(c.data_rate_bps)}</tspan>
              </text>

              <g transform={`translate(${labelWidth}, 0)`}>
                {/* window */}
                <rect
                  x={x(c.start_s)} y={rowHeight / 2 - 6}
                  width={Math.max(1, x(c.end_s) - x(c.start_s))} height={12}
                  rx={4} className="tl-window" opacity={used ? 1 : 0.5}
                />
                {used && (
                  <>
                    {used.waited_s > 0 && (
                      <rect
                        x={x(used.available_at_s)} y={rowHeight / 2 - 3}
                        width={Math.max(0, x(used.depart_s) - x(used.available_at_s))} height={6}
                        className="tl-wait"
                      />
                    )}
                    <rect
                      x={x(used.depart_s)} y={rowHeight / 2 - 5}
                      width={Math.max(1, x(used.arrive_s) - x(used.depart_s))} height={10}
                      rx={3} className="tl-transmit"
                    />
                  </>
                )}
              </g>
            </g>
          );
        })}
      </svg>
      <div className="tl-legend">
        <span><i className="tl-swatch tl-swatch--window" /> contact window</span>
        <span><i className="tl-swatch tl-swatch--wait" /> waiting in storage</span>
        <span><i className="tl-swatch tl-swatch--transmit" /> transmitting</span>
      </div>
    </div>
  );
}
