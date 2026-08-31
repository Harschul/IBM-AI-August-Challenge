"""Smooth browser-side frontend for the locked final replay.

Streamlit supplies a fully computed ReplayData object.  The browser renders and
animates it with requestAnimationFrame, so playback does not require continuous
Streamlit reruns.  All routing/science semantics remain in the existing Python
runner; this module is presentation only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.frontend.layout import topology_positions
from src.frontend.replay import ReplayData, node_label
from src.integration.config import GEO_IDS, GROUND_IDS, LEO_IDS, SCIENCE_IDS, link_profile_name, node_role


ROLE_RADIUS = {
    "SCIENCE": 1.18,
    "LEO": 1.34,
    "GEO": 1.82,
    "GROUND": 0.94,
}


def _call_number(obj: object, method: str, default: float | int | None = None):
    fn = getattr(obj, method, None)
    if not callable(fn):
        return default
    try:
        value = fn()
    except Exception:
        return default
    if isinstance(value, (int, float)):
        return value
    return default


def _node_metadata(replay: ReplayData) -> list[dict[str, Any]]:
    config = replay.config
    satellites = {node_id: replay.satellites[node_id] for node_id in range(len(replay.satellites))}
    ground_lookup = {station.node_id: station for station in config.ground_stations}
    rows: list[dict[str, Any]] = []
    for node_id in range(14):
        role = node_role(node_id)
        row: dict[str, Any] = {
            "id": node_id,
            "label": node_label(node_id, config),
            "role": role,
        }
        if node_id in satellites:
            sat = satellites[node_id]
            radius_km = float(_call_number(sat, "radius", config.earth_radius_km) or config.earth_radius_km)
            omega = float(_call_number(sat, "angular_velocity", 0.0) or 0.0)
            row.update({
                "radius_km": radius_km,
                "altitude_km": radius_km - config.earth_radius_km,
                "inclination_deg": math.degrees(float(_call_number(sat, "inclination", 0.0) or 0.0)),
                "raan_deg": math.degrees(float(_call_number(sat, "raan", 0.0) or 0.0)),
                "initial_phase_deg": math.degrees(float(_call_number(sat, "initial_phase", 0.0) or 0.0)),
                "angular_velocity_rad_s": omega,
                "orbital_period_s": (2.0 * math.pi / abs(omega)) if abs(omega) > 1e-12 else None,
                "connection_range_km": _call_number(sat, "connection_range", None),
                "storage_capacity": _call_number(sat, "storage_capacity", None),
                "transmit_limit": _call_number(sat, "transmission_limit", None),
                "link_bandwidth": _call_number(sat, "link_bandwidth", None),
            })
        else:
            station = ground_lookup[node_id]
            row.update({
                "lat_deg": station.lat_deg,
                "lon_deg": station.lon_deg,
                "min_elevation_deg": station.min_elevation_deg,
                "weather_risk": station.weather_risk,
            })
        rows.append(row)
    return rows


def _positions(replay: ReplayData) -> list[list[list[float]]]:
    rows: list[list[list[float]]] = []
    for time_s in replay.times:
        frame: list[list[float]] = []
        for node_id in range(14):
            xyz = np.asarray(replay.node_position_3d(node_id, time_s), dtype=float)
            frame.append([round(float(xyz[0]), 4), round(float(xyz[1]), 4), round(float(xyz[2]), 4)])
        rows.append(frame)
    return rows


def _contacts(replay: ReplayData) -> list[dict[str, Any]]:
    rows = []
    for contact in replay.plan.contacts:
        rows.append({
            "source": int(contact.source_id),
            "destination": int(contact.destination_id),
            "start": float(contact.start_s),
            "end": float(contact.end_s),
            "rate_bps": float(contact.data_rate_bps),
            "range_km": float(contact.range_km),
            "reliability": float(contact.reliability),
            "weather_risk": float(contact.weather_risk),
            "energy_cost": float(contact.energy_cost),
            "link_type": str(contact.link_type),
            "profile": link_profile_name(contact.source_id, contact.destination_id) or str(contact.link_type),
        })
    return rows


def _bundle_payload(replay: ReplayData) -> list[dict[str, Any]]:
    ordered = sorted(replay.bundle_runs, key=lambda bundle: (bundle.created_s, bundle.bundle_id))
    rows = []
    for index, bundle in enumerate(ordered, start=1):
        attempts = []
        for attempt in bundle.attempts:
            attempts.append({
                "attempt": int(attempt.attempt_index),
                "holder": int(attempt.holder_id),
                "destination": int(attempt.destination_id),
                "requested_algorithm": str(attempt.requested_algorithm),
                "actual_algorithm": str(attempt.actual_algorithm),
                "fallback_used": bool(attempt.fallback_used),
                "fallback_reason": attempt.fallback_reason,
                "contact_start": float(attempt.contact_start_s),
                "contact_end": float(attempt.contact_end_s),
                "failure_probability": float(attempt.failure_probability),
                "success_draw": float(attempt.success_draw),
                "success": bool(attempt.success),
                "transfer_progress": float(attempt.transfer_progress),
                "capacity_bytes_consumed": int(attempt.capacity_bytes_consumed),
                "depart": float(attempt.depart_s),
                "event_time": float(attempt.event_time_s),
                "arrival": None if attempt.arrival_s is None else float(attempt.arrival_s),
            })
        rows.append({
            "index": index,
            "id": bundle.bundle_id,
            "source": int(bundle.source_id),
            "created": float(bundle.created_s),
            "size_bytes": int(bundle.size_bytes),
            "priority": float(bundle.science_priority),
            "deadline": None if bundle.deadline_s is None else float(bundle.deadline_s),
            "data_type": str(bundle.data_type),
            "path": [int(node_id) for node_id in bundle.path],
            "delivered": bool(bundle.delivered),
            "on_time": bool(bundle.on_time),
            "arrival": None if bundle.arrival_s is None else float(bundle.arrival_s),
            "reason": str(bundle.reason),
            "fallbacks": int(bundle.fallbacks),
            "transfer_failures": int(bundle.transfer_failures),
            "wasted_capacity_bytes": int(bundle.wasted_capacity_bytes),
            "attempts": attempts,
        })
    return rows


def _benchmark_summary(replay: ReplayData) -> dict[str, Any] | None:
    path = replay.spec.benchmark.output_dir / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _model_metadata(replay: ReplayData) -> dict[str, Any] | None:
    path = replay.spec.model_metadata
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_client_payload(replay: ReplayData, *, scenario_label: str, scenario_description: str = "") -> dict[str, Any]:
    topo = topology_positions()
    return {
        "experiment": {
            "name": replay.spec.name,
            "reported": bool(replay.reported_experiment),
            "mode": replay.mode,
            "scenario_label": scenario_label,
            "scenario_description": scenario_description,
            "config_sha256": replay.spec.scenario_config_sha256,
            "model_path": replay.model_path,
            "model_loaded": bool(replay.model_loaded),
            "traffic_seed": int(replay.traffic_seed),
            "stochastic_seed": int(replay.stochastic_seed),
            "horizon_s": float(replay.horizon_s),
            "sample_step_s": float(replay.config.sample_step_s),
            "before_policy": replay.before_policy,
            "after_policy": replay.after_policy,
            "switch_time_s": replay.switch_time_s,
            "playback_rate": 12.0,
        },
        "earth": {
            "radius_km": float(replay.config.earth_radius_km),
            "rotation_rad_s": float(replay.config.earth_rotation_rad_s),
        },
        "nodes": _node_metadata(replay),
        "frames": {
            "times": [float(value) for value in replay.times],
            "positions": _positions(replay),
        },
        "topology": {str(node_id): [float(x), float(y)] for node_id, (x, y) in topo.items()},
        "contacts": _contacts(replay),
        "bundles": _bundle_payload(replay),
        "config": replay.config.raw,
        "model_metadata": _model_metadata(replay),
        "benchmark_summary": _benchmark_summary(replay),
    }


def render_client_html(replay: ReplayData, *, scenario_label: str, scenario_description: str = "") -> str:
    data = json.dumps(build_client_payload(replay, scenario_label=scenario_label, scenario_description=scenario_description), separators=(",", ":"))
    # Avoid terminating the script tag if a future text field contains HTML-like text.
    data = data.replace("</", "<\\/")
    template = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
:root{--ink:#172033;--muted:#718198;--line:#dce4ed;--soft:#f7f9fc;--blue:#155eef;--green:#087f5b;--red:#c1123f;--orange:#c25b00;--amber:#a45d00;--panel:#fff}
*{box-sizing:border-box}
html,body{margin:0;background:#fff;color:var(--ink);font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace;font-size:12px}
button,input,select{font:inherit}
.shell{padding:0 2px 24px}
.toolbar{display:flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:6px;padding:7px 8px;background:#fff;position:sticky;top:0;z-index:10}
.small{font-size:10px;color:var(--muted)}
.btn{border:1px solid #cfd8e3;background:#fff;color:#314158;border-radius:4px;padding:6px 10px;cursor:pointer;font-weight:750}
.btn.primary{background:#fff0d5;border-color:#e3ad47;color:#844500}
.btn:hover{background:#f7f9fc}
.time{font-weight:900;min-width:68px;text-align:right;font-variant-numeric:tabular-nums;margin-left:auto}
.metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border:1px solid var(--line);border-top:0;border-radius:0 0 6px 6px;margin-bottom:8px}
.metric{padding:8px 10px;border-right:1px solid #edf1f5}
.metric:last-child{border-right:0}
.ml{font-size:9px;color:#6d7e96;text-transform:uppercase}
.mv{font-weight:950;font-size:14px;margin-top:2px;font-variant-numeric:tabular-nums}
.benchmark{border:1px solid var(--line);border-radius:6px;margin:8px 0;background:#fff;overflow:hidden}
.benchmark-head{display:flex;justify-content:space-between;gap:10px;padding:7px 10px;background:#f7f9fc;border-bottom:1px solid var(--line)}
.benchmark-title{font-weight:950;font-size:10px;text-transform:uppercase}
.benchmark-note{font-size:9px;color:var(--muted)}
.benchmark table{font-size:10px}.benchmark th{position:static}.benchmark td,.benchmark th{padding:6px 8px}
.alg-temporal{color:#0b6b4f;font-weight:950}.alg-ppo{color:#155eef;font-weight:950}
.tabs{display:flex;gap:4px;margin:10px 0 8px}
.tab{border:1px solid var(--line);background:#f7f9fc;color:#42536b;border-radius:4px;padding:7px 12px;cursor:pointer;font-weight:900}
.tab.active{background:#172033;color:#fff;border-color:#172033}
.tabpanel{display:none} .tabpanel.active{display:block}
.grid2{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(360px,.8fr);gap:10px}
.panel{border:1px solid var(--line);border-radius:6px;background:#fff;padding:9px}
.ph{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-bottom:6px}
.pt{font-weight:950;font-size:11px;text-transform:uppercase;color:#26364d}
.note{font-size:9px;color:#8291a6}
canvas{width:100%;height:auto;display:block;background:#fbfcfe;border:1px solid #edf1f5;border-radius:4px}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:9px;color:#617188;margin-top:5px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px}
.tablewrap{max-height:410px;overflow:auto;border:1px solid var(--line);border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:10px}
th{position:sticky;top:0;background:#f4f7fa;color:#506078;text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);z-index:1}
td{padding:7px 8px;border-bottom:1px solid #edf1f5;vertical-align:top}
tr.clickable{cursor:pointer} tr.clickable:hover{background:#f5f8ff} tr.selected{background:#eef4ff}
.route{color:#155eef;font-weight:700;white-space:nowrap}
.status{font-weight:850}
.bad{color:var(--red)} .good{color:var(--green)} .warn{color:var(--orange)}
details{border:1px solid var(--line);border-radius:5px;background:#fff;margin-top:8px}
summary{cursor:pointer;padding:8px 10px;font-weight:900;color:#33445b}
.details-body{padding:0 10px 10px}
.inspector{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.65fr);gap:10px;margin-top:10px}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}
.card{background:#f8fafc;border:1px solid #e5ebf2;border-radius:4px;padding:7px;min-height:52px}
.cl{font-size:8px;color:#7c8da3;text-transform:uppercase} .cv{font-size:11px;font-weight:900;margin-top:4px;overflow-wrap:anywhere}
.attempts{margin-top:8px;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.attempt{display:grid;grid-template-columns:58px 1.2fr 1fr 1fr 90px;gap:6px;padding:7px 8px;border-bottom:1px solid #edf1f5;align-items:center}
.attempt:last-child{border-bottom:0} .attempt.fail{background:#fff4f6} .attempt.ok{background:#f7fcfa}
.node-layout{display:grid;grid-template-columns:280px minmax(0,1fr);gap:10px}
.node-list{border:1px solid var(--line);border-radius:5px;max-height:720px;overflow:auto;padding:6px}
.group{font-size:9px;color:#7b8ba0;font-weight:900;text-transform:uppercase;padding:7px 5px 3px}
.nodebtn{display:flex;width:100%;justify-content:space-between;align-items:center;border:1px solid transparent;background:#fff;padding:7px;border-radius:4px;cursor:pointer;text-align:left}
.nodebtn:hover{background:#f5f8ff} .nodebtn.active{background:#eef4ff;border-color:#bdd2ff}
.role{font-size:9px;color:#8291a6}
.queue{max-height:240px;overflow:auto}
.drawer{position:fixed;right:-520px;top:0;width:500px;max-width:92vw;height:100vh;background:white;border-left:1px solid var(--line);box-shadow:-10px 0 30px rgba(15,23,42,.12);z-index:50;transition:right .18s ease;padding:14px;overflow:auto}
.drawer.open{right:0} .drawer-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.code{white-space:pre-wrap;word-break:break-word;background:#f7f9fc;border:1px solid #e3eaf2;border-radius:4px;padding:8px;font-size:9px;max-height:300px;overflow:auto}
.empty{padding:18px;color:#8696ac;text-align:center;font-style:italic}
@media(max-width:1050px){.grid2,.inspector,.node-layout{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,1fr)}.cards{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="shell">
  <div class="toolbar">
    <button class="btn primary" id="playBtn">Play</button>
    <button class="btn" id="stepBtn">Step</button>
    <button class="btn" id="resetBtn">Reset</button>
    <div class="time" id="timeText">0.0s</div>
    <button class="btn" id="metaBtn">Run metadata</button>
  </div>
  <div class="metrics" id="metrics"></div>
  <div class="benchmark" id="benchmarkPanel"></div>

  <div class="tabs">
    <button class="tab active" data-tab="network">Live network</button>
    <button class="tab" data-tab="packets">Packets</button>
    <button class="tab" data-tab="satellites">Satellites</button>
  </div>

  <section class="tabpanel active" id="tab-network">
    <div class="grid2">
      <div class="panel">
        <div class="ph"><div class="pt">Orbital view</div><div class="note">2D orthographic render · display radii compressed · no interaction required</div></div>
        <canvas id="orbit" width="1040" height="590"></canvas>
        <div class="legend"><span><i class="dot" style="background:#111827"></i>science</span><span><i class="dot" style="background:#155eef"></i>LEO</span><span><i class="dot" style="background:#c25b00"></i>GEO</span><span><i class="dot" style="background:#087f5b"></i>ground</span></div>
      </div>
      <div class="panel">
        <div class="ph"><div class="pt">Network topology</div><div class="note">only links available at the current time</div></div>
        <canvas id="topology" width="650" height="590"></canvas>
        <div class="legend"><span>gray = current link</span><span style="color:#155eef">blue = selected route</span><span style="color:#c1123f">red = failed attempt</span></div>
      </div>
    </div>
  </section>

  <section class="tabpanel" id="tab-packets">
    <div class="panel">
      <div class="ph"><div class="pt">Generated packets</div><div class="note">stable packet index · click a row to inspect its route and failures</div></div>
      <div class="tablewrap"><table><thead><tr><th>#</th><th>Packet</th><th>Priority</th><th>Current node</th><th>Route so far</th><th>Status</th><th>Failures</th></tr></thead><tbody id="liveRows"></tbody></table></div>
      <details id="deliveredDetails"><summary id="deliveredSummary">Delivered packets (0)</summary><div class="details-body"><div class="tablewrap" style="max-height:280px"><table><thead><tr><th>#</th><th>Packet</th><th>Destination</th><th>Route</th><th>Arrival</th><th>On time</th><th>Failures</th></tr></thead><tbody id="deliveredRows"></tbody></table></div></div></details>
    </div>
    <div class="inspector">
      <div class="panel" id="packetInspector"><div class="empty">Select a packet to inspect all metadata and recorded transfer attempts.</div></div>
      <div class="panel">
        <div class="ph"><div class="pt">Selected packet topology</div><div class="note">recorded route + failures</div></div>
        <canvas id="packetTopology" width="650" height="520"></canvas>
      </div>
    </div>
  </section>

  <section class="tabpanel" id="tab-satellites">
    <div class="node-layout">
      <div class="node-list" id="nodeList"></div>
      <div class="panel" id="nodeInspector"></div>
    </div>
  </section>
</div>

<div class="drawer" id="drawer">
  <div class="drawer-head"><div class="pt">Run metadata</div><button class="btn" id="closeDrawer">Close</button></div>
  <div id="runMetadata"></div>
</div>

<script>
const DATA=__DATA__;
const NODE=Object.fromEntries(DATA.nodes.map(n=>[n.id,n]));
const labels=id=>NODE[id]?.label ?? String(id);
const roleRadius={SCIENCE:1.18,LEO:1.34,GEO:1.82,GROUND:.94};
const roleColor={SCIENCE:'#111827',LEO:'#155eef',GEO:'#c25b00',GROUND:'#087f5b'};
const horizon=DATA.experiment.horizon_s;
const step=DATA.experiment.sample_step_s;
const playbackRate=DATA.experiment.playback_rate;
let simTime=0, playing=false, lastWall=performance.now(), selectedPacket=null, selectedNode=0, activeTab='network';
let lastPanelUpdate=-1;

const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>`${(100*x).toFixed(1)}%`;
const mb=x=>`${(x/1e6).toFixed(0)} MB`;
const fmt=x=>x==null?'—':Number(x).toFixed(1);
const nodeLabel=id=>esc(labels(id));

function frameAt(t){
  const times=DATA.frames.times; if(t<=times[0]) return [0,0]; if(t>=times[times.length-1]) return [times.length-1,0];
  const raw=t/step; const i=Math.max(0,Math.min(times.length-2,Math.floor(raw))); return [i,Math.max(0,Math.min(1,(t-times[i])/(times[i+1]-times[i])))];
}
function pos(id,t){
  const [i,a]=frameAt(t), p0=DATA.frames.positions[i][id], p1=DATA.frames.positions[Math.min(i+1,DATA.frames.positions.length-1)][id];
  return [p0[0]+a*(p1[0]-p0[0]),p0[1]+a*(p1[1]-p0[1]),p0[2]+a*(p1[2]-p0[2])];
}
function displayPos(id,t){
  const p=pos(id,t), r=Math.hypot(...p)||1, role=NODE[id].role, rr=roleRadius[role];
  return [rr*p[0]/r, rr*p[1]/r, p[2]/r];
}
function activeContacts(t){return DATA.contacts.filter(c=>c.start<=t && t<=c.end)}
function attemptEnd(a){return a.success && a.arrival!=null ? a.arrival : a.event_time}
function stateFor(b,t){
  if(t<b.created) return {status:'NOT GENERATED',key:'future',holder:b.source,next:null,failures:0,hops:0,current:null};
  let holder=b.source, failures=0,hops=0,current=null,last=b.created;
  for(const a of b.attempts){
    const end=attemptEnd(a); if(t<a.depart) break;
    if(a.depart<=t && t<=end){current=a;holder=a.holder;return {status:'TRANSMITTING',key:'transmitting',holder,next:a.destination,failures,hops,current}}
    last=Math.max(last,end); if(a.success){holder=a.destination;hops++} else {holder=a.holder;failures++}
  }
  if(b.delivered && b.arrival!=null && t>=b.arrival) return {status:'DELIVERED',key:'delivered',holder:b.path.at(-1)??holder,next:null,failures,hops,current:null};
  if(t>=horizon && !b.delivered) return {status:'DROPPED',key:'dropped',holder,next:null,failures,hops,current:null};
  const recent=b.attempts.filter(a=>!a.success && attemptEnd(a)<=t).at(-1);
  if(recent && t-attemptEnd(recent)<=12) return {status:'TX FAILED / RETRY',key:'failed',holder,next:null,failures,hops,current:null};
  const future=b.attempts.find(a=>a.depart>t && a.holder===holder);
  return {status:'WAITING',key:'waiting',holder,next:future?.destination??null,failures,hops,current:null};
}
function generated(t){return DATA.bundles.filter(b=>b.created<=t)}
function metricsAt(t){
  const rows=generated(t).map(b=>[b,stateFor(b,t)]), delivered=rows.filter(x=>x[1].key==='delivered'), active=rows.filter(x=>!['delivered','dropped'].includes(x[1].key));
  const fails=DATA.bundles.reduce((n,b)=>n+b.attempts.filter(a=>!a.success && a.event_time<=t).length,0);
  const ontime=delivered.filter(x=>x[0].on_time).length;
  return {generated:rows.length,active:active.length,delivered:delivered.length,fails,rate:rows.length?delivered.length/rows.length:0,deadline:rows.length?ontime/rows.length:0};
}
function routeText(b){return b.path.length?b.path.map(labels).join(' → '):labels(b.source)}
function routeSoFar(b,t){const out=[b.source];for(const a of b.attempts){const end=attemptEnd(a);if(a.success&&end<=t&&out.at(-1)!==a.destination)out.push(a.destination);if(a.depart<=t&&t<end&&out.at(-1)!==a.destination)out.push(a.destination)}return out.map(labels).join(' → ')}
function destination(b){return b.path.length && NODE[b.path.at(-1)]?.role==='GROUND'?labels(b.path.at(-1)):'ANY GROUND'}
function priorityName(p){return p>=.75?'CRITICAL':p>=.45?'IMPORTANT':'ROUTINE'}
function statusClass(k){return k==='delivered'?'good':k==='failed'||k==='dropped'?'bad':k==='transmitting'?'warn':''}

function drawEarth(ctx,cx,cy,R,t){
  ctx.save();ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);ctx.clip();
  const g=ctx.createRadialGradient(cx-R*.28,cy-R*.32,R*.08,cx,cy,R);g.addColorStop(0,'#f8fbff');g.addColorStop(1,'#dfeaf6');ctx.fillStyle=g;ctx.fillRect(cx-R,cy-R,R*2,R*2);
  ctx.strokeStyle='rgba(86,121,157,.20)';ctx.lineWidth=1;
  for(const lat of [-60,-30,0,30,60]){const q=Math.sin(lat*Math.PI/180), ry=Math.cos(lat*Math.PI/180);ctx.beginPath();ctx.ellipse(cx,cy+q*R,R*ry,R*.11*ry,0,0,Math.PI*2);ctx.stroke()}
  const theta=t*DATA.earth.rotation_rad_s;
  for(let k=0;k<8;k++){const lon=k*Math.PI/4+theta;ctx.beginPath();let started=false;for(let j=-60;j<=60;j+=3){const lat=j*Math.PI/180, depth=Math.cos(lat)*Math.cos(lon);if(depth<0){started=false;continue}const x=cx+R*Math.cos(lat)*Math.sin(lon),y=cy-R*Math.sin(lat);if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)}ctx.stroke()}
  ctx.restore();ctx.strokeStyle='#b9c9d9';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);ctx.stroke();
}
function drawOrbit(){
  const c=document.getElementById('orbit'),ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);const cx=W*.50,cy=H*.50,scale=Math.min(W,H)*.245;
  drawEarth(ctx,cx,cy,scale,simTime);
  // guide rings: visual only, intentionally compressed
  ctx.setLineDash([4,5]);ctx.strokeStyle='#d7e1eb';ctx.lineWidth=1;for(const r of [1.18,1.34,1.82]){ctx.beginPath();ctx.ellipse(cx,cy,scale*r,scale*r*.78,0,0,Math.PI*2);ctx.stroke()}ctx.setLineDash([]);
  const ordered=[...DATA.nodes].sort((a,b)=>displayPos(a.id,simTime)[2]-displayPos(b.id,simTime)[2]);
  for(const n of ordered){const p=displayPos(n.id,simTime),x=cx+p[0]*scale,y=cy-p[1]*scale,front=.35+.65*((p[2]+1)/2);ctx.globalAlpha=front;ctx.fillStyle=roleColor[n.role];ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();if(n.role==='GROUND'){ctx.rect(x-5,y-5,10,10)}else{ctx.arc(x,y,n.role==='SCIENCE'?6:n.role==='GEO'?6:4.5,0,Math.PI*2)}ctx.fill();ctx.stroke();ctx.globalAlpha=1;if(n.role==='SCIENCE'||n.role==='GEO'){ctx.fillStyle='#314158';ctx.font='10px ui-monospace';ctx.fillText(n.label,x+8,y-7)}}
  if(selectedPacket){const b=DATA.bundles.find(x=>x.id===selectedPacket),s=b?stateFor(b,simTime):null;if(s?.current){const a=s.current,end=attemptEnd(a),alpha=Math.max(0,Math.min(1,(simTime-a.depart)/(end-a.depart||1)))*(a.success?1:a.transfer_progress),p0=displayPos(a.holder,simTime),p1=displayPos(a.destination,simTime),x=cx+(p0[0]+alpha*(p1[0]-p0[0]))*scale,y=cy-(p0[1]+alpha*(p1[1]-p0[1]))*scale;ctx.fillStyle='#ffb000';ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill()}}
}
function topoXY(id,c){const p=DATA.topology[String(id)],x=(p[0]+2.8)/5.6*c.width,y=(2.8-p[1])/5.6*c.height;return [x,y]}
function line(ctx,a,b,color,width=1,dash=[]){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.setLineDash([])}
function drawTopology(canvasId,packetOnly=false){
  const c=document.getElementById(canvasId),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,c.width,c.height);
  if(!packetOnly) for(const contact of activeContacts(simTime)) line(ctx,topoXY(contact.source,c),topoXY(contact.destination,c),'rgba(100,116,139,.25)',1);
  const b=selectedPacket?DATA.bundles.find(x=>x.id===selectedPacket):null;
  if(b){
    const showFullHistory=packetOnly && b.delivered && b.arrival!=null && simTime>=b.arrival;
    for(const a of b.attempts){const end=attemptEnd(a);if(!showFullHistory && a.depart>simTime) continue;if(!showFullHistory && a.success && end>simTime) continue;if(a.success) line(ctx,topoXY(a.holder,c),topoXY(a.destination,c),'#155eef',3);else if(showFullHistory||a.depart<=simTime){line(ctx,topoXY(a.holder,c),topoXY(a.destination,c),'#c1123f',3,[5,3]);const p0=topoXY(a.holder,c),p1=topoXY(a.destination,c),x=(p0[0]+p1[0])/2,y=(p0[1]+p1[1])/2;ctx.fillStyle='#c1123f';ctx.font='bold 14px ui-monospace';ctx.fillText('×',x-5,y+5)}}
  }
  for(const n of DATA.nodes){const [x,y]=topoXY(n.id,c);ctx.fillStyle='#fff';ctx.strokeStyle=roleColor[n.role];ctx.lineWidth=(b&&stateFor(b,simTime).holder===n.id)?4:2;ctx.beginPath();if(n.role==='GROUND')ctx.rect(x-7,y-7,14,14);else ctx.arc(x,y,n.role==='SCIENCE'?8:6,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#34445a';ctx.font='9px ui-monospace';ctx.textAlign='center';ctx.fillText(n.label,x,y-11);ctx.textAlign='left'}
}
function updateMetrics(){const m=metricsAt(simTime),vals=[['Generated',`${m.generated} / ${DATA.bundles.length}`,''],['Active',m.active,''],['Delivered',m.delivered,'good'],['Failed TX',m.fails,'warn'],['Delivery rate',pct(m.rate),''],['Deadline success',pct(m.deadline),'good']];document.getElementById('metrics').innerHTML=vals.map(v=>`<div class="metric"><div class="ml">${v[0]}</div><div class="mv ${v[2]}">${v[1]}</div></div>`).join('')}
function renderBenchmark(){
  const b=DATA.benchmark_summary,el=document.getElementById('benchmarkPanel');if(!b?.algorithms){el.style.display='none';return}
  const t=b.algorithms.temporal,r=b.algorithms.rl_pure;const val=(x,k)=>x?.[k]?.mean;
  const row=(name,cls,x)=>`<tr><td class="${cls}">${name}</td><td>${pct(val(x,'delivery_ratio'))}</td><td>${pct(val(x,'deadline_success'))}</td><td>${pct(val(x,'priority_weighted_timely'))}</td><td>${fmt(val(x,'mean_latency_s'))}s</td><td>${pct(val(x,'transfer_failure_rate'))}</td></tr>`;
  el.innerHTML=`<div class="benchmark-head"><div class="benchmark-title">Committed final benchmark</div><div class="benchmark-note">${b.num_seeds} held-out paired runs × ${b.bundles_per_seed} packets per algorithm</div></div><table><thead><tr><th>Algorithm</th><th>Delivery</th><th>On time</th><th>Priority-weighted timely</th><th>Successful-delivery latency</th><th>Transfer failure rate</th></tr></thead><tbody>${row('Temporal','alg-temporal',t)}${row('PPO','alg-ppo',r)}</tbody></table>`;
}
function renderRows(){
  const live=[],delivered=[];for(const b of DATA.bundles){if(b.created>simTime) continue;const s=stateFor(b,simTime);const row={b,s};(s.key==='delivered'?delivered:live).push(row)}
  live.sort((a,b)=>a.b.index-b.b.index);delivered.sort((a,b)=>b.b.index-a.b.index);
  document.getElementById('liveRows').innerHTML=live.map(({b,s})=>`<tr class="clickable ${selectedPacket===b.id?'selected':''}" data-packet="${b.id}"><td>${b.index}</td><td><b>${b.id}</b></td><td>${priorityName(b.priority)} · ${b.priority.toFixed(2)}</td><td>${nodeLabel(s.holder)}</td><td class="route">${esc(routeSoFar(b,simTime))}</td><td class="status ${statusClass(s.key)}">${s.status}</td><td class="${s.failures?'bad':''}">${s.failures}</td></tr>`).join('')||`<tr><td colspan="7" class="empty">No packets have been generated yet. Press Play.</td></tr>`;
  document.getElementById('deliveredSummary').textContent=`Delivered packets (${delivered.length})`;
  document.getElementById('deliveredRows').innerHTML=delivered.map(({b,s})=>`<tr class="clickable ${selectedPacket===b.id?'selected':''}" data-packet="${b.id}"><td>${b.index}</td><td><b>${b.id}</b></td><td>${esc(destination(b))}</td><td class="route">${esc(routeText(b))}</td><td>${fmt(b.arrival)}s</td><td class="${b.on_time?'good':'bad'}">${b.on_time?'Yes':'No'}</td><td class="${b.transfer_failures?'bad':''}">${b.transfer_failures}</td></tr>`).join('');
  document.querySelectorAll('tr[data-packet]').forEach(tr=>tr.onclick=()=>{selectedPacket=tr.dataset.packet;updatePanels(true);drawTopology('topology');drawTopology('packetTopology',true)})
}
function packetInspector(){
  const el=document.getElementById('packetInspector');if(!selectedPacket){el.innerHTML='<div class="empty">Select a packet to inspect all metadata and recorded transfer attempts.</div>';return}
  const b=DATA.bundles.find(x=>x.id===selectedPacket),s=stateFor(b,simTime),deadlineRemaining=b.deadline==null?null:b.deadline-simTime;
  const cards=[['Packet',`#${b.index} · ${b.id}`],['Status',s.status],['Current node',labels(s.holder)],['Next node',s.next==null?'—':labels(s.next)],['Data type',b.data_type],['Payload',mb(b.size_bytes)],['Priority',`${priorityName(b.priority)} · ${b.priority.toFixed(3)}`],['Created',`${fmt(b.created)}s`],['Deadline',b.deadline==null?'—':`${fmt(b.deadline)}s`],['Deadline remaining',deadlineRemaining==null?'—':`${fmt(deadlineRemaining)}s`],['Failures',b.transfer_failures],['Wasted capacity',mb(b.wasted_capacity_bytes)]];
  const attempts=b.attempts.map(a=>`<div class="attempt ${a.success?'ok':'fail'}"><b>#${a.attempt}</b><span>${nodeLabel(a.holder)} → ${nodeLabel(a.destination)}</span><span>${a.actual_algorithm.toUpperCase()}</span><span>risk ${pct(a.failure_probability)}</span><span class="${a.success?'good':'bad'}">${a.success?'SUCCESS':'FAILED'}</span></div>`).join('')||'<div class="empty">No transmission attempts recorded.</div>';
  const fullAttempts=b.attempts.map(a=>`<tr><td>#${a.attempt}</td><td>${nodeLabel(a.holder)} → ${nodeLabel(a.destination)}</td><td>${fmt(a.depart)}s</td><td>${fmt(a.contact_start)}–${fmt(a.contact_end)}s</td><td>${pct(a.failure_probability)}</td><td>${a.success_draw.toFixed(3)}</td><td>${pct(a.transfer_progress)}</td><td>${mb(a.capacity_bytes_consumed)}</td><td>${a.fallback_used?'Yes':'No'}</td></tr>`).join('')||'<tr><td colspan="9" class="empty">No attempts.</td></tr>';
  el.innerHTML=`<div class="ph"><div class="pt">Packet #${b.index} · ${b.id}</div><div class="note">all values come from the locked replay</div></div><div class="cards">${cards.map(c=>`<div class="card"><div class="cl">${c[0]}</div><div class="cv">${esc(c[1])}</div></div>`).join('')}</div><div style="margin-top:8px"><div class="cl">Realized route in this replay</div><div class="cv route">${esc(routeText(b))}</div><div class="note" style="margin-top:4px">The backend selects next hops during execution; PPO does not pre-program a full route.</div></div><div class="attempts">${attempts}</div><details><summary>Full transfer-attempt metadata</summary><div class="details-body"><div class="tablewrap" style="max-height:260px"><table><thead><tr><th>Attempt</th><th>Link</th><th>Depart</th><th>Contact window</th><th>Failure risk</th><th>Draw</th><th>Progress</th><th>Capacity</th><th>Fallback</th></tr></thead><tbody>${fullAttempts}</tbody></table></div></div></details><details><summary>Raw packet metadata</summary><div class="details-body code">${esc(JSON.stringify(b,null,2))}</div></details>`;
}
function nodeList(){
  const groups=[['SCIENCE',DATA.nodes.filter(n=>n.role==='SCIENCE')],['LEO RELAYS',DATA.nodes.filter(n=>n.role==='LEO')],['GEO RELAYS',DATA.nodes.filter(n=>n.role==='GEO')],['GROUND RECEIVERS',DATA.nodes.filter(n=>n.role==='GROUND')]];
  document.getElementById('nodeList').innerHTML=groups.map(g=>`<div class="group">${g[0]}</div>${g[1].map(n=>`<button class="nodebtn ${selectedNode===n.id?'active':''}" data-node="${n.id}"><b>${n.label}</b><span class="role">${n.role}</span></button>`).join('')}`).join('');
  document.querySelectorAll('[data-node]').forEach(btn=>btn.onclick=()=>{selectedNode=Number(btn.dataset.node);nodeList();nodeInspector()})
}
function nodeInspector(){
  const n=NODE[selectedNode],p=pos(n.id,simTime),contacts=activeContacts(simTime).filter(c=>c.source===n.id||c.destination===n.id),queue=DATA.bundles.map(b=>[b,stateFor(b,simTime)]).filter(x=>x[0].created<=simTime&&x[1].holder===n.id&&!['delivered','dropped'].includes(x[1].key));
  const base=[['Node',`${n.label} · ID ${n.id}`],['Role',n.role],['X / Y / Z',`${p.map(v=>v.toFixed(0)).join(' / ')} km`]];
  if(n.role==='GROUND') base.push(['Latitude',`${n.lat_deg.toFixed(4)}°`],['Longitude',`${n.lon_deg.toFixed(4)}°`],['Min elevation',`${n.min_elevation_deg.toFixed(1)}°`],['Weather risk',pct(n.weather_risk)]);else base.push(['Altitude',`${n.altitude_km.toFixed(0)} km`],['Inclination',`${n.inclination_deg.toFixed(2)}°`],['RAAN',`${n.raan_deg.toFixed(2)}°`],['Orbital period',n.orbital_period_s?`${(n.orbital_period_s/60).toFixed(1)} min`:'—']);
  const links=contacts.map(c=>`<tr><td>${nodeLabel(c.source)} → ${nodeLabel(c.destination)}</td><td>${esc(c.profile)}</td><td>${(c.rate_bps/1e6).toFixed(0)} Mbps</td><td>${c.range_km.toFixed(0)} km</td></tr>`).join('')||'<tr><td colspan="4" class="empty">No current links.</td></tr>';
  const qrows=queue.map((x)=>`<tr class="clickable" data-qpacket="${x[0].id}"><td>#${x[0].index}</td><td>${x[0].id}</td><td>${x[1].status}</td><td>${priorityName(x[0].priority)}</td></tr>`).join('')||'<tr><td colspan="4" class="empty">No packets currently held here.</td></tr>';
  document.getElementById('nodeInspector').innerHTML=`<div class="ph"><div class="pt">${n.label} metadata</div><div class="note">current time ${simTime.toFixed(1)}s</div></div><div class="cards">${base.map(c=>`<div class="card"><div class="cl">${c[0]}</div><div class="cv">${esc(c[1])}</div></div>`).join('')}</div><div class="pt" style="margin-top:12px">Current links</div><div class="tablewrap" style="max-height:220px"><table><thead><tr><th>Link</th><th>Profile</th><th>Rate</th><th>Range</th></tr></thead><tbody>${links}</tbody></table></div><div class="pt" style="margin-top:12px">Packets currently held here</div><div class="note">Reconstructed from the recorded replay holder state; this is not a separate queue simulator.</div><div class="tablewrap queue"><table><thead><tr><th>#</th><th>Packet</th><th>Status</th><th>Priority</th></tr></thead><tbody>${qrows}</tbody></table></div>${n.role==='GROUND'?'':`<details><summary>Satellite object fields not enforced as final routing constraints</summary><div class="details-body cards"><div class="card"><div class="cl">Storage field</div><div class="cv">${esc(n.storage_capacity??'—')}</div></div><div class="card"><div class="cl">TX limit field</div><div class="cv">${esc(n.transmit_limit??'—')}</div></div><div class="card"><div class="cl">Link bandwidth field</div><div class="cv">${esc(n.link_bandwidth??'—')}</div></div></div></details>`}<details><summary>Raw node metadata</summary><div class="details-body code">${esc(JSON.stringify(n,null,2))}</div></details>`;
  document.querySelectorAll('[data-qpacket]').forEach(tr=>tr.onclick=()=>{selectedPacket=tr.dataset.qpacket;setTab('packets');updatePanels(true)})
}
function metadata(){
  const exp=DATA.experiment, model=DATA.model_metadata||{},bench=DATA.benchmark_summary||{};
  const head=[['Traffic profile',exp.scenario_label],['Routing mode',exp.mode],['Reported benchmark mode',exp.reported?'Yes':'No'],['Traffic seed',exp.traffic_seed],['Stochastic seed',exp.stochastic_seed],['Config SHA-256',exp.config_sha256],['PPO model',exp.model_path],['Model loaded',exp.model_loaded?'Yes':'No']];
  document.getElementById('runMetadata').innerHTML=`<div class="cards" style="grid-template-columns:1fr 1fr">${head.map(c=>`<div class="card"><div class="cl">${c[0]}</div><div class="cv">${esc(c[1])}</div></div>`).join('')}</div><details open><summary>Backend features surfaced by this frontend</summary><div class="details-body">Temporal earliest-arrival routing · pure MaskablePPO routing · scheduled policy switching for demo mode · physical contact windows · shared contact capacity · seeded stochastic transfer failures/retries · three research satellites · six LEO relays · two GEO relays · three operational ground receivers · packet priorities/deadlines · committed paired benchmark evidence.</div></details><details open><summary>Frontend-only presentation / derived views</summary><div class="details-body">The 2D orbit projection, compressed display radii, browser playback rate, human-readable traffic-run summaries, route highlighting, and reconstructed “packets currently held here” views are presentation logic. They do not add new routing or simulation behavior.</div></details><details><summary>Locked scenario configuration</summary><div class="details-body code">${esc(JSON.stringify(DATA.config,null,2))}</div></details><details><summary>PPO model metadata</summary><div class="details-body code">${esc(JSON.stringify(model,null,2))}</div></details><details><summary>Committed benchmark summary</summary><div class="details-body code">${esc(JSON.stringify(bench,null,2))}</div></details>`;
}
function setTab(name){activeTab=name;document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));document.querySelectorAll('.tabpanel').forEach(p=>p.classList.toggle('active',p.id===`tab-${name}`));if(name==='packets'){packetInspector();drawTopology('packetTopology',true)}if(name==='satellites'){nodeList();nodeInspector()}}
function updatePanels(force=false){
  const second=Math.floor(simTime);if(!force&&second===lastPanelUpdate)return;lastPanelUpdate=second;updateMetrics();document.getElementById('timeText').textContent=`${simTime.toFixed(1)}s`;renderRows();packetInspector();if(activeTab==='satellites')nodeInspector();
}
function animate(now){
  const dt=(now-lastWall)/1000;lastWall=now;if(playing){simTime=Math.min(horizon,simTime+dt*playbackRate);if(simTime>=horizon){playing=false;document.getElementById('playBtn').textContent='Play'}}
  drawOrbit();drawTopology('topology');if(activeTab==='packets')drawTopology('packetTopology',true);updatePanels(false);requestAnimationFrame(animate);
}

document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
document.getElementById('playBtn').onclick=()=>{playing=!playing;lastWall=performance.now();document.getElementById('playBtn').textContent=playing?'Pause':'Play'};
document.getElementById('stepBtn').onclick=()=>{playing=false;document.getElementById('playBtn').textContent='Play';simTime=Math.min(horizon,simTime+step);updatePanels(true)};
document.getElementById('resetBtn').onclick=()=>{playing=false;document.getElementById('playBtn').textContent='Play';simTime=0;selectedPacket=null;updatePanels(true)};
document.getElementById('metaBtn').onclick=()=>document.getElementById('drawer').classList.add('open');
document.getElementById('closeDrawer').onclick=()=>document.getElementById('drawer').classList.remove('open');
renderBenchmark();metadata();nodeList();updatePanels(true);requestAnimationFrame(animate);
</script>
</body>
</html>'''
    return template.replace('__DATA__', data)
