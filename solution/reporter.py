# -*- coding: utf-8 -*-
"""
reporter.py v2  -  Dashboard HTML completo con:
  - Video izquierdo + derecho sincronizados
  - Alertas en tiempo real (feed scrolling)
  - 8 gauges de eficiencia animados
  - Graficos IMU scrolling
  - Tabla de ciclos + despacho + transporte
"""

import json, os
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime

from imu_processor  import LoadCycle, WaitEvent, SessionMetrics, BUCKET_MAX_T, BUCKET_M3
from metrics        import (EfficiencyProfile, Alert, TruckLoadEvent,
                             TransportMetrics, ALERT_SEVERITY)

# ─── PALETA ──────────────────────────────────────────────────────────────────
C = dict(
  bg='#0d1117', panel='#161b22', panel2='#1c2128', border='#30363d',
  text='#e6edf3', muted='#8b949e',
  blue='#4a9eff', orange='#ff8c00', green='#3fb950',
  red='#f85149',  yellow='#ffd700', purple='#bc8cff', cyan='#39d353',
  critical='#f85149', warning='#ffd700', info='#4a9eff', success='#3fb950',
)
PHASE_COLORS = dict(
  DIG='#ff8c00', SWING_LOADED='#4a9eff', DUMP='#f85149',
  SWING_EMPTY='#bc8cff', REPOSITION='#39d353', WAIT='#30363d',
)


def generate_report(df_imu, cycles, wait_events, metrics,
                    video_events, df_cycles, df_waits, output_dir,
                    profiles=None, alerts=None,
                    truck_events=None, transport=None,
                    timeline=None,
                    session_label='JEBI 2026 - EX-5600') -> str:

    html = _build(df_imu, cycles, wait_events, metrics,
                  video_events, profiles or [], alerts or [],
                  truck_events or [], transport,
                  timeline or [], session_label)

    path = os.path.join(output_dir, 'dashboard.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  [reporter] Dashboard guardado: {path}')
    return path


# ─── BUILDER PRINCIPAL ───────────────────────────────────────────────────────

def _build(df_imu, cycles, wait_events, metrics, video_events,
           profiles, alerts, truck_events, transport, timeline, label):

    t   = df_imu.timestamp_s.values.tolist()
    ax  = df_imu.ax.values.tolist()
    ay  = df_imu.ay.values.tolist()
    az  = df_imu.az.values.tolist()
    gx  = df_imu.gx.values.tolist()
    gy  = df_imu.gy.values.tolist()
    gz  = df_imu.gz.values.tolist()
    an  = df_imu.accel_norm.values.tolist()
    gn  = df_imu.gyro_norm.values.tolist()

    full = [c for c in cycles if not c.is_mini_cycle]
    now  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # JSON para JS
    tl_json       = json.dumps(timeline)
    alerts_json   = json.dumps([dict(
        t=a.timestamp_s, type=a.alert_type, sev=a.severity,
        msg=a.message, val=a.value, unit=a.unit, cid=a.cycle_id
    ) for a in alerts])
    shapes_json   = _cycle_shapes(cycles, wait_events)
    payload_json  = _payload_json(full)
    trucks_json   = json.dumps([dict(
        id=te.truck_id, model=te.model, cap=te.capacity_t,
        passes=te.n_passes, payload=te.total_payload_t,
        fill=te.fill_pct, done=te.dispatched
    ) for te in truck_events])

    # HTML building blocks
    kpis_html   = _kpi_cards(metrics, transport)
    eff_html    = _efficiency_table(profiles)
    trucks_html = _trucks_table(truck_events)
    waits_html  = _waits_table(wait_events)
    thumbs_html = _thumbs(video_events)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Shovel Intelligence | {label}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{C['bg']};color:{C['text']};font-family:-apple-system,'Segoe UI',sans-serif;font-size:13px;line-height:1.5}}
::-webkit-scrollbar{{width:6px;height:6px}}::-webkit-scrollbar-track{{background:{C['panel']}}}::-webkit-scrollbar-thumb{{background:{C['border']};border-radius:3px}}

/* HEADER */
.hdr{{background:{C['panel']};border-bottom:1px solid {C['border']};padding:10px 20px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
.hdr h1{{font-size:1.1rem;color:{C['blue']};font-weight:700}}
.hdr .meta{{color:{C['muted']};font-size:0.75rem}}
#live-clock{{color:{C['yellow']};font-weight:700;font-size:0.9rem}}
#phase-badge{{padding:3px 12px;border-radius:12px;font-size:0.75rem;font-weight:700;background:{C['border']};margin-left:10px}}

/* LAYOUT */
.container{{padding:14px 18px;max-width:1800px;margin:0 auto}}
.sec{{margin-bottom:20px}}
.sec-title{{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{C['muted']};border-bottom:1px solid {C['border']};padding-bottom:5px;margin-bottom:12px}}
.row{{display:flex;gap:12px}}
.col-video{{flex:0 0 960px}}
.col-alerts{{flex:1;min-width:280px}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.g4{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px}}
.full{{grid-column:1/-1}}
.panel{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:10px}}

/* KPI */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}}
.kpi{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:12px 14px}}
.kpi-lbl{{font-size:0.65rem;color:{C['muted']};text-transform:uppercase;letter-spacing:.06em}}
.kpi-val{{font-size:1.6rem;font-weight:700;margin-top:3px}}
.kpi-sub{{font-size:0.68rem;color:{C['muted']};margin-top:1px}}
.g{{color:{C['green']}}}.w{{color:{C['yellow']}}}.r{{color:{C['red']}}}.b{{color:{C['blue']}}}
.gauge-section{{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start}}
.gauge-panel{{flex:1 1 560px;display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:12px}}
.explain-panel{{flex:0 1 320px;background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:16px}}
.explain-title{{font-size:0.85rem;font-weight:700;color:{C['text']};margin-bottom:10px}}
.explain-item{{font-size:0.82rem;color:{C['muted']};line-height:1.6;margin-bottom:10px}}
.explain-item strong{{color:{C['text']}}}

/* VIDEO */
.video-wrap{{display:flex;gap:4px;background:#000;border-radius:8px;overflow:hidden}}
.video-wrap video{{flex:1;height:240px;object-fit:cover}}
.video-label{{font-size:0.65rem;color:{C['muted']};margin-bottom:4px}}
.video-overlay{{position:relative}}
.video-overlay .ov{{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.7);border-radius:4px;padding:2px 8px;font-size:0.65rem;color:#fff}}

/* ALERTS */
#alert-feed{{height:260px;overflow-y:auto;display:flex;flex-direction:column-reverse;gap:4px}}
.alert-item{{padding:6px 10px;border-radius:5px;border-left:3px solid;font-size:0.75rem;animation:fadeIn .3s ease}}
.alert-critical{{background:rgba(248,81,73,.12);border-color:{C['critical']};color:{C['critical']}}}
.alert-warning {{background:rgba(255,215,0,.10);border-color:{C['warning']};color:{C['warning']}}}
.alert-info    {{background:rgba(74,158,255,.10);border-color:{C['info']};color:{C['info']}}}
.alert-success {{background:rgba(63,185,80,.12);border-color:{C['success']};color:{C['success']}}}
.alert-time{{font-weight:700;margin-right:6px}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(-4px)}}to{{opacity:1;transform:translateY(0)}}}}

/* GAUGE ROW */
.gauge-row{{display:flex;flex-wrap:wrap;gap:8px;justify-content:space-between}}
.gauge-box{{flex:1;min-width:180px;max-width:240px;background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:8px}}
.gauge-title{{font-size:0.65rem;text-transform:uppercase;color:{C['muted']};letter-spacing:.06em;margin-bottom:4px}}

/* TABLES */
table{{width:100%;border-collapse:collapse;font-size:0.78rem}}
th{{background:{C['panel2']};color:{C['muted']};font-size:0.65rem;text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;text-align:left;border-bottom:1px solid {C['border']}}}
td{{padding:5px 8px;border-bottom:1px solid {C['border']}}}
tr:hover td{{background:rgba(255,255,255,.02)}}
.badge{{display:inline-block;padding:1px 7px;border-radius:4px;font-size:.65rem;font-weight:600;text-transform:uppercase}}
.bg{{background:rgba(63,185,80,.15);color:{C['green']}}}.bw{{background:rgba(255,215,0,.15);color:{C['yellow']}}}.br{{background:rgba(248,81,73,.15);color:{C['red']}}}.bm{{background:rgba(188,140,255,.15);color:{C['purple']}}}.bb{{background:rgba(74,158,255,.15);color:{C['blue']}}}

/* PLAYBACK */
#playbar{{display:flex;align-items:center;gap:10px;background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:8px 14px}}
#play-btn{{background:{C['blue']};color:#fff;border:none;border-radius:5px;padding:5px 16px;cursor:pointer;font-size:0.8rem;font-weight:600}}
#play-btn:hover{{background:#3a8ee0}}
#progress{{flex:1;height:4px;-webkit-appearance:none;border-radius:2px;background:{C['border']};cursor:pointer}}
#progress::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:{C['blue']};cursor:pointer}}
#t-display{{font-size:0.8rem;color:{C['yellow']};min-width:60px;text-align:right;font-weight:700}}
.speed-btn{{background:{C['panel2']};color:{C['muted']};border:1px solid {C['border']};border-radius:4px;padding:3px 8px;cursor:pointer;font-size:0.72rem}}
.speed-btn.active{{background:{C['blue']};color:#fff;border-color:{C['blue']}}}

/* THUMBS */
.thumb-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}}
.thumb-card{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;overflow:hidden}}
.thumb-card img{{width:100%;display:block}}
.thumb-info{{padding:7px 10px;font-size:0.72rem;color:{C['muted']}}}
.thumb-info strong{{color:{C['blue']}}}

/* PHASE indicator */
.phase-DIG{{background:rgba(255,140,0,.15);color:{C['orange']}}}
.phase-SWING_LOADED{{background:rgba(74,158,255,.15);color:{C['blue']}}}
.phase-DUMP{{background:rgba(248,81,73,.15);color:{C['red']}}}
.phase-SWING_EMPTY{{background:rgba(188,140,255,.15);color:{C['purple']}}}
.phase-WAIT{{background:rgba(48,54,61,.5);color:{C['muted']}}}
.phase-REPOSITION{{background:rgba(57,211,83,.15);color:{C['green']}}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div>
    <h1>Shovel Intelligence Dashboard</h1>
    <div class="meta">JEBI Hackathon 2026 &nbsp;|&nbsp; Hitachi EX-5600 &nbsp;|&nbsp;
      CAT 793F (218t) &nbsp;&amp;&nbsp; EH4000 AC-3 (221t) &nbsp;|&nbsp; {now}
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span id="phase-badge" class="badge">WAIT</span>
    <span id="live-clock">t = 0.0 s</span>
  </div>
</div>

<div class="container">

<!-- PLAYBACK CONTROL -->
<div class="sec">
  <div id="playbar">
    <button id="play-btn" onclick="togglePlay()">&#9654; PLAY</button>
    <input type="range" id="progress" min="0" max="{len(timeline)-1}" value="0"
           oninput="seekTo(parseInt(this.value))">
    <span id="t-display">0.0 s</span>
    <span style="color:{C['muted']};font-size:.72rem">Velocidad:</span>
    <button class="speed-btn active" onclick="setSpeed(1,this)">1x</button>
    <button class="speed-btn" onclick="setSpeed(5,this)">5x</button>
    <button class="speed-btn" onclick="setSpeed(15,this)">15x</button>
    <button class="speed-btn" onclick="setSpeed(30,this)">30x</button>
  </div>
</div>

<!-- VIDEO + ALERTAS -->
<div class="sec row">
  <div class="col-video">
    <div class="sec-title">Video Estereo &mdash; Camara Izquierda / Derecha</div>
    <div class="video-wrap">
      <div class="video-overlay" style="flex:1">
        <div class="video-label">CAM IZQUIERDA</div>
        <video id="vid-left" src="../inputs/shovel_left.mp4"
               muted playsinline preload="auto"
               style="width:100%;height:240px;object-fit:cover;border-radius:6px"></video>
        <div class="ov" id="vid-left-info">t=0.0s</div>
      </div>
      <div class="video-overlay" style="flex:1">
        <div class="video-label">CAM DERECHA</div>
        <video id="vid-right" src="../inputs/shovel_right.mp4"
               muted playsinline preload="auto"
               style="width:100%;height:240px;object-fit:cover;border-radius:6px"></video>
        <div class="ov" id="vid-right-info">t=0.0s</div>
      </div>
    </div>
    <!-- Sensor overlay on video -->
    <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap">
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Accel Norm</div>
        <div class="kpi-val b" id="hud-an">—</div>
        <div class="kpi-sub">m/s²</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Gyro Norm</div>
        <div class="kpi-val b" id="hud-gn">—</div>
        <div class="kpi-sub">deg/s</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Fill Factor</div>
        <div class="kpi-val" id="hud-fill">—</div>
        <div class="kpi-sub">% bucket</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Payload</div>
        <div class="kpi-val" id="hud-payload">—</div>
        <div class="kpi-sub">toneladas</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Volumen</div>
        <div class="kpi-val" id="hud-vol">—</div>
        <div class="kpi-sub">m³</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Ciclo #</div>
        <div class="kpi-val b" id="hud-cid">—</div>
        <div class="kpi-sub">activo</div>
      </div>
    </div>
  </div>

  <!-- ALERT FEED -->
  <div class="col-alerts">
    <div class="sec-title">Alertas en Tiempo Real
      <span id="alert-count" style="float:right;color:{C['red']};font-weight:700">0 alertas</span>
    </div>
    <div class="panel" style="padding:8px">
      <div id="alert-feed"></div>
    </div>
    <div style="margin-top:8px">
      <div class="sec-title">Fase Actual</div>
      <div class="panel">
        <div id="phase-detail" style="font-size:0.82rem;padding:4px 0">
          Esperando datos...
        </div>
        <div id="phase-bar" style="height:6px;border-radius:3px;background:{C['border']};margin-top:8px;transition:width .3s,background .3s"></div>
      </div>
    </div>
  </div>
</div>

<!-- GAUGES DE EFICIENCIA -->
<div class="sec">
  <div class="sec-title">Eficiencias Operacionales en Tiempo Real</div>
  <div class="gauge-section">
    <div class="gauge-panel">
      <div class="gauge-box">
        <div class="gauge-title">Recoleccion (Fill)</div>
        <div id="g-coll" style="height:100px"></div>
      </div>
      <div class="gauge-box">
        <div class="gauge-title">Maniobra</div>
        <div id="g-man" style="height:100px"></div>
      </div>
      <div class="gauge-box">
        <div class="gauge-title">Posicionamiento</div>
        <div id="g-pos" style="height:100px"></div>
      </div>
      <div class="gauge-box">
        <div class="gauge-title">Descarga</div>
        <div id="g-disc" style="height:100px"></div>
      </div>
      <div class="gauge-box">
        <div class="gauge-title">OEE Ciclo</div>
        <div id="g-oee" style="height:100px"></div>
      </div>
      <div class="gauge-box">
        <div class="gauge-title">Tiempo Util</div>
        <div id="g-teff" style="height:100px"></div>
      </div>
      <div class="gauge-box">
        <div class="gauge-title">Despacho</div>
        <div id="g-disp" style="height:100px"></div>
      </div>
      <div class="gauge-box">
        <div class="gauge-title">Carga Camion</div>
        <div id="g-truck" style="height:100px"></div>
      </div>
    </div>
    <div class="explain-panel">
      <div class="explain-title">¿Qué significa cada indicador?</div>
      <div class="explain-item"><strong>Recolección:</strong> cómo de lleno quedó el bucket en cada ciclo.</div>
      <div class="explain-item"><strong>Maniobra:</strong> porcentaje de tiempo activo en el movimiento del ciclo.</div>
      <div class="explain-item"><strong>Posicionamiento:</strong> velocidad para alcanzar la postura óptima antes del dig.</div>
      <div class="explain-item"><strong>Descarga:</strong> suavidad y calidad del vuelco del material.</div>
      <div class="explain-item"><strong>OEE Ciclo:</strong> eficiencia combinada del ciclo completo.</div>
      <div class="explain-item"><strong>Tiempo Útil:</strong> proporción de la sesión sin esperas largas.</div>
      <div class="explain-item"><strong>Despacho:</strong> qué tan eficiente es el servicio de camiones respecto a su llenado.</div>
      <div class="explain-item"><strong>Carga Camión:</strong> cuánto porcentaje de la capacidad del camión se llenó.</div>
    </div>
  </div>
  </div>
</div>

<!-- GRAFICOS IMU SCROLLING -->
<div class="sec">
  <div class="sec-title">Sensores IMU &mdash; Ventana Deslizante 30s</div>
  <div class="g2">
    <div class="panel full">
      <div id="chart-accel" style="height:160px"></div>
    </div>
    <div class="panel full">
      <div id="chart-gyro" style="height:160px"></div>
    </div>
  </div>
</div>

<!-- KPIs GLOBALES -->
<div class="sec">
  <div class="sec-title">KPIs de la Sesion</div>
  {kpis_html}
</div>

<!-- DETALLE DE CICLOS -->
<div class="sec">
  <div class="sec-title">Detalle por Ciclo &mdash; Eficiencias y Alertas</div>
  {eff_html}
</div>

<!-- DESPACHO Y CAMIONES -->
<div class="sec g2">
  <div>
    <div class="sec-title">Carga por Camion &mdash; Despacho</div>
    {trucks_html}
  </div>
  <div>
    <div class="sec-title">Tiempos de Espera</div>
    {waits_html}
  </div>
</div>

<!-- THUMBNAILS -->
<div class="sec">
  <div class="sec-title">Capturas de Video &mdash; Backtracking en Eventos Clave</div>
  {thumbs_html}
</div>

</div><!-- /container -->

<!-- ═══════════════════════════════════════════════════════════════
     JAVASCRIPT  —  Real-time simulation engine
═══════════════════════════════════════════════════════════════ -->
<script>
// ── DATA ────────────────────────────────────────────────────────────────────
const TL        = {tl_json};
const ALL_ALERTS= {alerts_json};
const SHAPES    = {shapes_json};
const PAYLOAD_D = {payload_json};
const TRUCKS    = {trucks_json};

const T_FULL    = {t};
const AX_FULL   = {ax};
const AY_FULL   = {ay};
const AZ_FULL   = {az};
const GX_FULL   = {gx};
const GY_FULL   = {gy};
const GZ_FULL   = {gz};
const AN_FULL   = {an};
const GN_FULL   = {gn};

const SESSION_METRICS = {{
  duration_s:    {metrics.total_duration_s},
  time_eff:      {metrics.time_efficiency_pct},
  fill_avg:      {metrics.avg_fill_factor_pct},
  cycles_hr:     {metrics.cycles_per_hour},
  productivity:  {metrics.productivity_tph},
  total_payload: {metrics.total_payload_t},
}};

const TRANSPORT = {json.dumps(dict(
    n_trucks=transport.n_trucks_served if transport else 0,
    n_full=transport.n_trucks_full if transport else 0,
    disp_eff=transport.dispatch_efficiency if transport else 0,
    avg_fill=transport.avg_truck_fill_pct if transport else 0,
) if transport else dict(n_trucks=0,n_full=0,disp_eff=0,avg_fill=0))};

const DARK_LAYOUT = {{
  paper_bgcolor: '{C['bg']}', plot_bgcolor: '{C['panel']}',
  font: {{color: '{C['text']}', size:10}},
  margin: {{l:38,r:8,t:22,b:28}},
  xaxis: {{gridcolor:'{C['border']}', zerolinecolor:'{C['border']}'}},
  yaxis: {{gridcolor:'{C['border']}', zerolinecolor:'{C['border']}'}},
  legend: {{bgcolor:'rgba(0,0,0,.3)',bordercolor:'{C['border']}',borderwidth:1,font:{{size:9}}}},
}};

// ── STATE ───────────────────────────────────────────────────────────────────
let curIdx   = 0;
let playing  = false;
let playTimer= null;
let speed    = 1;        // steps per tick
let alertsShown = new Set();
let totalAlerts = 0;

// Last known values for HUD
let lastFillPct = null;
let lastPayloadT = null;
let lastVolumeM3 = null;
let lastCycleId = null;

const vidL = document.getElementById('vid-left');
const vidR = document.getElementById('vid-right');

// ── PLAYBACK ────────────────────────────────────────────────────────────────
function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').innerHTML = playing
    ? '&#9646;&#9646; PAUSE' : '&#9654; PLAY';

  if (playing) {{
    if (vidL.paused) vidL.play().catch(()=>{{}});
    if (vidR.paused) vidR.play().catch(()=>{{}});
    playTimer = setInterval(tick, 100);
  }} else {{
    vidL.pause(); vidR.pause();
    clearInterval(playTimer);
  }}
}}

function setSpeed(s, btn) {{
  speed = s;
  document.querySelectorAll('.speed-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  // Ajustar playback rate del video proporcional
  const vr = Math.min(s, 16);
  vidL.playbackRate = vr; vidR.playbackRate = vr;
}}

function tick() {{
  curIdx = Math.min(curIdx + speed, TL.length - 1);
  update(curIdx);
  if (curIdx >= TL.length - 1) {{
    playing = false;
    document.getElementById('play-btn').innerHTML = '&#9654; PLAY';
    clearInterval(playTimer);
    vidL.pause(); vidR.pause();
  }}
}}

function seekTo(idx) {{
  curIdx = Math.min(Math.max(0, idx), TL.length - 1);
  update(curIdx);
}}

// ── MAIN UPDATE ─────────────────────────────────────────────────────────────
function update(idx) {{
  const snap = TL[idx];
  const t    = snap.t;

  // Progress bar + clock
  document.getElementById('progress').value   = idx;
  document.getElementById('t-display').textContent = t.toFixed(1) + ' s';
  document.getElementById('live-clock').textContent = 't = ' + t.toFixed(1) + ' s';

  // Sync video (if not playing — when playing, video runs on its own)
  if (!playing) {{
    if (Math.abs(vidL.currentTime - t) > 0.5) vidL.currentTime = t;
    if (Math.abs(vidR.currentTime - t) > 0.5) vidR.currentTime = t;
  }}
  document.getElementById('vid-left-info').textContent  = 't=' + t.toFixed(1) + 's';
  document.getElementById('vid-right-info').textContent = 't=' + t.toFixed(1) + 's';

  // Update last known values
  if (snap.fill_pct != null) lastFillPct = snap.fill_pct;
  if (snap.payload_t != null) lastPayloadT = snap.payload_t;
  if (snap.volume_m3 != null) lastVolumeM3 = snap.volume_m3;
  if (snap.cycle_id != null) lastCycleId = snap.cycle_id;

  // HUD numbers
  _hud('hud-an',      snap.accel_norm, 'm/s²', 12, 20);
  _hud('hud-gn',      snap.gyro_norm,  'deg/s', 20, 60);
  _hud('hud-fill',    lastFillPct,     '%',     80, 95);
  _hud('hud-payload', lastPayloadT || 0, 't',  null, null);
  _hud('hud-vol',     lastVolumeM3 || 0, 'm³', null, null);
  document.getElementById('hud-cid').textContent = lastCycleId ? '#'+lastCycleId : '—';

  // Phase badge
  const ph   = snap.phase || 'WAIT';
  const pbEl = document.getElementById('phase-badge');
  pbEl.textContent  = ph.replace('_',' ');
  pbEl.className    = 'badge phase-' + ph;
  document.getElementById('phase-detail').innerHTML =
    '<b>' + ph.replace('_',' ') + '</b>' +
    (lastCycleId ? '  &mdash; Ciclo #' + lastCycleId : '') +
    '<br><span style="color:{C['muted']}">Fill: ' + (lastFillPct!=null?lastFillPct.toFixed(1):'—') +
    '%  |  Payload: ' + ((lastPayloadT || 0).toFixed(1)) + ' t</span>';

  // Gauges
  const deflt = 50;
  _gauge('g-coll',  snap.coll_eff    ?? deflt, 'Recol.', '%');
  _gauge('g-man',   snap.maneuver_eff?? deflt, 'Maniob.','%');
  _gauge('g-pos',   snap.pos_eff     ?? deflt, 'Posic.', '%');
  _gauge('g-disc',  snap.disc_eff    ?? deflt, 'Descarg','%');
  _gauge('g-oee',   snap.cycle_eff   ?? deflt, 'OEE',    '%');
  _gauge('g-teff',  SESSION_METRICS.time_eff,  'T.Util', '%');
  _gauge('g-disp',  TRANSPORT.disp_eff,        'Desp.', '%');
  const truckFill = TRANSPORT.avg_fill ?? 0;
  _gauge('g-truck', truckFill, 'Camion','%');

  // IMU scrolling graphs
  _updateScrollingCharts(t);

  // Alerts
  _processAlerts(t, snap.alerts || []);
}}

// ── HUD helper ───────────────────────────────────────────────────────────────
function _hud(id, val, unit, warnLow, warnHigh) {{
  const el = document.getElementById(id);
  if (val == null) {{ el.textContent = '—'; el.className='kpi-val'; return; }}
  el.textContent = typeof val === 'number' ? val.toFixed(1) : val;
  if (warnLow  != null && val < warnLow)  el.className='kpi-val r';
  else if (warnHigh != null && val > warnHigh) el.className='kpi-val w';
  else el.className='kpi-val g';
}}

// ── GAUGE ────────────────────────────────────────────────────────────────────
const gaugeInstances = {{}};
function _gauge(id, value, title, unit) {{
  const color = value >= 85 ? '{C['green']}' : value >= 65 ? '{C['yellow']}' : '{C['red']}';
  const data  = [{{ type:'indicator', mode:'gauge+number',
    value: value,
    number: {{ suffix: unit, font:{{size:16,color:color}} }},
    gauge: {{
      axis: {{ range:[0,100], tickfont:{{size:8}}, tickcolor:'{C['muted']}' }},
      bar:  {{ color: color, thickness:.6 }},
      bgcolor: '{C['panel2']}',
      bordercolor: '{C['border']}',
      steps: [
        {{ range:[0,65], color:'rgba(248,81,73,.08)' }},
        {{ range:[65,85], color:'rgba(255,215,0,.08)' }},
        {{ range:[85,100], color:'rgba(63,185,80,.08)' }},
      ],
    }},
  }}];
  const layout = {{ ...DARK_LAYOUT, margin:{{l:10,r:10,t:15,b:5}}, height:100 }};
  if (gaugeInstances[id]) {{
    Plotly.react(id, data, layout, {{displayModeBar:false}});
  }} else {{
    Plotly.newPlot(id, data, layout, {{displayModeBar:false}});
    gaugeInstances[id] = true;
  }}
}}

// ── SCROLLING CHARTS ─────────────────────────────────────────────────────────
let accelInited = false, gyroInited = false;
const WIN_S = 30;

function _updateScrollingCharts(t_now) {{
  const iEnd   = T_FULL.findIndex(x => x >= t_now);
  const iStart = T_FULL.findIndex(x => x >= t_now - WIN_S);
  const i0 = Math.max(0, iStart < 0 ? 0 : iStart);
  const i1 = iEnd < 0 ? T_FULL.length - 1 : iEnd;

  const ts = T_FULL.slice(i0, i1+1);

  const accelTraces = [
    {{ x:ts, y:AX_FULL.slice(i0,i1+1), name:'accel_x', mode:'lines', line:{{color:'{C['blue']}',width:1}}}},
    {{ x:ts, y:AY_FULL.slice(i0,i1+1), name:'accel_y', mode:'lines', line:{{color:'{C['orange']}',width:1}}}},
    {{ x:ts, y:AZ_FULL.slice(i0,i1+1), name:'accel_z', mode:'lines', line:{{color:'{C['green']}',width:1}}}},
  ];
  const gyroTraces = [
    {{ x:ts, y:GX_FULL.slice(i0,i1+1), name:'gyro_x', mode:'lines', line:{{color:'{C['blue']}',width:1}}}},
    {{ x:ts, y:GY_FULL.slice(i0,i1+1), name:'gyro_y', mode:'lines', line:{{color:'{C['orange']}',width:1}}}},
    {{ x:ts, y:GZ_FULL.slice(i0,i1+1), name:'gyro_z', mode:'lines', line:{{color:'{C['green']}',width:1}}}},
  ];

  const tRange = [t_now - WIN_S, t_now];
  const aLayout = {{ ...DARK_LAYOUT, title:{{text:'Accelerometer (m/s²)',font:{{size:11}}}},
                     xaxis:{{ ...DARK_LAYOUT.xaxis, range:tRange }}, height:160 }};
  const gLayout = {{ ...DARK_LAYOUT, title:{{text:'Gyroscope (deg/s)',font:{{size:11}}}},
                     xaxis:{{ ...DARK_LAYOUT.xaxis, range:tRange }}, height:160 }};

  if (!accelInited) {{ Plotly.newPlot('chart-accel',accelTraces,aLayout,{{displayModeBar:false}}); accelInited=true; }}
  else Plotly.react('chart-accel', accelTraces, aLayout);
  if (!gyroInited)  {{ Plotly.newPlot('chart-gyro',gyroTraces,gLayout,{{displayModeBar:false}}); gyroInited=true; }}
  else Plotly.react('chart-gyro', gyroTraces, gLayout);
}}

// ── ALERTS ───────────────────────────────────────────────────────────────────
function _processAlerts(t_now, snap_alerts) {{
  // Alertas del timeline snapshot
  snap_alerts.forEach(a => {{
    const key = a.type + '_' + Math.floor(t_now);
    if (!alertsShown.has(key)) {{
      alertsShown.add(key);
      _addAlert(t_now, a.type, a.sev || 'info', a.msg, a.val, a.unit);
    }}
  }});

  // Alertas pre-computadas
  ALL_ALERTS.forEach(a => {{
    const key = a.type + '_' + Math.floor(a.t);
    if (a.t <= t_now && !alertsShown.has(key)) {{
      alertsShown.add(key);
      _addAlert(a.t, a.type, a.sev, a.msg, a.val, a.unit);
    }}
  }});
}}

function _addAlert(t, type, sev, msg, val, unit) {{
  totalAlerts++;
  document.getElementById('alert-count').textContent = totalAlerts + ' alertas';

  const feed = document.getElementById('alert-feed');
  const el   = document.createElement('div');
  el.className = 'alert-item alert-' + (sev||'info');
  el.innerHTML =
    '<span class="alert-time">t=' + t.toFixed(1) + 's</span>' +
    '<b>' + type + '</b>: ' + msg +
    (val != null ? ' <span style="opacity:.8">['+val+' '+(unit||'')+']</span>' : '');
  feed.insertBefore(el, feed.firstChild);

  // Mantener max 50 items
  while (feed.children.length > 50) feed.removeChild(feed.lastChild);
}}

// ── INIT ─────────────────────────────────────────────────────────────────────
// Initialize last values
lastFillPct = null;
lastPayloadT = 0;
lastVolumeM3 = 0;
lastCycleId = null;

// Inicializar gauges en 50%
['g-coll','g-man','g-pos','g-disc','g-oee','g-teff','g-disp','g-truck']
  .forEach((id,i) => _gauge(id, 50, '', '%'));

// Inicializar graficos IMU vacios
Plotly.newPlot('chart-accel',[],{{...DARK_LAYOUT,height:160,title:{{text:'Accelerometer'}}}},{{displayModeBar:false}});
accelInited=true;
Plotly.newPlot('chart-gyro', [],{{...DARK_LAYOUT,height:160,title:{{text:'Gyroscope'}}}},  {{displayModeBar:false}});
gyroInited=true;

// Render frame 0
update(0);
</script>
</body>
</html>"""


# ─── HTML HELPERS ─────────────────────────────────────────────────────────────

def _kpi_cards(m: SessionMetrics, t: TransportMetrics) -> str:
    def kpi(lbl, val, sub='', cls=''):
        return (f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                f'<div class="kpi-val {cls}">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>')

    fc = 'g' if m.avg_fill_factor_pct >= 85 else ('w' if m.avg_fill_factor_pct >= 65 else 'r')
    tc = 'g' if m.time_efficiency_pct >= 75 else 'w'
    tr_c = 'g' if (t and t.dispatch_efficiency >= 80) else 'w' if t else ''

    tr_trucks = t.n_trucks_served if t else '—'
    tr_full   = t.n_trucks_full   if t else '—'
    tr_eff    = f'{t.dispatch_efficiency:.0f}%' if t else '—'
    tr_fill   = f'{t.avg_truck_fill_pct:.1f}%' if t else '—'

    return f"""<div class="kpi-grid">
      {kpi('Ciclos Completos', m.n_full_cycles, 'sesion', 'g')}
      {kpi('Mini-Ciclos', m.n_mini_cycles, 'correcciones', 'w')}
      {kpi('Fill Factor Prom', f'{m.avg_fill_factor_pct:.1f}%', 'bucket', fc)}
      {kpi('Payload Total', f'{m.total_payload_t:.0f}t', 'estimado', 'g')}
      {kpi('Productividad', f'{m.productivity_tph:.0f} t/h', 'ton/hora', 'g')}
      {kpi('Ciclos/Hora', f'{m.cycles_per_hour:.1f}', 'ciclos', 'b')}
      {kpi('Efic. Temporal', f'{m.time_efficiency_pct:.1f}%', 'vs esperas', tc)}
      {kpi('Wait Total', f'{m.total_wait_s/60:.1f} min', f'{m.n_wait_events} eventos', 'w')}
      {kpi('Underfill &lt;80%', m.underfill_count, f'-{m.underfill_loss_t:.0f}t perdidas', 'r')}
      {kpi('Camiones', str(tr_trucks), 'servidos', 'b')}
      {kpi('Camiones Llenos', str(tr_full), 'completados', 'g')}
      {kpi('Efic. Despacho', tr_eff, 'completados/total', tr_c)}
    </div>"""


def _efficiency_table(profiles: List[EfficiencyProfile]) -> str:
    if not profiles:
        return '<p style="color:#8b949e">Sin datos de eficiencia.</p>'
    rows = ''
    for p in profiles:
        oee_c = 'bg' if p.cycle_eff_pct >= 75 else ('bw' if p.cycle_eff_pct >= 55 else 'br')
        fil_c = 'bg' if p.collection_eff_pct >= 85 else ('bw' if p.collection_eff_pct >= 65 else 'br')
        al_badges = ''.join(
            f'<span class="badge b{a.severity[0] if a.severity!="success" else "g"}" '
            f'title="{a.message}">{a.alert_type}</span> '
            for a in p.alerts[:3]
        )
        rows += (
            f'<tr>'
            f'<td>#{p.cycle_id}</td>'
            f'<td>{p.t_start:.1f}s</td>'
            f'<td><span class="badge {fil_c}">{p.collection_eff_pct:.0f}%</span></td>'
            f'<td>{p.volume_m3:.1f} m³</td>'
            f'<td>{p.maneuver_eff_pct:.0f}%</td>'
            f'<td>{p.positioning_eff_pct:.0f}%</td>'
            f'<td>{p.discharge_eff_pct:.0f}%</td>'
            f'<td><span class="badge {oee_c}">{p.cycle_eff_pct:.0f}%</span></td>'
            f'<td>{al_badges or "—"}</td>'
            f'</tr>\n'
        )
    return f"""<table>
      <thead><tr>
        <th>#</th><th>T.Inicio</th><th>Recol.</th><th>Volumen</th>
        <th>Maniobra</th><th>Posic.</th><th>Descarga</th><th>OEE</th><th>Alertas</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _trucks_table(events: List[TruckLoadEvent]) -> str:
    if not events:
        return '<p style="color:#8b949e">Sin camiones identificados.</p>'
    rows = ''
    for te in events:
        fc = 'bg' if te.fill_pct >= 90 else ('bw' if te.fill_pct >= 75 else 'br')
        st = '<span class="badge bg">DESPACHADO</span>' if te.dispatched else '<span class="badge bw">PARCIAL</span>'
        rows += (
            f'<tr>'
            f'<td><b>{te.truck_id}</b></td>'
            f'<td>{te.model}</td>'
            f'<td>{te.capacity_t:.0f}t</td>'
            f'<td>{te.n_passes}</td>'
            f'<td>{te.total_payload_t:.1f}t</td>'
            f'<td><span class="badge {fc}">{te.fill_pct:.0f}%</span></td>'
            f'<td>{st}</td>'
            f'</tr>\n'
        )
    return f"""<table>
      <thead><tr>
        <th>ID Camion</th><th>Modelo</th><th>Cap.</th>
        <th>Pasadas</th><th>Payload</th><th>% Lleno</th><th>Estado</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _waits_table(waits) -> str:
    if not waits:
        return '<p style="color:#8b949e">Sin esperas significativas.</p>'
    rows = ''
    for w in waits:
        rb = dict(pre_load='bb',inter_cycle='bw',end_session='bm').get(w.reason,'bm')
        rows += (
            f'<tr>'
            f'<td>{w.t_start:.1f}s</td>'
            f'<td>{w.duration_s:.1f}s</td>'
            f'<td><span class="badge {rb}">{w.reason.replace("_"," ").upper()}</span></td>'
            f'</tr>\n'
        )
    return f"""<table>
      <thead><tr><th>T.Inicio</th><th>Duracion</th><th>Tipo</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _thumbs(events) -> str:
    if not events:
        return '<p style="color:#8b949e">Sin thumbnails disponibles.</p>'
    cards = ''
    for ev in events[:12]:  # max 12
        b64  = ev.get('thumb_b64','')
        img  = f'<img src="data:image/jpeg;base64,{b64}" alt="frame">' if b64 else ''
        cards += (
            f'<div class="thumb-card">{img}'
            f'<div class="thumb-info">'
            f'<strong>Ciclo #{ev["cycle_id"]} | {ev["event_type"].upper()}</strong> '
            f't={ev["timestamp_s"]:.1f}s<br>'
            f'Camion: {ev.get("truck_id","—")} | {ev.get("truck_model","—")} | '
            f'Conf: {ev.get("confidence",0)*100:.0f}%'
            f'</div></div>'
        )
    return f'<div class="thumb-grid">{cards}</div>'


def _cycle_shapes(cycles, waits) -> str:
    sh = []
    for c in cycles:
        col = 'rgba(188,140,255,.10)' if c.is_mini_cycle else 'rgba(74,158,255,.08)'
        sh.append(dict(type='rect',xref='x',yref='paper',
                       x0=c.t_start,x1=c.t_end,y0=0,y1=1,
                       fillcolor=col,line=dict(width=0)))
    for w in waits:
        sh.append(dict(type='rect',xref='x',yref='paper',
                       x0=w.t_start,x1=w.t_end,y0=0,y1=1,
                       fillcolor='rgba(139,148,158,.06)',line=dict(width=0)))
    return json.dumps(sh)


def _payload_json(full_cycles) -> str:
    ids,p,f,pc,fc,pl,fl=[],[],[],[],[],[],[]
    for c in full_cycles:
        ids.append(c.cycle_id); p.append(round(c.payload_t,1))
        fp=round(c.fill_factor*100,1); f.append(fp)
        col=C['green'] if fp>=85 else(C['yellow'] if fp>=65 else C['red'])
        pc.append(col); fc.append(col)
        pl.append(f'{c.payload_t:.0f}t'); fl.append(f'{fp:.0f}%')
    return json.dumps(dict(cycle_ids=ids,payloads=p,fills=f,
                           colors=pc,fill_colors=fc,labels=pl,fill_labels=fl))
