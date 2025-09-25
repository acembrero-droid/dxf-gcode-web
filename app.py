import streamlit as st
import ezdxf
import math
import io
import plotly.graph_objects as go
import os

# ======= Configuración por defecto =======
CUT_FEED_DEFAULT = 100        # mm/min
PAUSE_DEFAULT_MS = 5          # ms por mm
ARC_SEGMENTS_DEFAULT = 24     # resolución arcos en preview
JOIN_TOL_DEFAULT = 0.01       # mm, tolerancia para unir vértices
paths = []                    # segmentos "atómicos" (línea / arco)
chains = []                   # lista de cadenas continuas (índices + reverse)

# -------- Utilidades geométricas --------
def dist(a,b): return math.hypot(b[0]-a[0], b[1]-a[1])
def deg(a):    return math.degrees(a)
def rad(a):    return math.radians(a)
def rot90_ccw(v): return (-v[1], v[0])
def unit(v):
    l = math.hypot(v[0], v[1])
    return (0.0,0.0) if l==0 else (v[0]/l, v[1]/l)

def ang_of(p, c):  # ángulo de vector (p - c)
    return deg(math.atan2(p[1]-c[1], p[0]-c[0]))

def signed_sweep_deg(a0, a1, ccw=True):
    d = (a1 - a0) % 360.0
    return d if ccw else d - 360.0

def point_key(p, tol):
    return (round(p[0]/tol), round(p[1]/tol))

# ======= Conversión DXF → segmentos =======
def add_line(p0, p1):
    paths.append({"type":"line", "start":p0, "end":p1})

def add_arc_from_center(p0, p1, center):
    cx, cy = center
    r = dist(p0, center)
    a0 = ang_of(p0, center)
    a1 = ang_of(p1, center)
    v0 = (p0[0]-cx, p0[1]-cy)
    v1 = (p1[0]-cx, p1[1]-cy)
    ccw = (v0[0]*v1[1] - v0[1]*v1[0]) > 0
    paths.append({
        "type":"arc",
        "center":(cx, cy),
        "radius":r,
        "start":p0, "end":p1,
        "start_angle":a0, "end_angle":a1,
        "ccw":ccw
    })

def add_arc_from_bulge(p0, p1, b):
    if abs(b) < 1e-12:
        add_line(p0, p1); return
    x0,y0 = p0; x1,y1 = p1
    chord = (x1-x0, y1-y0)
    c = math.hypot(chord[0], chord[1])
    if c < 1e-12:
        return
    theta = 4.0*math.atan(b)
    sin_half = math.sin(theta/2.0)
    if abs(sin_half) < 1e-12:
        add_line(p0, p1); return
    R = c/(2.0*abs(sin_half))
    mid = ((x0+x1)/2.0, (y0+y1)/2.0)
    n = rot90_ccw(unit(chord))
    d = R*math.cos(theta/2.0)
    d = d if b > 0 else -d
    center = (mid[0] + n[0]*d, mid[1] + n[1]*d)
    add_arc_from_center(p0, p1, center)

def load_dxf(file):
    global paths
    paths = []
    doc = ezdxf.readfile(file)
    msp = doc.modelspace()
    for e in msp:
        dxft = e.dxftype()
        if dxft == "LINE":
            add_line((e.dxf.start[0], e.dxf.start[1]),
                     (e.dxf.end[0],   e.dxf.end[1]))
        elif dxft == "ARC":
            c = (e.dxf.center[0], e.dxf.center[1])
            r = e.dxf.radius
            a0 = e.dxf.start_angle
            a1 = e.dxf.end_angle
            p0 = (c[0] + r*math.cos(rad(a0)), c[1] + r*math.sin(rad(a0)))
            p1 = (c[0] + r*math.cos(rad(a1)), c[1] + r*math.sin(rad(a1)))
            add_arc_from_center(p0, p1, c)
        elif dxft == "LWPOLYLINE":
            try:
                pts = list(e.get_points("xyb"))
            except TypeError:
                pts = [(p[0], p[1], p[4] if len(p) > 4 else 0.0) for p in e.get_points()]
            for i in range(len(pts)-1):
                p0 = (pts[i][0],   pts[i][1])
                p1 = (pts[i+1][0], pts[i+1][1])
                b  = float(pts[i][2])
                add_arc_from_bulge(p0, p1, b)
            if e.closed:
                p0 = (pts[-1][0], pts[-1][1])
                p1 = (pts[0][0],  pts[0][1])
                b  = float(pts[-1][2])
                add_arc_from_bulge(p0, p1, b)

# ======= Construir cadenas continuas (topología) =======
def build_chains(join_tol):
    nodemap = {}
    def register_endpoint(idx, pt, is_start):
        k = point_key(pt, join_tol)
        nodemap.setdefault(k, {"pt": pt, "inc": []})
        nodemap[k]["inc"].append((idx, is_start))

    for i, s in enumerate(paths):
        register_endpoint(i, s["start"], True)
        register_endpoint(i, s["end"], False)

    used = [False]*len(paths)
    result = []

    def find_seed():
        for k, nd in nodemap.items():
            avail = [(i, st) for (i, st) in nd["inc"] if not used[i]]
            if len(avail) == 1:
                return k
        for k, nd in nodemap.items():
            for (i, st) in nd["inc"]:
                if not used[i]:
                    return k
        return None

    def other_endpoint(seg_idx, at_start):
        s = paths[seg_idx]
        return s["end"] if at_start else s["start"]

    while True:
        seed_key = find_seed()
        if seed_key is None:
            break
        chain = []
        curr_key = seed_key
        while True:
            inc = nodemap[curr_key]["inc"]
            cand = [(i, st) for (i, st) in inc if not used[i]]
            if not cand:
                break
            seg_idx, at_start = cand[0]
            used[seg_idx] = True
            reverse = (not at_start)
            chain.append((seg_idx, reverse))
            seg = paths[seg_idx]
            nxt_pt = other_endpoint(seg_idx, at_start)
            curr_key = point_key(nxt_pt, join_tol)
        if chain:
            result.append(chain)

    return result

# ======= Generar G-code continuo =======
def generar_gcode(cut_feed, pause_ms, join_tol):
    global chains
    chains = build_chains(join_tol)

    # === Forzar inicio en el punto más a la izquierda ===
    min_x = float("inf")
    start_chain_idx = 0
    start_seg_idx = 0
    start_reverse = False

    for ci, chain in enumerate(chains):
        for si, (seg_idx, reverse) in enumerate(chain):
            seg = paths[seg_idx]
            p0 = seg["end"] if reverse else seg["start"]
            if p0[0] < min_x:
                min_x = p0[0]
                start_chain_idx = ci
                start_seg_idx = si
                start_reverse = reverse

    if chains:
        chain = chains[start_chain_idx]
        new_chain = chain[start_seg_idx:] + chain[:start_seg_idx]  # 🚨 abierta, no cerrada
        chains[start_chain_idx] = new_chain
        if start_chain_idx != 0:
            chains.insert(0, chains.pop(start_chain_idx))

    gcode_lines = [
        "G21 ; mm",
        "G90 ; coordenadas absolutas",
        f"G1 F{cut_feed}",
        "M3 ; 🔥 ENCENDER HILO"
    ]
    preview_segments = []
    total_time = 0.0
    current_x, current_y = None, None

    def move_to(x, y, feed):
        nonlocal current_x, current_y, total_time
        gcode_lines.append(f"G1 X{x:.3f} Y{y:.3f} F{feed}")
        if current_x is not None:
            d = dist((current_x,current_y), (x,y))
            preview_segments.append(((current_x,current_y),(x,y)))
            total_time += d / (feed/60.0)
        current_x, current_y = x, y

    for chain in chains:
        for seg_idx, reverse in chain:
            s = paths[seg_idx]
            if s["type"] == "line":
                p0 = s["end"] if reverse else s["start"]
                p1 = s["start"] if reverse else s["end"]
                if current_x is None or dist((current_x,current_y), p0) > 1e-6:
                    move_to(p0[0], p0[1], cut_feed)
                move_to(p1[0], p1[1], cut_feed)
                L = dist(p0,p1)
                pause_t = (pause_ms/1000.0)*L
                gcode_lines.append(f"G4 P{pause_t:.3f} ; pausa")
                total_time += pause_t
            else:
                cx, cy = s["center"]
                r = s["radius"]
                a0 = s["end_angle"] if reverse else s["start_angle"]
                a1 = s["start_angle"] if reverse else s["end_angle"]
                ccw_eff = (not reverse) == s["ccw"]
                start = (cx + r*math.cos(rad(a0)), cy + r*math.sin(rad(a0)))
                end   = (cx + r*math.cos(rad(a1)), cy + r*math.sin(rad(a1)))
                if current_x is None or dist((current_x,current_y), start) > 1e-6:
                    move_to(start[0], start[1], cut_feed)
                i = cx - start[0]; j = cy - start[1]
                gcode_lines.append(f"{'G3' if ccw_eff else 'G2'} X{end[0]:.3f} Y{end[1]:.3f} I{i:.3f} J{j:.3f} F{cut_feed}")
                sweep_deg = signed_sweep_deg(a0, a1, ccw_eff)
                L = abs(rad(sweep_deg))*r
                total_time += L / (cut_feed/60.0)
                steps = max(8, int(ARC_SEGMENTS_DEFAULT*abs(sweep_deg)/180.0))
                for k in range(steps):
                    t1 = rad(a0 + sweep_deg*(k/steps))
                    t2 = rad(a0 + sweep_deg*((k+1)/steps))
                    x1,y1 = cx + r*math.cos(t1), cy + r*math.sin(t1)
                    x2,y2 = cx + r*math.cos(t2), cy + r*math.sin(t2)
                    preview_segments.append(((x1,y1),(x2,y2)))
                current_x, current_y = end
                pause_t = (pause_ms/1000.0)*L
                gcode_lines.append(f"G4 P{pause_t:.3f} ; pausa")
                total_time += pause_t

    return gcode_lines, preview_segments, total_time

# ======= Interfaz Streamlit =======
st.title("DXF → G-code continuo (máquina de hilo)")

uploaded_file = st.file_uploader("Sube tu archivo DXF", type=["dxf"])

if uploaded_file:
    with open("temp.dxf","wb") as f:
        f.write(uploaded_file.getbuffer())
    load_dxf("temp.dxf")

    base_name = os.path.splitext(uploaded_file.name)[0]+".gcode"
    if "nombre_archivo" not in st.session_state or st.session_state.get("ultimo_dxf")!=uploaded_file.name:
        st.session_state["nombre_archivo"] = base_name
        st.session_state["ultimo_dxf"] = uploaded_file.name
    st.text_input("Nombre del archivo:", key="nombre_archivo")

    st.write("### Parámetros")
    col1, col2, col3 = st.columns(3)
    with col1:
        cut_feed = st.number_input("Velocidad (mm/min)", 10, 5000, CUT_FEED_DEFAULT, 1)
    with col2:
        pause_ms = st.number_input("Pausa por mm (ms)", 0, 5000, PAUSE_DEFAULT_MS, 1)
    with col3:
        join_tol = st.number_input("Tolerancia unión (mm)", 0.001, 1.0, JOIN_TOL_DEFAULT, 0.001, format="%.3f")

    if st.button("Generar y Previsualizar G-code"):
        gcode_lines, preview_segments, total_time = generar_gcode(cut_feed, pause_ms, join_tol)

        st.subheader("🖼 Vista previa (trayectorias continuas)")
        fig = go.Figure()
        for (x1, y1), (x2, y2) in preview_segments:
            fig.add_trace(go.Scatter(x=[x1,x2], y=[y1,y2], mode='lines',
                                     line=dict(color='blue', width=0.8)))
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        horas = int(total_time//3600)
        minutos = int((total_time%3600)//60)
        segundos = round(total_time%60)
        st.subheader("⏱ Tiempo estimado")
        st.write(f"**{horas:02d}:{minutos:02d}:{segundos:02d}** (incluyendo pausas)")

        st.subheader("📄 G-code")
        st.text_area("G-code", "\n".join(gcode_lines), height=300)

        output = io.StringIO("\n".join(gcode_lines))
        st.download_button("💾 Descargar G-code", data=output.getvalue(),
                           file_name=st.session_state["nombre_archivo"], mime="text/plain")
