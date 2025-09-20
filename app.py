import streamlit as st
import ezdxf
import math
import io
import plotly.graph_objects as go
import os

CUT_FEED_DEFAULT = 800      # mm/min
PAUSE_DEFAULT_MS = 500      # milisegundos por mm
ARC_SEGMENTS_DEFAULT = 40   # resolución fija para arcos
paths = []
ordered_paths = []

# ======= Funciones para procesar DXF =======
def load_dxf(file):
    global paths
    paths = []
    doc = ezdxf.readfile(file)
    msp = doc.modelspace()
    for entity in msp:
        if entity.dxftype() == "LINE":
            paths.append({"type": "line", "start": (entity.dxf.start[0], entity.dxf.start[1]),
                          "end": (entity.dxf.end[0], entity.dxf.end[1])})
        elif entity.dxftype() == "LWPOLYLINE":
            points = [(p[0], p[1]) for p in entity.get_points()]
            paths.append({"type": "polyline", "points": points})
        elif entity.dxftype() == "ARC":
            paths.append({"type": "arc", "center": (entity.dxf.center[0], entity.dxf.center[1]),
                          "radius": entity.dxf.radius, "start_angle": entity.dxf.start_angle,
                          "end_angle": entity.dxf.end_angle})

def get_entity_points(entity):
    if entity["type"] == "line":
        return entity["start"], entity["end"]
    elif entity["type"] == "polyline":
        return entity["points"][0], entity["points"][-1]
    elif entity["type"] == "arc":
        cx, cy = entity["center"]
        r = entity["radius"]
        start = (cx + r * math.cos(math.radians(entity["start_angle"])),
                 cy + r * math.sin(math.radians(entity["start_angle"])))
        end = (cx + r * math.cos(math.radians(entity["end_angle"])),
               cy + r * math.sin(math.radians(entity["end_angle"])))
        return start, end

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def length_entity(entity):
    if entity["type"] == "line":
        return distance(entity["start"], entity["end"])
    elif entity["type"] == "polyline":
        l = 0
        pts = entity["points"]
        for i in range(len(pts) - 1):
            l += distance(pts[i], pts[i + 1])
        return l
    elif entity["type"] == "arc":
        angle = math.radians(abs(entity["end_angle"] - entity["start_angle"]))
        return angle * entity["radius"]

def order_paths():
    global ordered_paths
    remaining = paths[:]
    remaining.sort(key=lambda e: min(get_entity_points(e)[0][0], get_entity_points(e)[1][0]))
    current_pos = get_entity_points(remaining[0])[0]
    ordered_paths = []
    while remaining:
        best_index = None
        best_distance = float("inf")
        reverse = False
        for i, entity in enumerate(remaining):
            start, end = get_entity_points(entity)
            dist_start = distance(current_pos, start)
            dist_end = distance(current_pos, end)
            if dist_start < best_distance:
                best_distance = dist_start
                best_index = i
                reverse = False
            if dist_end < best_distance:
                best_distance = dist_end
                best_index = i
                reverse = True
        entity = remaining.pop(best_index)
        entity['reverse'] = reverse
        ordered_paths.append(entity)
        start, end = get_entity_points(entity)
        current_pos = end if not reverse else start

# ======= Generar G-code, preview y tiempo =======
def generar_gcode(cut_feed, pause_factor_ms):
    order_paths()
    gcode_lines = ["G21 ; mm", "G90 ; coordenadas absolutas", f"G1 F{cut_feed}"]
    preview_segments = []
    total_time = 0.0
    current_x, current_y = None, None

    def move_to(x, y, feed):
        nonlocal current_x, current_y, total_time
        gcode_lines.append(f"G1 X{x:.3f} Y{y:.3f} F{feed}")
        if current_x is not None:
            dist = distance((current_x, current_y), (x, y))
            preview_segments.append(((current_x, current_y), (x, y)))
            total_time += (dist / (feed / 60))  # tiempo de corte
        current_x, current_y = x, y

    def is_close(x1, y1, x2, y2, tol=0.001):
        return abs(x1 - x2) < tol and abs(y1 - y2) < tol

    for entity in ordered_paths:
        rev = entity.get('reverse', False)
        entity_len = length_entity(entity)

        if entity["type"] == "line":
            sx, sy = entity["start"]
            ex, ey = entity["end"]
            if rev: sx, sy, ex, ey = ex, ey, sx, sy
            if current_x is None or not is_close(current_x, current_y, sx, sy):
                move_to(sx, sy, cut_feed)
            move_to(ex, ey, cut_feed)

        elif entity["type"] == "polyline":
            points = entity["points"][::-1] if rev else entity["points"]
            if current_x is None or not is_close(current_x, current_y, points[0][0], points[0][1]):
                move_to(points[0][0], points[0][1], cut_feed)
            for x, y in points[1:]:
                move_to(x, y, cut_feed)

        elif entity["type"] == "arc":
            cx, cy = entity["center"]
            r = entity["radius"]
            start_angle = entity["start_angle"]
            end_angle = entity["end_angle"]
            if rev: start_angle, end_angle = end_angle, start_angle
            start_x = cx + r * math.cos(math.radians(start_angle))
            start_y = cy + r * math.sin(math.radians(start_angle))
            end_x = cx + r * math.cos(math.radians(end_angle))
            end_y = cy + r * math.sin(math.radians(end_angle))
            if current_x is None or not is_close(current_x, current_y, start_x, start_y):
                move_to(start_x, start_y, cut_feed)
            ccw = end_angle > start_angle
            i = cx - start_x
            j = cy - start_y
            gcode_lines.append(f"{'G3' if ccw else 'G2'} X{end_x:.3f} Y{end_y:.3f} I{i:.3f} J{j:.3f} F{cut_feed}")
            arc_length = length_entity(entity)
            total_time += (arc_length / (cut_feed / 60))
            for k in range(ARC_SEGMENTS_DEFAULT):
                angle1 = math.radians(start_angle + (end_angle - start_angle) * k / ARC_SEGMENTS_DEFAULT)
                angle2 = math.radians(start_angle + (end_angle - start_angle) * (k + 1) / ARC_SEGMENTS_DEFAULT)
                x1, y1 = cx + r * math.cos(angle1), cy + r * math.sin(angle1)
                x2, y2 = cx + r * math.cos(angle2), cy + r * math.sin(angle2)
                preview_segments.append(((x1, y1), (x2, y2)))
            current_x, current_y = end_x, end_y

        pause_time = (pause_factor_ms / 1000.0) * entity_len
        gcode_lines.append(f"G4 P{pause_time:.3f} ; pausa")
        total_time += pause_time

    return gcode_lines, preview_segments, total_time

# ======= Interfaz Streamlit =======
st.title("DXF → G-code con Vista Previa y Tiempo Estimado")

uploaded_file = st.file_uploader("Sube tu archivo DXF", type=["dxf"])

st.write("### Parámetros de corte")
col1, col2 = st.columns(2)
with col1:
    cut_feed_slider = st.slider("Velocidad de corte (mm/min)", min_value=10, max_value=1000,
                                value=CUT_FEED_DEFAULT, step=10)
with col2:
    cut_feed_input = st.number_input("Valor exacto", min_value=10, max_value=1000,
                                     value=CUT_FEED_DEFAULT, step=1)

cut_feed = cut_feed_input if cut_feed_input != CUT_FEED_DEFAULT else cut_feed_slider
pause_factor_ms = st.slider("Pausa por mm (ms)", min_value=0, max_value=2000,
                            value=PAUSE_DEFAULT_MS, step=10)

if uploaded_file:
    with open("temp.dxf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    load_dxf("temp.dxf")

    # Inicializar nombre del archivo basado en el DXF solo si es un DXF nuevo
    base_name = os.path.splitext(uploaded_file.name)[0] + ".gcode"
    if "nombre_archivo" not in st.session_state or st.session_state.get("ultimo_dxf") != uploaded_file.name:
        st.session_state["nombre_archivo"] = base_name
        st.session_state["ultimo_dxf"] = uploaded_file.name

    if st.button("Generar y Previsualizar G-code"):
        gcode_lines, preview_segments, total_time = generar_gcode(cut_feed, pause_factor_ms)

        # Vista previa interactiva
        st.subheader("🖼 Vista Previa de Trayectoria")
        fig = go.Figure()
        for (x1, y1), (x2, y2) in preview_segments:
            fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2],
                                     mode='lines',
                                     line=dict(color='blue', width=0.2)))
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=True),
            yaxis=dict(showgrid=True),
            title="Trayectoria Generada",
            showlegend=False,
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tiempo total
        horas = int(total_time // 3600)
        minutos = int((total_time % 3600) // 60)
        segundos = round(total_time % 60)
        st.subheader("⏱ Tiempo estimado de ejecución")
        st.write(f"**{horas:02d}:{minutos:02d}:{segundos:02d}** (incluyendo pausas)")

        # Nombre editable persistente
        st.text_input("Nombre del archivo:", key="nombre_archivo")

        # Botón de descarga
        output = io.StringIO("\n".join(gcode_lines))
        st.download_button(
            "💾 Descargar G-code",
            data=output.getvalue(),
            file_name=st.session_state["nombre_archivo"],
            mime="text/plain"
        )
