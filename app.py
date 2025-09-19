import streamlit as st
import ezdxf
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile

CUT_FEED_DEFAULT = 800
PAUSE_DEFAULT = 0.5  # segundos por mm de longitud

paths = []
ordered_paths = []

# ==== Funciones de DXF ====
def load_dxf(uploaded_file):
    global paths
    paths = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    doc = ezdxf.readfile(tmp_file_path)
    msp = doc.modelspace()
    for entity in msp:
        if entity.dxftype() == "LINE":
            paths.append({"type":"line","start":(entity.dxf.start[0], entity.dxf.start[1]),
                          "end":(entity.dxf.end[0], entity.dxf.end[1])})
        elif entity.dxftype() == "LWPOLYLINE":
            points = [(p[0],p[1]) for p in entity.get_points()]
            paths.append({"type":"polyline","points":points})
        elif entity.dxftype() == "ARC":
            paths.append({"type":"arc","center":(entity.dxf.center[0], entity.dxf.center[1]),
                          "radius":entity.dxf.radius,"start_angle":entity.dxf.start_angle,
                          "end_angle":entity.dxf.end_angle})

def get_entity_points(entity):
    if entity["type"]=="line":
        return entity["start"],entity["end"]
    elif entity["type"]=="polyline":
        return entity["points"][0],entity["points"][-1]
    elif entity["type"]=="arc":
        cx,cy=entity["center"]
        r=entity["radius"]
        start=(cx+r*math.cos(math.radians(entity["start_angle"])),
               cy+r*math.sin(math.radians(entity["start_angle"])))
        end=(cx+r*math.cos(math.radians(entity["end_angle"])),
             cy+r*math.sin(math.radians(entity["end_angle"])))
        return start,end

def distance(p1,p2):
    return math.hypot(p2[0]-p1[0],p2[1]-p1[1])

def length_entity(entity):
    if entity["type"]=="line":
        return distance(entity["start"], entity["end"])
    elif entity["type"]=="polyline":
        l = 0
        pts = entity["points"]
        for i in range(len(pts)-1):
            l += distance(pts[i], pts[i+1])
        return l
    elif entity["type"]=="arc":
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
        for i,entity in enumerate(remaining):
            start,end=get_entity_points(entity)
            dist_start = distance(current_pos,start)
            dist_end = distance(current_pos,end)
            if dist_start<best_distance:
                best_distance=dist_start
                best_index=i
                reverse=False
            if dist_end<best_distance:
                best_distance=dist_end
                best_index=i
                reverse=True
        entity=remaining.pop(best_index)
        entity['reverse']=reverse
        ordered_paths.append(entity)
        start,end=get_entity_points(entity)
        current_pos=end if not reverse else start

# ==== Generar G-code ====
def generar_gcode(cut_feed, pause_factor):
    order_paths()
    gcode_lines=["G21 ; mm","G90 ; coordenadas absolutas",f"G1 F{cut_feed}"]

    current_x,current_y=None,None
    def move_to(x,y):
        nonlocal current_x,current_y
        gcode_lines.append(f"G1 X{x:.3f} Y{y:.3f} F{cut_feed}")
        current_x,current_y=x,y
    def is_close(x1,y1,x2,y2,tol=0.001):
        return abs(x1-x2)<tol and abs(y1-y2)<tol

    for entity in ordered_paths:
        rev=entity.get('reverse',False)
        entity_len = length_entity(entity)
        if entity["type"]=="line":
            sx,sy=entity["start"]
            ex,ey=entity["end"]
            if rev: sx,sy,ex,ey=ex,ey,sx,sy
            if current_x is None or not is_close(current_x,current_y,sx,sy):
                move_to(sx,sy)
            move_to(ex,ey)
        elif entity["type"]=="polyline":
            points=entity["points"][::-1] if rev else entity["points"]
            if current_x is None or not is_close(current_x,current_y,points[0][0],points[0][1]):
                move_to(points[0][0],points[0][1])
            for x,y in points[1:]:
                move_to(x,y)
        elif entity["type"]=="arc":
            cx,cy=entity["center"]
            r=entity["radius"]
            start_angle=entity["start_angle"]
            end_angle=entity["end_angle"]
            if rev: start_angle,end_angle=end_angle,start_angle
            start_x=cx+r*math.cos(math.radians(start_angle))
            start_y=cy+r*math.sin(math.radians(start_angle))
            end_x=cx+r*math.cos(math.radians(end_angle))
            end_y=cy+r*math.sin(math.radians(end_angle))
            if current_x is None or not is_close(current_x,current_y,start_x,start_y):
                move_to(start_x,start_y)
            ccw=end_angle>start_angle
            i=cx-start_x
            j=cy-start_y
            gcode_lines.append(f"{'G3' if ccw else 'G2'} X{end_x:.3f} Y{end_y:.3f} I{i:.3f} J{j:.3f} F{cut_feed}")
            current_x,current_y=end_x,end_y

        pause_time = pause_factor * entity_len
        gcode_lines.append(f"G4 P{pause_time:.3f} ; pausa")

    return "\n".join(gcode_lines)

# ==== Vista previa ====
def draw_paths():
    if not paths:
        return None
    order_paths()
    fig, ax = plt.subplots()
    for e in ordered_paths:
        rev = e.get('reverse',False)
        if e["type"]=="line":
            sx,sy = e["start"]
            ex,ey = e["end"]
            if rev: sx,sy,ex,ey=ex,ey,sx,sy
            ax.plot([sx,ex],[sy,ey],'r-')
        elif e["type"]=="polyline":
            pts = e["points"][::-1] if rev else e["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys,'r-')
        elif e["type"]=="arc":
            cx,cy=e["center"]
            r=e["radius"]
            start_angle=e["start_angle"]
            end_angle=e["end_angle"]
            if rev: start_angle,end_angle=end_angle,start_angle
            segments=20
            xs,ys=[],[]
            for i in range(segments+1):
                angle=math.radians(start_angle+(end_angle-start_angle)*i/segments)
                xs.append(cx+r*math.cos(angle))
                ys.append(cy+r*math.sin(angle))
            ax.plot(xs, ys,'r-')
    ax.set_aspect('equal')
    return fig

# ==== Interfaz Web ====
st.title("DXF a G-code - Vista Web")

uploaded_file = st.file_uploader("Sube tu archivo DXF", type=["dxf"])
cut_feed = st.number_input("Velocidad de corte (mm/min):", value=CUT_FEED_DEFAULT, min_value=1)
pause_factor = st.slider("Pausa por longitud (s/mm):", 0.0, 5.0, value=PAUSE_DEFAULT, step=0.05)
filename = st.text_input("Nombre del archivo G-code:", value="gcode.txt")

if uploaded_file:
    load_dxf(uploaded_file)
    st.success(f"Se cargaron {len(paths)} entidades.")

    fig = draw_paths()
    if fig:
        st.pyplot(fig)

    if st.button("Generar y descargar G-code"):
        gcode = generar_gcode(cut_feed, pause_factor)
        output = BytesIO()
        output.write(gcode.encode())
        output.seek(0)
        st.download_button("⬇️ Descargar G-code", data=output, file_name=filename)
