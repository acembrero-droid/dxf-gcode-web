import tkinter as tk
from tkinter import filedialog, messagebox
import ezdxf
import math
from copy import deepcopy

# === Parámetros por defecto ===
CUT_FEED_DEFAULT = 800  # mm/min
PAUSE_DEFAULT = 0.5     # segundos por mm de longitud
START_DELAY_DEFAULT = 1 # segundos de espera tras encender el hilo

canvas_width = 600
canvas_height = 600
scale_factor = 1
paths = []
ordered_paths = []

# ==== Funciones de DXF ====
def load_dxf(file_path):
    global paths
    paths = []
    doc = ezdxf.readfile(file_path)
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

# ==== Vista previa de trayectorias ====
def draw_paths():
    canvas.delete("all")
    if not paths:
        return
    xs, ys = [], []
    for e in paths:
        if e["type"]=="line":
            xs+=[e["start"][0], e["end"][0]]
            ys+=[e["start"][1], e["end"][1]]
        elif e["type"]=="polyline":
            for x,y in e["points"]:
                xs.append(x)
                ys.append(y)
        elif e["type"]=="arc":
            cx,cy = e["center"]
            r = e["radius"]
            xs+=[cx-r, cx+r]
            ys+=[cy-r, cy+r]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    global scale_factor
    scale_factor = min(canvas_width/(max_x-min_x+1), canvas_height/(max_y-min_y+1))*0.9
    offset_x = (canvas_width - (max_x-min_x)*scale_factor)/2
    offset_y = (canvas_height - (max_y-min_y)*scale_factor)/2

    order_paths()
    for e in ordered_paths:
        rev = e.get('reverse',False)
        if e["type"]=="line":
            sx,sy = e["start"]
            ex,ey = e["end"]
            if rev:
                sx,sy,ex,ey=ex,ey,sx,sy
            canvas.create_line((sx-min_x)*scale_factor+offset_x,
                               canvas_height - ((sy-min_y)*scale_factor+offset_y),
                               (ex-min_x)*scale_factor+offset_x,
                               canvas_height - ((ey-min_y)*scale_factor+offset_y),
                               fill="red",width=2)
        elif e["type"]=="polyline":
            points=e["points"][::-1] if rev else e["points"]
            pts=[]
            for x,y in points:
                px=(x-min_x)*scale_factor+offset_x
                py=canvas_height-((y-min_y)*scale_factor+offset_y)
                pts.append((px,py))
            for i in range(len(pts)-1):
                canvas.create_line(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],fill="red",width=2)
        elif e["type"]=="arc":
            cx,cy=e["center"]
            r=e["radius"]
            start_angle=e["start_angle"]
            end_angle=e["end_angle"]
            if rev:
                start_angle,end_angle=end_angle,start_angle
            segments=20
            pts=[]
            for i in range(segments+1):
                angle=math.radians(start_angle+(end_angle-start_angle)*i/segments)
                x=cx+r*math.cos(angle)
                y=cy+r*math.sin(angle)
                px=(x-min_x)*scale_factor+offset_x
                py=canvas_height - ((y-min_y)*scale_factor+offset_y)
                pts.append((px,py))
            for i in range(len(pts)-1):
                canvas.create_line(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],fill="red",width=2)

# ==== Generar G-code con vista previa y tiempo ====
def generar_gcode_interfaz(file_path, cut_feed, pause_factor, start_delay):
    if not file_path:
        messagebox.showwarning("Error","Seleccione un archivo DXF")
        return
    try:
        cut_feed = float(cut_feed)
        pause_factor = float(pause_factor)
        start_delay = float(start_delay)
    except:
        messagebox.showwarning("Error","Ingrese valores numéricos válidos")
        return

    output_file = filedialog.asksaveasfilename(defaultextension=".gcode",
                                               filetypes=[("G-code files","*.gcode"),("Text files","*.txt")])
    if not output_file:
        return

    global paths
    original_paths = deepcopy(paths)
    order_paths()
    gcode_lines=["G21 ; mm","G90 ; coordenadas absolutas",f"G1 F{cut_feed}",
                 "M3 ; 🔥 ENCENDER HILO"]
    if start_delay>0:
        gcode_lines.append(f"G4 P{start_delay:.3f} ; Espera de inicio")

    current_x,current_y=None,None
    total_time=0.0
    def move_to(x,y):
        nonlocal current_x,current_y,total_time
        gcode_lines.append(f"G1 X{x:.3f} Y{y:.3f} F{cut_feed}")
        if current_x is not None:
            dist = distance((current_x,current_y),(x,y))
            total_time += (dist / (cut_feed/60))
        current_x,current_y=x,y

    for entity in ordered_paths:
        rev=entity.get('reverse',False)
        entity_len = length_entity(entity)
        if entity["type"]=="line":
            sx,sy=entity["start"]
            ex,ey=entity["end"]
            if rev: sx,sy,ex,ey=ex,ey,sx,sy
            move_to(sx,sy)
            move_to(ex,ey)
        elif entity["type"]=="polyline":
            points=entity["points"][::-1] if rev else entity["points"]
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
            move_to(start_x,start_y)
            ccw=end_angle>start_angle
            i=cx-start_x
            j=cy-start_y
            gcode_lines.append(f"{'G3' if ccw else 'G2'} X{end_x:.3f} Y{end_y:.3f} I{i:.3f} J{j:.3f} F{cut_feed}")
            arc_angle = abs(end_angle-start_angle)
            arc_length = math.radians(arc_angle)*r
            total_time += arc_length / (cut_feed/60)
            current_x,current_y=end_x,end_y

        # Añadimos pausa proporcional a la longitud
        pause_time = pause_factor * entity_len
        gcode_lines.append(f"G4 P{pause_time:.3f} ; pausa")
        total_time += pause_time

    gcode_lines.append("M5 ; 🔌 APAGAR HILO")

    with open(output_file,"w") as f:
        f.write("\n".join(gcode_lines))

    messagebox.showinfo("Éxito",f"G-code generado en:\n{output_file}\n\n⏱ Tiempo estimado: {total_time:.1f} s")
    paths = deepcopy(original_paths)

# ==== Interfaz gráfica ====
def seleccionar_archivo():
    file_path = filedialog.askopenfilename(filetypes=[("DXF files","*.dxf")])
    if file_path:
        entry_file.delete(0, tk.END)
        entry_file.insert(0, file_path)
        load_dxf(file_path)
        draw_paths()

def ejecutar():
    file_path = entry_file.get()
    cut_feed = entry_feed.get()
    pause_factor = entry_pause.get()
    start_delay = entry_delay.get()
    generar_gcode_interfaz(file_path, cut_feed, pause_factor, start_delay)

root = tk.Tk()
root.title("DXF a G-code - Vista Previa y Tiempo")

tk.Label(root,text="Archivo DXF:").grid(row=0,column=0,padx=5,pady=5)
entry_file = tk.Entry(root,width=50)
entry_file.grid(row=0,column=1,padx=5,pady=5)
tk.Button(root,text="Seleccionar",command=seleccionar_archivo).grid(row=0,column=2,padx=5,pady=5)

tk.Label(root,text="Velocidad de corte (mm/min):").grid(row=1,column=0,padx=5,pady=5)
entry_feed = tk.Entry(root)
entry_feed.insert(0,str(CUT_FEED_DEFAULT))
entry_feed.grid(row=1,column=1,padx=5,pady=5)

tk.Label(root,text="Pausa (s por mm):").grid(row=2,column=0,padx=5,pady=5)
entry_pause = tk.Entry(root)
entry_pause.insert(0,str(PAUSE_DEFAULT))
entry_pause.grid(row=2,column=1,padx=5,pady=5)

tk.Label(root,text="Retardo tras encender hilo (s):").grid(row=3,column=0,padx=5,pady=5)
entry_delay = tk.Entry(root)
entry_delay.insert(0,str(START_DELAY_DEFAULT))
entry_delay.grid(row=3,column=1,padx=5,pady=5)

tk.Button(root,text="Generar G-code",command=ejecutar).grid(row=4,column=0,columnspan=3,padx=5,pady=10)

canvas = tk.Canvas(root,width=canvas_width,height=canvas_height,bg="white")
canvas.grid(row=5,column=0,columnspan=3,padx=5,pady=5)

root.mainloop()
