import streamlit as st
import ezdxf
import math
import io
import plotly.graph_objects as go
import os

# ======= Configuración =======
CUT_FEED_DEFAULT = 100
PAUSE_DEFAULT_MS = 5
ARC_SEGMENTS_DEFAULT = 24
JOIN_TOL_DEFAULT = 0.01

paths = []

# -------- Utilidades geométricas --------
def dist(a,b): return math.hypot(b[0]-a[0], b[1]-a[1])
def rad(a):    return math.radians(a)
def deg(a):    return math.degrees(a)
def ang_of(p, c): return (math.degrees(math.atan2(p[1]-c[1], p[0]-c[0]))+360)%360
def signed_sweep_deg(a0, a1, ccw=True):
    a0 %= 360.0; a1 %= 360.0
    if ccw:
        return (a1 - a0) % 360.0
    else:
        return -((a0 - a1) % 360.0)
def point_key(p, tol): return (round(p[0]/tol), round(p[1]/tol))

# ======= Crear segmentos =======
def add_line(p0, p1):
    paths.append({"type":"line","start":p0,"end":p1})

def add_arc_from_center(p0, p1, center):
    cx,cy = center; r = dist(p0,center)
    a0 = ang_of(p0,center); a1 = ang_of(p1,center)
    v0 = (p0[0]-cx, p0[1]-cy); v1 = (p1[0]-cx, p1[1]-cy)
    ccw = (v0[0]*v1[1]-v0[1]*v1[0]) > 0
    paths.append({"type":"arc","center":(cx,cy),"radius":r,
                  "start":p0,"end":p1,"start_angle":a0,"end_angle":a1,"ccw":ccw})

def add_arc_from_bulge(p0,p1,b):
    if abs(b)<1e-12: add_line(p0,p1); return
    x0,y0=p0; x1,y1=p1; chord=(x1-x0,y1-y0); c=math.hypot(*chord)
    if c<1e-12: return
    theta=4*math.atan(b); R=c/(2*abs(math.sin(theta/2)))
    mid=((x0+x1)/2,(y0+y1)/2)
    n=(-chord[1]/c, chord[0]/c)
    d=R*math.cos(theta/2); d=d if b>0 else -d
    center=(mid[0]+n[0]*d, mid[1]+n[1]*d)
    add_arc_from_center(p0,p1,center)

# ======= Cargar DXF =======
def load_dxf(file):
    global paths; paths=[]
    doc=ezdxf.readfile(file); msp=doc.modelspace()
    for e in msp:
        dxft=e.dxftype()
        if dxft=="LINE":
            add_line((e.dxf.start[0],e.dxf.start[1]),(e.dxf.end[0],e.dxf.end[1]))
        elif dxft=="ARC":
            c=(e.dxf.center[0],e.dxf.center[1]); r=e.dxf.radius
            a0=e.dxf.start_angle%360; a1=e.dxf.end_angle%360
            p0=(c[0]+r*math.cos(rad(a0)), c[1]+r*math.sin(rad(a0)))
            p1=(c[0]+r*math.cos(rad(a1)), c[1]+r*math.sin(rad(a1)))
            add_arc_from_center(p0,p1,c)
        elif dxft=="LWPOLYLINE":
            pts=list(e.get_points("xyb"))
            for i in range(len(pts)-1):
                p0=(pts[i][0],pts[i][1]); p1=(pts[i+1][0],pts[i+1][1]); b=float(pts[i][2])
                add_arc_from_bulge(p0,p1,b)
            if e.closed:
                p0=(pts[-1][0],pts[-1][1]); p1=(pts[0][0],pts[0][1]); b=float(pts[-1][2])
                add_arc_from_bulge(p0,p1,b)

# ======= Construir cadenas =======
def build_nodemap(join_tol):
    nodemap={}
    def reg(idx,pt,is_start):
        k=point_key(pt,join_tol)
        if k not in nodemap: nodemap[k]={"pt":pt,"inc":[]}
        nodemap[k]["inc"].append((idx,is_start))
    for i,s in enumerate(paths):
        reg(i,s["start"],True); reg(i,s["end"],False)
    return nodemap

def build_chains(join_tol):
    nodemap=build_nodemap(join_tol)
    used=[False]*len(paths); chains=[]
    def other_endpoint(seg_idx,at_start):
        s=paths[seg_idx]; return s["end"] if at_start else s["start"]
    def find_seed():
        for k,nd in nodemap.items():
            free=[(i,st) for (i,st) in nd["inc"] if not used[i]]
            if free: return k
        return None
    while True:
        sk=find_seed(); 
        if sk is None: break
        chain=[]; curr_key=sk
        while True:
            inc=nodemap[curr_key]["inc"]
            cand=[(i,st) for (i,st) in inc if not used[i]]
            if not cand: break
            seg_idx,at_start=cand[0]; used[seg_idx]=True
            chain.append(seg_idx)
            nxt=other_endpoint(seg_idx,at_start)
            curr_key=point_key(nxt,join_tol)
        if chain: chains.append(chain)
    return chains, nodemap

# ======= Encontrar X mínima =======
def find_leftmost_point():
    best_x=float("inf"); best_pt=None
    for s in paths:
        for pt in (s["start"], s["end"]):
            if pt[0]<best_x: best_x, best_pt=pt[0], pt
    return best_pt

# ======= Orientar cadena desde nodo dado =======
def orient_chain_from_point(chain_idxs,start_pt,join_tol):
    start_key=point_key(start_pt,join_tol)
    oriented=[]
    curr_key=start_key
    for seg_idx in chain_idxs:
        s=paths[seg_idx]; p0,p1=s["start"],s["end"]
        if point_key(p0,join_tol)==curr_key:
            reverse=False; next_key=point_key(p1,join_tol)
        elif point_key(p1,join_tol)==curr_key:
            reverse=True; next_key=point_key(p0,join_tol)
        else:
            continue
        oriented.append((seg_idx,reverse)); curr_key=next_key
    return oriented

# ======= Generar G-code =======
def generar_gcode(cut_feed,pause_ms,join_tol):
    chains,nodemap=build_chains(join_tol)
    start_pt=find_leftmost_point()
    if not chains or start_pt is None: return ["G21","G90"],[],0.0
    # Encontrar cadena que contiene el nodo
    start_key=point_key(start_pt,join_tol)
    target_chain=None
    for chain in chains:
        for seg_idx in chain:
            s=paths[seg_idx]
            if point_key(s["start"],join_tol)==start_key or point_key(s["end"],join_tol)==start_key:
                target_chain=chain; break
        if target_chain: break
    if target_chain is None: target_chain=chains[0]
    oriented=orient_chain_from_point(target_chain,start_pt,join_tol)

    gcode=["G21 ; mm","G90 ; coordenadas absolutas",f"G1 F{cut_feed}","M3 ; 🔥 ENCENDER HILO"]
    preview_segments=[]; total_time=0.0; current_x,current_y=None,None
    def move_to(x,y,feed):
        nonlocal current_x,current_y,total_time
        gcode.append(f"G1 X{x:.3f} Y{y:.3f} F{feed}")
        if current_x is not None:
            d=dist((current_x,current_y),(x,y))
            preview_segments.append(((current_x,current_y),(x,y)))
            total_time+=d/(feed/60.0)
        current_x,current_y=x,y
    for seg_idx,reverse in oriented:
        s=paths[seg_idx]
        if s["type"]=="line":
            p0=s["end"] if reverse else s["start"]; p1=s["start"] if reverse else s["end"]
            if current_x is None or dist((current_x,current_y),p0)>1e-6: move_to(*p0,cut_feed)
            move_to(*p1,cut_feed)
            L=dist(p0,p1); pause_t=(pause_ms/1000.0)*L
            gcode.append(f"G4 P{pause_t:.3f} ; pausa"); total_time+=pause_t
        else:
            cx,cy=s["center"]; r=s["radius"]
            a0=s["end_angle"] if reverse else s["start_angle"]
            a1=s["start_angle"] if reverse else s["end_angle"]
            ccw_eff=s["ccw"] if not reverse else (not s["ccw"])
            start=(cx+r*math.cos(rad(a0)), cy+r*math.sin(rad(a0)))
            end=(cx+r*math.cos(rad(a1)), cy+r*math.sin(rad(a1)))
            if current_x is None or dist((current_x,current_y),start)>1e-6: move_to(*start,cut_feed)
            i=cx-start[0]; j=cy-start[1]
            gcode.append(f"{'G3' if ccw_eff else 'G2'} X{end[0]:.3f} Y{end[1]:.3f} I{i:.3f} J{j:.3f} F{cut_feed}")
            sweep=signed_sweep_deg(a0,a1,ccw_eff); L=abs(rad(sweep))*r; total_time+=L/(cut_feed/60.0)
            steps=max(8,int(ARC_SEGMENTS_DEFAULT*abs(sweep)/180.0))
            for k in range(steps):
                t1=rad(a0+sweep*(k/steps)); t2=rad(a0+sweep*((k+1)/steps))
                x1,y1=cx+r*math.cos(t1), cy+r*math.sin(t1); x2,y2=cx+r*math.cos(t2), cy+r*math.sin(t2)
                preview_segments.append(((x1,y1),(x2,y2)))
            current_x,current_y=end; pause_t=(pause_ms/1000.0)*L
            gcode.append(f"G4 P{pause_t:.3f} ; pausa"); total_time+=pause_t
    return gcode,preview_segments,total_time

# ======= Interfaz =======
st.title("DXF → G-code continuo (empieza en X mínima)")

uploaded_file=st.file_uploader("Sube tu archivo DXF",type=["dxf"])
if uploaded_file:
    with open("temp.dxf","wb") as f: f.write(uploaded_file.getbuffer())
    load_dxf("temp.dxf")
    base_name=os.path.splitext(uploaded_file.name)[0]+".gcode"
    if "nombre_archivo" not in st.session_state or st.session_state.get("ultimo_dxf")!=uploaded_file.name:
        st.session_state["nombre_archivo"]=base_name; st.session_state["ultimo_dxf"]=uploaded_file.name
    st.text_input("Nombre del archivo:",key="nombre_archivo")
    st.write("### Parámetros")
    col1,col2,col3=st.columns(3)
    with col1: cut_feed=st.number_input("Velocidad (mm/min)",10,5000,CUT_FEED_DEFAULT,1)
    with col2: pause_ms=st.number_input("Pausa por mm (ms)",0,5000,PAUSE_DEFAULT_MS,1)
    with col3: join_tol=st.number_input("Tolerancia unión (mm)",0.001,1.0,JOIN_TOL_DEFAULT,0.001,format="%.3f")
    if st.button("Generar y Previsualizar G-code"):
        gcode_lines,preview_segments,total_time=generar_gcode(cut_feed,pause_ms,join_tol)
        st.subheader("🖼 Vista previa")
        fig=go.Figure()
        for (x1,y1),(x2,y2) in preview_segments:
            fig.add_trace(go.Scatter(x=[x1,x2],y=[y1,y2],mode='lines',line=dict(color='blue',width=0.8)))
        fig.update_layout(xaxis=dict(scaleanchor="y",scaleratio=1),height=700,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
        horas=int(total_time//3600); minutos=int((total_time%3600)//60); segundos=round(total_time%60)
        st.subheader("⏱ Tiempo estimado"); st.write(f"**{horas:02d}:{minutos:02d}:{segundos:02d}**")
        st.subheader("📄 G-code"); st.text_area("G-code","\n".join(gcode_lines),height=300)
        output=io.StringIO("\n".join(gcode_lines))
        st.download_button("💾 Descargar G-code",data=output.getvalue(),
                           file_name=st.session_state["nombre_archivo"],mime="text/plain")
