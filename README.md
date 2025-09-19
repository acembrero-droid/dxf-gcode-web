# DXF a G-code Web

Esta aplicación permite subir archivos DXF y generar G-code continuo para máquinas de hilo de 2 ejes, respetando la geometría exacta de líneas, polilíneas y arcos.

**Características:**
- Ordena las entidades DXF automáticamente.
- Genera G-code continuo sin saltos.
- Arcos tratados correctamente con G2/G3.
- Pausas proporcionales a la longitud.
- Vista previa del recorrido.

**Librerías requeridas:** `streamlit`, `ezdxf`, `matplotlib`

**Uso:**
1. Subir un archivo DXF.
2. Configurar velocidad de corte y pausa.
3. Descargar el G-code generado.
