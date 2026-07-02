import os
import cv2
import math
from ultralytics import YOLO

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIANZA_MINIMA = 0.50

# Fallback si no hay personas suficientes para estimar
# una distancia adaptativa basada en el ancho de las cajas.
DISTANCIA_AGRUPACION_FALLBACK = 100

# Factor multiplicador del ancho promedio de bounding box.
# Si el ancho promedio de una persona es ~80px en el video,
# se consideran "del mismo grupo" si sus centros están a
# menos de 80 * 1.5 = 120px. Esto se auto-ajusta al zoom
# y resolución del video.
FACTOR_DISTANCIA_AGRUPACION = 1.5

# Umbrales de aglomeración (cuántas personas en el grupo mayor)
# clasificar_aglomeracion() de abajo: 2-3 → BAJO, 4-5 → MEDIO, 6+ → ALTO
# (mismo criterio que benchmark/vision_core/modelos/aglomeracion.py)
UMBRAL_MEDIO = 4   # a partir de 4 personas → MEDIO
UMBRAL_ALTO = 6    # 6 o más → ALTO

# ============================================================
# CARGA DEL MODELO
# ============================================================

def cargar_modelo():
    return YOLO("yolov8s.pt")


modelo = cargar_modelo()


# ============================================================
# DETECCIÓN
# ============================================================

def esta_en_zona_ignorada(x1, y1, x2, y2, zonas):
    centro_x = int((x1 + x2) / 2)
    centro_y = int((y1 + y2) / 2)

    for zx1, zy1, zx2, zy2 in zonas:
        if zx1 <= centro_x <= zx2 and zy1 <= centro_y <= zy2:
            return True
    return False


def detectar_personas(frame, zonas):
    resultados = modelo(frame, verbose=False)
    personas = []

    for resultado in resultados:
        for caja in resultado.boxes:
            clase = int(caja.cls[0])
            confianza = float(caja.conf[0])

            if clase == 0 and confianza >= CONFIANZA_MINIMA:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])

                if esta_en_zona_ignorada(x1, y1, x2, y2, zonas):
                    continue

                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)
                ancho = x2 - x1

                personas.append({
                    "bbox": (x1, y1, x2, y2),
                    "centro": (centro_x, centro_y),
                    "confianza": confianza,
                    "ancho": ancho,
                })

    return personas


# ============================================================
# AGRUPACIÓN
# ============================================================

def calcular_distancia(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calcular_distancia_adaptativa(personas):
    """
    Calcula el umbral de distancia para agrupar personas,
    basado en el ancho promedio de las cajas detectadas.

    Esto hace que el agrupamiento se adapte automáticamente al
    zoom y resolución del video, en lugar de depender de un
    número fijo de píxeles.
    """
    if len(personas) < 2:
        return DISTANCIA_AGRUPACION_FALLBACK

    anchos = [p["ancho"] for p in personas if p["ancho"] > 0]
    if not anchos:
        return DISTANCIA_AGRUPACION_FALLBACK

    ancho_promedio = sum(anchos) / len(anchos)
    return ancho_promedio * FACTOR_DISTANCIA_AGRUPACION


def agrupar_personas(personas, distancia_umbral):
    grupos = []
    visitados = set()

    for i in range(len(personas)):
        if i in visitados:
            continue

        grupo_actual = []
        cola = [i]
        visitados.add(i)

        while cola:
            indice_actual = cola.pop(0)
            persona_actual = personas[indice_actual]
            grupo_actual.append(indice_actual)

            for j in range(len(personas)):
                if j not in visitados:
                    distancia = calcular_distancia(
                        persona_actual["centro"],
                        personas[j]["centro"]
                    )
                    if distancia <= distancia_umbral:
                        visitados.add(j)
                        cola.append(j)

        grupos.append(grupo_actual)

    return grupos


def obtener_grupo_mas_grande(grupos):
    if not grupos:
        return 0
    return max(len(grupo) for grupo in grupos)


# ============================================================
# CLASIFICACIÓN
# ============================================================

def clasificar_aglomeracion(grupo_mas_grande):
    if grupo_mas_grande <= 1:
        return "BAJO", (0, 255, 0)
    elif grupo_mas_grande < UMBRAL_MEDIO:
        return "BAJO", (0, 255, 0)
    elif grupo_mas_grande < UMBRAL_ALTO:
        return "MEDIO", (0, 255, 255)
    else:
        return "ALTO", (0, 0, 255)


# ============================================================
# DIBUJO
# ============================================================

def mostrar_zonas(frame, zonas):
    for zx1, zy1, zx2, zy2 in zonas:
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (80, 80, 80), 1)


def dibujar_personas(frame, personas):
    for persona in personas:
        x1, y1, x2, y2 = persona["bbox"]
        centro_x, centro_y = persona["centro"]
        confianza = persona["confianza"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (centro_x, centro_y), 4, (255, 0, 0), -1)
        cv2.putText(frame, f"Persona {confianza:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def dibujar_grupos(frame, personas, grupos, distancia_umbral):
    for grupo in grupos:
        if len(grupo) <= 1:
            continue

        puntos = [personas[i]["centro"] for i in grupo]

        for i in range(len(puntos)):
            for j in range(i + 1, len(puntos)):
                distancia = calcular_distancia(puntos[i], puntos[j])
                if distancia <= distancia_umbral:
                    cv2.line(frame, puntos[i], puntos[j], (255, 255, 0), 1)


# ============================================================
# STREAM PRINCIPAL
# ============================================================

def generar_stream_video(ruta_entrada, nombre_video="video", zonas=None, preset_id=None, preset_nombre=None, user_id=None):
    from database import guardar_analisis

    zonas = zonas or []

    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        raise Exception("No se pudo abrir el video.")

    personas_maximas = 0
    grupo_mayor_maximo = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            personas = detectar_personas(frame, zonas)

            # Distancia de agrupación adaptativa según tamaño de las personas
            distancia_umbral = calcular_distancia_adaptativa(personas)

            grupos = agrupar_personas(personas, distancia_umbral)
            grupo_mas_grande = obtener_grupo_mas_grande(grupos)

            # Nivel del frame actual (solo para mostrar en pantalla)
            nivel_aglomeracion, color_aglomeracion = clasificar_aglomeracion(grupo_mas_grande)

            personas_maximas = max(personas_maximas, len(personas))
            grupo_mayor_maximo = max(grupo_mayor_maximo, grupo_mas_grande)

            dibujar_personas(frame, personas)
            dibujar_grupos(frame, personas, grupos, distancia_umbral)
            mostrar_zonas(frame, zonas)

            if preset_nombre:
                cv2.putText(frame, f"Preset: {preset_nombre}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.putText(frame, f"Personas detectadas: {len(personas)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Grupo mayor: {grupo_mas_grande}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Aglomeracion: {nivel_aglomeracion}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_aglomeracion, 2)

            if nivel_aglomeracion == "ALTO":
                cv2.putText(frame, "ALERTA: AGLOMERACION DETECTADA", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes +
                b"\r\n"
            )
    finally:
        cap.release()

        # Borrar el video después de procesarlo (ya no se necesita)
        try:
            if os.path.exists(ruta_entrada):
                os.remove(ruta_entrada)
        except Exception as e:
            print(f"[WARN] No se pudo borrar {ruta_entrada}: {e}")

        # Calcular nivel final desde el pico real, no del último frame
        nivel_final, _ = clasificar_aglomeracion(grupo_mayor_maximo)

        # Guardar el análisis en la base de datos
        try:
            guardar_analisis(
                nombre_video,
                personas_maximas,
                grupo_mayor_maximo,
                nivel_final,
                user_id=user_id,
                preset_id=preset_id
            )
        except Exception as e:
            print(f"[WARN] No se pudo guardar el análisis: {e}")