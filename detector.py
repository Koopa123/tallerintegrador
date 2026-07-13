import os
import time
import cv2
import math
from ultralytics import YOLO

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Se probo cambiar YOLOv8s por modelos mas pesados (YOLOv8m, YOLOv8l) para
# ver si mejoraban el error de conteo (HU-2.3) y la exactitud de
# clasificacion de densidad (validar_densidad.py, que dio solo 46.2% con
# YOLOv8s+0.55). YOLOv8l no mejora sobre YOLOv8m (no es "mientras mas
# pesado, mejor"); YOLOv8m es el mejor punto de los tres. Con YOLOv8m se
# volvio a barrer el umbral (0.20-0.70): valores bajos (<=0.30) dan
# "100% de exactitud de densidad" pero es un artefacto -- las 26 muestras
# de validacion son todas nivel ALTO, asi que cualquier umbral que
# sobre-detecte (mas falsos positivos, ver precision/error de conteo en
# esa misma fila) termina clasificando ALTO por pura casualidad, no
# porque distinga bien los niveles. Ignorando esos puntos degenerados,
# 0.60 es el mejor real: mejor F1 (0.94) y mejor error de conteo (13.4%)
# de todo el experimento (incluyendo el YOLOv8s+0.55 anterior), y
# exactitud de densidad 80.8% (vs 46.2% antes). Costo de velocidad
# despreciable en GPU (~9ms -> ~13ms por frame, muy lejos del limite de
# 100ms). Sigue sin alcanzar el <=10% de error de conteo (HU-2.3) ni el
# >=85% de exactitud de densidad del Project Charter -- limite real del
# modelo en escenas con oclusion, no arreglado del todo, pero mejorado.
CONFIANZA_MINIMA = 0.60

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

# Streams de análisis en curso, para poder detenerlos desde el botón
# "Detener" del frontend (antes la única forma de salir era matar el
# proceso del backend). Se indexan por el nombre físico único del video
# (el mismo que viaja en la URL del stream), no por el nombre "de
# exhibición" que se usa para guardar en la BD.
STREAMS_ACTIVOS = {}    # nombre_fisico -> user_id
STREAMS_CANCELADOS = set()  # nombres marcados para detener

# ============================================================
# CARGA DEL MODELO
# ============================================================

def cargar_modelo():
    # YOLOv8m en vez de YOLOv8s (ver nota de CONFIANZA_MINIMA mas arriba):
    # mejora F1, error de conteo y exactitud de densidad, a costo de
    # velocidad despreciable en GPU.
    return YOLO("yolov8m.pt")


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


def obtener_personas_grupo_mas_grande(personas, grupos):
    """Devuelve las personas (no solo el conteo) del grupo más numeroso."""
    if not grupos:
        return []
    grupo = max(grupos, key=len)
    return [personas[i] for i in grupo]


# ============================================================
# CLASIFICACIÓN
# ============================================================

def clasificar_aglomeracion(grupo_mas_grande, umbral_medio=UMBRAL_MEDIO, umbral_alto=UMBRAL_ALTO):
    if grupo_mas_grande <= 1:
        return "BAJO", (0, 255, 0)
    elif grupo_mas_grande < umbral_medio:
        return "BAJO", (0, 255, 0)
    elif grupo_mas_grande < umbral_alto:
        return "MEDIO", (0, 255, 255)
    else:
        return "ALTO", (0, 0, 255)


# ============================================================
# ZONA Y MOMENTO DE LA ALERTA
# ============================================================
# Misma idea que obtenerZonaPromedio()/obtenerZona() en Realtime.jsx del
# frontend, para que la alerta de un video pregrabado sea igual de útil
# que la de la cámara en vivo: no solo "hay aglomeración", sino dónde y
# en qué momento del video.

def calcular_zona(personas, frame_width, frame_height):
    if not personas or not frame_width or not frame_height:
        return "Vista actual"

    suma_x = sum(p["centro"][0] for p in personas)
    suma_y = sum(p["centro"][1] for p in personas)
    cx = suma_x / len(personas)
    cy = suma_y / len(personas)

    if cy < frame_height / 3:
        vertical = "superior"
    elif cy > frame_height * 2 / 3:
        vertical = "inferior"
    else:
        vertical = "centro"

    if cx < frame_width / 3:
        horizontal = "izquierda"
    elif cx > frame_width * 2 / 3:
        horizontal = "derecha"
    else:
        horizontal = "centro"

    if vertical == "centro" and horizontal == "centro":
        return "Centro"
    if vertical == "centro":
        return f"Centro {horizontal}"
    if horizontal == "centro":
        return f"{vertical} centro"
    return f"{vertical} {horizontal}"


def formatear_momento(segundos):
    minutos = int(segundos) // 60
    segs = int(segundos) % 60
    return f"{minutos:02d}:{segs:02d}"


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

def generar_stream_video(ruta_entrada, nombre_video="video", zonas=None, preset_id=None, preset_nombre=None,
                          user_id=None, umbral_medio=UMBRAL_MEDIO, umbral_alto=UMBRAL_ALTO):
    from database import guardar_analisis

    zonas = zonas or []

    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        raise Exception("No se pudo abrir el video.")

    nombre_fisico = os.path.basename(ruta_entrada)
    STREAMS_ACTIVOS[nombre_fisico] = user_id
    detenido_manualmente = False

    personas_maximas = 0
    grupo_mayor_maximo = 0

    # Zona y momento del PRIMER frame en que se llega a ALTO, para guardarlos
    # junto con el análisis (antes la alerta era solo un texto fijo en el
    # video, sin decir dónde ni cuándo pasó).
    zona_alerta = None
    momento_alerta_seg = None

    # FPS de procesamiento (cuadros/segundo reales que el servidor logra
    # analizar, no el framerate del video). Antes solo se mostraba en
    # Tiempo Real; acá se calcula igual, suavizado con media móvil.
    tiempo_frame_anterior = time.perf_counter()
    fps_procesamiento = 0.0

    try:
        while True:
            if nombre_fisico in STREAMS_CANCELADOS:
                detenido_manualmente = True
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            momento_actual_seg = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            ahora = time.perf_counter()
            dt = ahora - tiempo_frame_anterior
            if dt > 0:
                fps_instantaneo = 1.0 / dt
                fps_procesamiento = (
                    0.8 * fps_procesamiento + 0.2 * fps_instantaneo
                    if fps_procesamiento > 0
                    else fps_instantaneo
                )
            tiempo_frame_anterior = ahora

            personas = detectar_personas(frame, zonas)

            # Distancia de agrupación adaptativa según tamaño de las personas
            distancia_umbral = calcular_distancia_adaptativa(personas)

            grupos = agrupar_personas(personas, distancia_umbral)
            grupo_mas_grande = obtener_grupo_mas_grande(grupos)

            # Nivel del frame actual (solo para mostrar en pantalla)
            nivel_aglomeracion, color_aglomeracion = clasificar_aglomeracion(grupo_mas_grande, umbral_medio, umbral_alto)

            personas_maximas = max(personas_maximas, len(personas))
            grupo_mayor_maximo = max(grupo_mayor_maximo, grupo_mas_grande)

            dibujar_personas(frame, personas)
            dibujar_grupos(frame, personas, grupos, distancia_umbral)
            mostrar_zonas(frame, zonas)

            if preset_nombre:
                cv2.putText(frame, f"Preset: {preset_nombre}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # FPS de procesamiento, arriba a la derecha (mismo criterio que Tiempo Real)
            texto_fps = f"FPS: {fps_procesamiento:.1f}"
            (ancho_texto, _), _ = cv2.getTextSize(texto_fps, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, texto_fps, (frame_w - ancho_texto - 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.putText(frame, f"Personas detectadas: {len(personas)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Grupo mayor: {grupo_mas_grande}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Aglomeracion: {nivel_aglomeracion}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_aglomeracion, 2)

            if nivel_aglomeracion == "ALTO":
                personas_grupo = obtener_personas_grupo_mas_grande(personas, grupos)
                zona_actual = calcular_zona(personas_grupo, frame_w, frame_h)

                # Solo se guarda la primera vez que aparece ALTO en el video.
                if zona_alerta is None:
                    zona_alerta = zona_actual
                    momento_alerta_seg = momento_actual_seg

                # cv2.putText solo soporta ASCII (los fonts Hershey no
                # tienen glifos para "—" u otros caracteres Unicode), así
                # que se usa un guion simple en vez de guion largo.
                cv2.putText(
                    frame,
                    f"ALERTA: AGLOMERACION DETECTADA - Zona: {zona_actual} - Momento: {formatear_momento(momento_actual_seg)}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
                )

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
        STREAMS_ACTIVOS.pop(nombre_fisico, None)
        STREAMS_CANCELADOS.discard(nombre_fisico)

        # Borrar el video después de procesarlo (ya no se necesita)
        try:
            if os.path.exists(ruta_entrada):
                os.remove(ruta_entrada)
        except Exception as e:
            print(f"[WARN] No se pudo borrar {ruta_entrada}: {e}")

        # Si el usuario lo detuvo a mano, el análisis quedó incompleto:
        # no se guarda, para no ensuciar el historial ni las métricas de
        # validación con conteos parciales que no representan el video.
        if detenido_manualmente:
            return

        # Calcular nivel final desde el pico real, no del último frame
        nivel_final, _ = clasificar_aglomeracion(grupo_mayor_maximo, umbral_medio, umbral_alto)

        # Guardar el análisis en la base de datos
        try:
            guardar_analisis(
                nombre_video,
                personas_maximas,
                grupo_mayor_maximo,
                nivel_final,
                user_id=user_id,
                preset_id=preset_id,
                zona_alerta=zona_alerta,
                momento_alerta_seg=momento_alerta_seg
            )
        except Exception as e:
            print(f"[WARN] No se pudo guardar el análisis: {e}")