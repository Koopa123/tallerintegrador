import cv2
import os
import json
import math
from ultralytics import YOLO

ARCHIVO_ZONAS = "zonas.json"
DISTANCIA_AGRUPACION = 100
CONFIANZA_MINIMA = 0.50

zonas_ignoradas = []


def cargar_zonas():
    global zonas_ignoradas

    if os.path.exists(ARCHIVO_ZONAS):
        try:
            with open(ARCHIVO_ZONAS, "r") as archivo:
                contenido = archivo.read().strip()

                if contenido == "":
                    zonas_ignoradas = []
                else:
                    zonas_ignoradas = json.loads(contenido)

        except json.JSONDecodeError:
            zonas_ignoradas = []
    else:
        zonas_ignoradas = []


def cargar_modelo():
    modelo = YOLO("yolov8n.pt")
    return modelo


modelo = cargar_modelo()
cargar_zonas()


def esta_en_zona_ignorada(x1, y1, x2, y2):
    centro_x = int((x1 + x2) / 2)
    centro_y = int((y1 + y2) / 2)

    for zx1, zy1, zx2, zy2 in zonas_ignoradas:
        if zx1 <= centro_x <= zx2 and zy1 <= centro_y <= zy2:
            return True

    return False


def detectar_personas(frame):
    resultados = modelo(frame, verbose=False)

    personas = []

    for resultado in resultados:
        for caja in resultado.boxes:
            clase = int(caja.cls[0])
            confianza = float(caja.conf[0])

            if clase == 0 and confianza >= CONFIANZA_MINIMA:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])

                if esta_en_zona_ignorada(x1, y1, x2, y2):
                    continue

                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)

                personas.append({
                    "bbox": (x1, y1, x2, y2),
                    "centro": (centro_x, centro_y),
                    "confianza": confianza
                })

    return personas


def calcular_distancia(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def agrupar_personas(personas):
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

                    if distancia <= DISTANCIA_AGRUPACION:
                        visitados.add(j)
                        cola.append(j)

        grupos.append(grupo_actual)

    return grupos


def obtener_grupo_mas_grande(grupos):
    if not grupos:
        return 0

    return max(len(grupo) for grupo in grupos)


def clasificar_aglomeracion(grupo_mas_grande):
    if grupo_mas_grande <= 1:
        return "BAJO", (0, 255, 0)

    elif grupo_mas_grande <= 3:
        return "MEDIO", (0, 255, 255)

    else:
        return "ALTO", (0, 0, 255)


def mostrar_zonas(frame):
    for zx1, zy1, zx2, zy2 in zonas_ignoradas:
        cv2.rectangle(
            frame,
            (zx1, zy1),
            (zx2, zy2),
            (80, 80, 80),
            1
        )


def dibujar_personas(frame, personas):
    for persona in personas:
        x1, y1, x2, y2 = persona["bbox"]
        centro_x, centro_y = persona["centro"]
        confianza = persona["confianza"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (centro_x, centro_y), 4, (255, 0, 0), -1)

        cv2.putText(
            frame,
            f"Persona {confianza:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


def dibujar_grupos(frame, personas, grupos):
    for grupo in grupos:
        if len(grupo) <= 1:
            continue

        puntos = [personas[i]["centro"] for i in grupo]

        for i in range(len(puntos)):
            for j in range(i + 1, len(puntos)):
                distancia = calcular_distancia(puntos[i], puntos[j])

                if distancia <= DISTANCIA_AGRUPACION:
                    cv2.line(frame, puntos[i], puntos[j], (255, 255, 0), 1)


def procesar_video_web(ruta_entrada, ruta_salida):
    cargar_zonas()

    cap = cv2.VideoCapture(ruta_entrada)

    if not cap.isOpened():
        raise Exception("No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = 24

    ruta_salida = ruta_salida.replace(".mp4", ".avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    salida = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))

    personas_maximas = 0
    grupo_mayor_maximo = 0
    nivel_final = "BAJO"

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        personas = detectar_personas(frame)
        grupos = agrupar_personas(personas)
        grupo_mas_grande = obtener_grupo_mas_grande(grupos)

        nivel_aglomeracion, color_aglomeracion = clasificar_aglomeracion(
            grupo_mas_grande
        )

        personas_maximas = max(personas_maximas, len(personas))
        grupo_mayor_maximo = max(grupo_mayor_maximo, grupo_mas_grande)
        nivel_final = nivel_aglomeracion

        dibujar_personas(frame, personas)
        dibujar_grupos(frame, personas, grupos)
        mostrar_zonas(frame)

        cv2.putText(
            frame,
            f"Personas detectadas: {len(personas)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Grupo mayor: {grupo_mas_grande}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Aglomeracion: {nivel_aglomeracion}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_aglomeracion,
            2
        )

        if nivel_aglomeracion == "ALTO":
            cv2.putText(
                frame,
                "ALERTA: AGLOMERACION DETECTADA",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        salida.write(frame)

    cap.release()
    salida.release()

    return {
        "personas_maximas": personas_maximas,
        "grupo_mayor_maximo": grupo_mayor_maximo,
        "nivel_final": nivel_final,
        "ruta_salida": ruta_salida
    }

def generar_stream_video(ruta_entrada, nombre_video="video"):
    from database import guardar_analisis

    cargar_zonas()

    cap = cv2.VideoCapture(ruta_entrada)

    if not cap.isOpened():
        raise Exception("No se pudo abrir el video.")

    personas_maximas = 0
    grupo_mayor_maximo = 0
    nivel_final = "BAJO"

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        personas = detectar_personas(frame)
        grupos = agrupar_personas(personas)
        grupo_mas_grande = obtener_grupo_mas_grande(grupos)

        nivel_aglomeracion, color_aglomeracion = clasificar_aglomeracion(
            grupo_mas_grande
        )

        personas_maximas = max(personas_maximas, len(personas))
        grupo_mayor_maximo = max(grupo_mayor_maximo, grupo_mas_grande)
        nivel_final = nivel_aglomeracion

        dibujar_personas(frame, personas)
        dibujar_grupos(frame, personas, grupos)
        mostrar_zonas(frame)

        cv2.putText(
            frame,
            f"Personas detectadas: {len(personas)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Grupo mayor: {grupo_mas_grande}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Aglomeracion: {nivel_aglomeracion}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_aglomeracion,
            2
        )

        if nivel_aglomeracion == "ALTO":
            cv2.putText(
                frame,
                "ALERTA: AGLOMERACION DETECTADA",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
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

    cap.release()

    guardar_analisis(
        nombre_video,
        personas_maximas,
        grupo_mayor_maximo,
        nivel_final
    )

def guardar_zonas_web(zonas):
    global zonas_ignoradas

    zonas_ignoradas = zonas

    with open(ARCHIVO_ZONAS, "w") as archivo:
        json.dump(zonas_ignoradas, archivo)

    cargar_zonas()


def obtener_zonas_web():
    cargar_zonas()
    return zonas_ignoradas


def borrar_zonas_web():
    global zonas_ignoradas

    zonas_ignoradas = []

    with open(ARCHIVO_ZONAS, "w") as archivo:
        json.dump([], archivo)