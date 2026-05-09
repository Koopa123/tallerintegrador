import cv2
import time
import os
import json
from ultralytics import YOLO

zonas_ignoradas = []
dibujando = False
x_inicial, y_inicial = -1, -1

ARCHIVO_ZONAS = "zonas.json"
NOMBRE_VENTANA = "Deteccion de personas y aglomeraciones"

def guardar_zonas():
    with open(ARCHIVO_ZONAS, "w") as archivo:
        json.dump(zonas_ignoradas, archivo)

    print("Zonas guardadas correctamente.")

def cargar_zonas():
    global zonas_ignoradas

    if os.path.exists(ARCHIVO_ZONAS):
        with open(ARCHIVO_ZONAS, "r") as archivo:
            zonas_ignoradas = json.load(archivo)

        print("Zonas cargadas correctamente.")
    else:
        print("No hay zonas guardadas todavía.")

def dibujar_rectangulo(evento, x, y, flags, param):
    global x_inicial, y_inicial, dibujando, zonas_ignoradas

    if evento == cv2.EVENT_LBUTTONDOWN:
        dibujando = True
        x_inicial, y_inicial = x, y

    elif evento == cv2.EVENT_LBUTTONUP:
        dibujando = False

        x_final, y_final = x, y

        zona = (
            min(x_inicial, x_final),
            min(y_inicial, y_final),
            max(x_inicial, x_final),
            max(y_inicial, y_final)
        )

        zonas_ignoradas.append(zona)
        guardar_zonas()

        print("Zona ignorada agregada:", zona)

def mostrar_zonas(frame):
    for zx1, zy1, zx2, zy2 in zonas_ignoradas:
        cv2.rectangle(
            frame,
            (zx1, zy1),
            (zx2, zy2),
            (80, 80, 80),
            1
        )

def esta_en_zona_ignorada(x1, y1, x2, y2):
    centro_x = int((x1 + x2) / 2)
    centro_y = int((y1 + y2) / 2)

    for zx1, zy1, zx2, zy2 in zonas_ignoradas:
        if zx1 <= centro_x <= zx2 and zy1 <= centro_y <= zy2:
            return True

    return False

def seleccionar_fuente():
    print("=== SISTEMA DE PROCESAMIENTO DE VIDEO ===")
    print("1. Cargar video pregrabado")
    print("2. Usar cámara en tiempo real")

    opcion = input("Seleccione una opción: ")

    if opcion == "1":
        ruta = input("Ingrese la ruta del video MP4: ")

        if not os.path.exists(ruta):
            print("Error: El archivo no existe.")
            return None

        if not ruta.lower().endswith(".mp4"):
            print("Error: Solo se permiten archivos MP4.")
            return None

        cap = cv2.VideoCapture(ruta)
        fuente = "Video pregrabado"

    elif opcion == "2":
        cap = cv2.VideoCapture(0)
        fuente = "Cámara en tiempo real"

    else:
        print("Error: opción no válida.")
        return None

    if not cap.isOpened():
        print("Error: No se pudo abrir la fuente de video.")
        return None

    print(f"Fuente seleccionada correctamente: {fuente}")
    return cap, fuente

def cargar_modelo():
    print("Cargando modelo YOLO...")

    modelo = YOLO("yolov8s.pt")

    print("Modelo cargado correctamente.")
    return modelo

def detectar_personas(frame, modelo):
    resultados = modelo(frame, verbose=False)

    cantidad_personas = 0

    for resultado in resultados:
        for caja in resultado.boxes:
            clase = int(caja.cls[0])
            confianza = float(caja.conf[0])

            if clase == 0 and confianza >= 0.50:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])

                if esta_en_zona_ignorada(x1, y1, x2, y2):
                    continue

                cantidad_personas += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Persona {confianza:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    return frame, cantidad_personas

def clasificar_aglomeracion(cantidad_personas):
    if cantidad_personas <= 1:
        return "BAJO", (0, 255, 0)

    elif cantidad_personas <= 3:
        return "MEDIO", (0, 255, 255)

    else:
        return "ALTO", (0, 0, 255)

def procesar_video(cap, fuente, modelo):
    print("Procesando video con detección de personas y aglomeraciones...")
    print("Controles:")
    print("- Click y arrastrar: crear zona ignorada")
    print("- C: borrar todas las zonas")
    print("- ESC: salir")

    tiempo_anterior = time.time()

    cv2.namedWindow(NOMBRE_VENTANA)
    cv2.setMouseCallback(NOMBRE_VENTANA, dibujar_rectangulo)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fin del video o error al leer la fuente.")
            break

        frame, cantidad_personas = detectar_personas(frame, modelo)

        nivel_aglomeracion, color_aglomeracion = clasificar_aglomeracion(
            cantidad_personas
        )

        mostrar_zonas(frame)

        tiempo_actual = time.time()
        diferencia = tiempo_actual - tiempo_anterior

        if diferencia > 0:
            fps = 1 / diferencia
        else:
            fps = 0

        tiempo_anterior = tiempo_actual

        cv2.putText(
            frame,
            f"Fuente: {fuente}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Personas detectadas: {cantidad_personas}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Aglomeracion: {nivel_aglomeracion}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_aglomeracion,
            2
        )

        if nivel_aglomeracion == "ALTO":
            cv2.putText(
                frame,
                "ALERTA: AGLOMERACION EN PASADIZO PUERTA DERECHA",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(NOMBRE_VENTANA, frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            print("Procesamiento detenido por el usuario.")
            break

        elif tecla == ord("c"):
            zonas_ignoradas.clear()
            guardar_zonas()
            print("Zonas eliminadas correctamente.")

    cap.release()
    cv2.destroyAllWindows()

def main():
    cargar_zonas()

    resultado = seleccionar_fuente()

    if resultado is None:
        print("No se pudo iniciar el sistema.")
        return

    cap, fuente = resultado
    modelo = cargar_modelo()
    procesar_video(cap, fuente, modelo)

if __name__ == "__main__":
    main()