import cv2
import time
import os


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


def procesar_video(cap, fuente):
    print("Procesando video...")
    print("Presiona ESC para salir.")

    tiempo_anterior = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fin del video o error al leer la fuente.")
            break

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

        cv2.imshow("Procesamiento de video", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            print("Procesamiento detenido por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    resultado = seleccionar_fuente()

    if resultado is None:
        print("No se pudo iniciar el sistema.")
        return

    cap, fuente = resultado
    procesar_video(cap, fuente)


if __name__ == "__main__":
    main()