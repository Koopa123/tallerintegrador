"""
Validacion de precision, error de conteo y tiempo de procesamiento del
detector de personas (HU-2.1 CA3, HU-2.3 CA3, HU-6.2), usando el MISMO
detector.py que corre en produccion (no una copia ni una version simplificada).

No hay atajo de codigo para esto: hace falta comparar contra un conteo real
hecho por una persona mirando los frames.

USO:

  1) Generar la muestra de frames para contar a mano:

       python validar_precision.py --generar

     Esto crea la carpeta validacion/muestra/ con imagenes de frames
     tomados del video cada VENTANA_MUESTREO_SEG segundos, y un archivo
     validacion/resultados_muestra.csv con una fila por frame.

  2) Abrir cada imagen *_original.jpg en validacion/muestra/, contar a
     ojo cuantas personas hay REALMENTE, y escribir ese numero en la
     columna "personas_reales" del CSV (viene vacia).

  3) Calcular las metricas ya con el CSV completo:

       python validar_precision.py --calcular

     Imprime error de conteo, precision aproximada y tiempo de
     procesamiento promedio, comparados contra los umbrales de las
     historias de usuario (>=85%, <=10%, <=100ms).
"""
import argparse
import csv
import os
import time

import cv2

from detector import detectar_personas, dibujar_personas

# ============================================================
# CONFIGURACION
# ============================================================

VIDEO_A_VALIDAR = "videos_entrada/video1.mp4"
CARPETA_VALIDACION = "validacion"
CARPETA_MUESTRA = os.path.join(CARPETA_VALIDACION, "muestra")
CSV_RESULTADOS = os.path.join(CARPETA_VALIDACION, "resultados_muestra.csv")

VENTANA_MUESTREO_SEG = 1.5  # cada cuantos segundos de video se toma un frame de muestra

# Umbrales que definen las historias de usuario a validar
UMBRAL_PRECISION_MIN = 0.85   # HU-2.1 CA3: precision >= 85%
UMBRAL_ERROR_CONTEO_MAX = 0.10  # HU-2.3 CA3: error de conteo <= 10%
UMBRAL_TIEMPO_MS_MAX = 100    # HU-6.2 CA1: <=100ms por frame


# ============================================================
# PASO 1: GENERAR LA MUESTRA
# ============================================================

def generar_muestra():
    if not os.path.exists(VIDEO_A_VALIDAR):
        raise SystemExit(f"No se encontro el video: {VIDEO_A_VALIDAR}")

    os.makedirs(CARPETA_MUESTRA, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_A_VALIDAR)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    intervalo_frames = max(int(round(fps_video * VENTANA_MUESTREO_SEG)), 1)

    filas = []
    indice_frame = 0
    indice_muestra = 0

    print(f"Video: {VIDEO_A_VALIDAR} ({fps_video:.1f} fps)")
    print(f"Tomando 1 frame cada {VENTANA_MUESTREO_SEG}s (~cada {intervalo_frames} frames)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if indice_frame % intervalo_frames == 0:
            timestamp_seg = indice_frame / fps_video

            # Medimos el tiempo de deteccion real, con el detector de produccion.
            inicio = time.perf_counter()
            personas = detectar_personas(frame, zonas=[])
            tiempo_ms = (time.perf_counter() - inicio) * 1000

            frame_anotado = frame.copy()
            dibujar_personas(frame_anotado, personas)

            nombre_base = f"{indice_muestra:03d}_t{timestamp_seg:05.1f}s"
            ruta_original = os.path.join(CARPETA_MUESTRA, f"{nombre_base}_original.jpg")
            ruta_detectado = os.path.join(CARPETA_MUESTRA, f"{nombre_base}_detectado.jpg")

            cv2.imwrite(ruta_original, frame)
            cv2.imwrite(ruta_detectado, frame_anotado)

            filas.append({
                "muestra": indice_muestra,
                "timestamp_seg": round(timestamp_seg, 1),
                "personas_detectadas": len(personas),
                "tiempo_ms": round(tiempo_ms, 1),
                "personas_reales": "",  # <- completar a mano mirando *_original.jpg
            })

            print(f"  muestra {indice_muestra:03d}  t={timestamp_seg:5.1f}s  "
                  f"detectadas={len(personas):2d}  tiempo={tiempo_ms:6.1f}ms")

            indice_muestra += 1

        indice_frame += 1

    cap.release()

    with open(CSV_RESULTADOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "muestra", "timestamp_seg", "personas_detectadas", "tiempo_ms", "personas_reales"
        ])
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n{len(filas)} frames de muestra guardados en {CARPETA_MUESTRA}/")
    print(f"CSV generado: {CSV_RESULTADOS}")
    print("\nSiguiente paso:")
    print(f"  1. Abri las imagenes '*_original.jpg' en {CARPETA_MUESTRA}/")
    print(f"  2. Conta a ojo cuantas personas hay REALMENTE en cada una")
    print(f"  3. Escribi ese numero en la columna 'personas_reales' de {CSV_RESULTADOS}")
    print(f"  4. Corre: python validar_precision.py --calcular")


# ============================================================
# PASO 2: CALCULAR METRICAS CON EL CSV YA COMPLETADO
# ============================================================

def calcular_metricas():
    if not os.path.exists(CSV_RESULTADOS):
        raise SystemExit(f"No existe {CSV_RESULTADOS}. Corre primero --generar")

    with open(CSV_RESULTADOS, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    faltantes = [f["muestra"] for f in filas if not f["personas_reales"].strip()]
    if faltantes:
        raise SystemExit(
            f"Faltan completar 'personas_reales' en {len(faltantes)} fila(s): {faltantes}\n"
            f"Completa el CSV antes de calcular."
        )

    total_tp = total_fp = total_fn = 0
    errores_pct = []
    tiempos_ms = []

    for fila in filas:
        detectadas = int(fila["personas_detectadas"])
        reales = int(fila["personas_reales"])
        tiempo_ms = float(fila["tiempo_ms"])
        tiempos_ms.append(tiempo_ms)

        # Aproximacion por conteo (no hay bounding boxes de referencia,
        # asi que se estima TP/FP/FN comparando cuantas se detectaron vs
        # cuantas habia realmente). Es una aproximacion razonable para
        # validar el sistema, no un matching exacto por IoU.
        tp = min(detectadas, reales)
        fp = max(0, detectadas - reales)
        fn = max(0, reales - detectadas)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if reales > 0:
            errores_pct.append(abs(detectadas - reales) / reales)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    error_conteo_promedio = sum(errores_pct) / len(errores_pct) if errores_pct else 0
    tiempo_promedio_ms = sum(tiempos_ms) / len(tiempos_ms)
    tiempo_max_ms = max(tiempos_ms)

    def verdicto(cumple):
        return "CUMPLE" if cumple else "NO CUMPLE"

    print("=" * 60)
    print(f"Muestras evaluadas: {len(filas)}")
    print("=" * 60)

    print(f"\nHU-2.1 CA3 - Precision >= {UMBRAL_PRECISION_MIN:.0%}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision estimada: {precision:.1%}  Recall estimado: {recall:.1%}")
    print(f"  -> {verdicto(precision >= UMBRAL_PRECISION_MIN)}")

    print(f"\nHU-2.3 CA3 - Error de conteo <= {UMBRAL_ERROR_CONTEO_MAX:.0%}")
    print(f"  Error promedio: {error_conteo_promedio:.1%}")
    print(f"  -> {verdicto(error_conteo_promedio <= UMBRAL_ERROR_CONTEO_MAX)}")

    print(f"\nHU-6.2 CA1 - Tiempo de procesamiento <= {UMBRAL_TIEMPO_MS_MAX}ms por frame")
    print(f"  Promedio: {tiempo_promedio_ms:.1f}ms   Maximo observado: {tiempo_max_ms:.1f}ms")
    print(f"  -> {verdicto(tiempo_promedio_ms <= UMBRAL_TIEMPO_MS_MAX)}")
    print("=" * 60)

    print(
        "\nNota: la 'precision' de arriba es una aproximacion por conteo "
        "(cuantas detectadas vs. cuantas reales por frame), no un matching "
        "exacto persona por persona con bounding boxes de referencia. Es "
        "una forma honesta y liviana de estimarla sin tener que anotar "
        "cajas a mano en cada frame."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generar", action="store_true", help="Genera la muestra de frames para contar a mano")
    parser.add_argument("--calcular", action="store_true", help="Calcula las metricas con el CSV ya completado")
    args = parser.parse_args()

    if args.generar:
        generar_muestra()
    elif args.calcular:
        calcular_metricas()
    else:
        parser.print_help()
