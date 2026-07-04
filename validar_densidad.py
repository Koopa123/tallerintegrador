"""
Validacion de exactitud de clasificacion de densidad (BAJO/MEDIO/ALTO),
Project Charter objetivo especifico 2, usando el MISMO pipeline de
deteccion+agrupacion+clasificacion que corre en produccion (detector.py),
no una copia ni una version simplificada.

No hay atajo de codigo para esto: hace falta comparar contra un nivel
real juzgado por una persona mirando los frames (no existe ground truth
de "nivel real" en ningun lado, a diferencia del conteo de personas que
ya se valida en validar_precision.py).

USO:

  1) Generar la muestra de frames para juzgar a mano:

       python validar_densidad.py --generar

     Esto crea la carpeta validacion/muestra_densidad/ con imagenes
     donde ya se ven las personas detectadas, los grupos (lineas entre
     centros cercanos) y el nivel que el sistema clasifico, tomadas del
     mismo video y con la misma cadencia que usa validar_precision.py
     (cada VENTANA_MUESTREO_SEG segundos). Tambien crea
     validacion/resultados_densidad.csv con una fila por frame y la
     columna "nivel_real" vacia para completar a mano.

  2) Abrir cada imagen *_grupos.jpg en validacion/muestra_densidad/,
     juzgar a ojo si el nivel de aglomeracion real es BAJO/MEDIO/ALTO,
     y escribir eso en la columna "nivel_real" del CSV.

  3) Calcular las metricas ya con el CSV completo:

       python validar_densidad.py --calcular

     Imprime exactitud y error de clasificacion, comparados contra los
     umbrales del Project Charter (>=85%, <=15%).
"""
import argparse
import csv
import os

import cv2

from detector import (
    detectar_personas,
    dibujar_personas,
    dibujar_grupos,
    calcular_distancia_adaptativa,
    agrupar_personas,
    obtener_grupo_mas_grande,
    clasificar_aglomeracion,
)

# ============================================================
# CONFIGURACION
# ============================================================

VIDEO_A_VALIDAR = "videos_entrada/video1.mp4"  # mismo video que validar_precision.py

# Deliberadamente sin zonas ignoradas (zonas=[] mas abajo): se probo con las
# del preset real "pasillo c" (id 5) y, al ser rectangulos fijos pegados al
# borde del pasillo, tapaban tambien clientes reales que caminan por esa
# franja, no solo los maniquies de la vitrina (ver nota en
# validar_precision.py). Se valida el detector puro por la misma razon.
CARPETA_VALIDACION = "validacion"
CARPETA_MUESTRA = os.path.join(CARPETA_VALIDACION, "muestra_densidad")
CSV_RESULTADOS = os.path.join(CARPETA_VALIDACION, "resultados_densidad.csv")

VENTANA_MUESTREO_SEG = 1.5  # misma cadencia que validar_precision.py, para comparar los mismos frames

# Umbrales del Project Charter (objetivo especifico 2, aglomeracion/densidad)
UMBRAL_EXACTITUD_MIN = 0.85           # exactitud en zonas de alta densidad >= 85%
UMBRAL_ERROR_CLASIFICACION_MAX = 0.15  # error de clasificacion <= 15%


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

            personas = detectar_personas(frame, zonas=[])
            distancia_umbral = calcular_distancia_adaptativa(personas)
            grupos = agrupar_personas(personas, distancia_umbral)
            grupo_mas_grande = obtener_grupo_mas_grande(grupos)
            nivel_detectado, color = clasificar_aglomeracion(grupo_mas_grande)

            frame_anotado = frame.copy()
            dibujar_personas(frame_anotado, personas)
            dibujar_grupos(frame_anotado, personas, grupos, distancia_umbral)
            cv2.putText(
                frame_anotado, f"Nivel detectado: {nivel_detectado}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

            nombre_base = f"{indice_muestra:03d}_t{timestamp_seg:05.1f}s"
            ruta_grupos = os.path.join(CARPETA_MUESTRA, f"{nombre_base}_grupos.jpg")
            cv2.imwrite(ruta_grupos, frame_anotado)

            filas.append({
                "muestra": indice_muestra,
                "timestamp_seg": round(timestamp_seg, 1),
                "personas_detectadas": len(personas),
                "grupo_mas_grande": grupo_mas_grande,
                "nivel_detectado": nivel_detectado,
                "nivel_real": "",  # <- completar a mano mirando *_grupos.jpg
            })

            print(f"  muestra {indice_muestra:03d}  t={timestamp_seg:5.1f}s  "
                  f"grupo_mas_grande={grupo_mas_grande:2d}  nivel_detectado={nivel_detectado}")

            indice_muestra += 1

        indice_frame += 1

    cap.release()

    with open(CSV_RESULTADOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "muestra", "timestamp_seg", "personas_detectadas",
            "grupo_mas_grande", "nivel_detectado", "nivel_real"
        ])
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n{len(filas)} frames de muestra guardados en {CARPETA_MUESTRA}/")
    print(f"CSV generado: {CSV_RESULTADOS}")
    print("\nSiguiente paso:")
    print(f"  1. Abri las imagenes '*_grupos.jpg' en {CARPETA_MUESTRA}/")
    print(f"  2. Juzga a ojo el nivel REAL de aglomeracion (BAJO/MEDIO/ALTO)")
    print(f"  3. Escribi ese nivel en la columna 'nivel_real' de {CSV_RESULTADOS}")
    print(f"  4. Corre: python validar_densidad.py --calcular")


# ============================================================
# PASO 2: CALCULAR METRICAS CON EL CSV YA COMPLETADO
# ============================================================

def calcular_metricas():
    if not os.path.exists(CSV_RESULTADOS):
        raise SystemExit(f"No existe {CSV_RESULTADOS}. Corre primero --generar")

    with open(CSV_RESULTADOS, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    faltantes = [f["muestra"] for f in filas if not f["nivel_real"].strip()]
    if faltantes:
        raise SystemExit(
            f"Faltan completar 'nivel_real' en {len(faltantes)} fila(s): {faltantes}\n"
            f"Completa el CSV antes de calcular."
        )

    aciertos = 0
    for fila in filas:
        detectado = fila["nivel_detectado"].strip().upper()
        real = fila["nivel_real"].strip().upper()
        if detectado == real:
            aciertos += 1

    total = len(filas)
    exactitud = aciertos / total if total else 0
    error_clasificacion = 1 - exactitud

    def verdicto(cumple):
        return "CUMPLE" if cumple else "NO CUMPLE"

    print("=" * 60)
    print(f"Muestras evaluadas: {total}")
    print("=" * 60)

    print(f"\nProject Charter - Exactitud en zonas de alta densidad >= {UMBRAL_EXACTITUD_MIN:.0%}")
    print(f"  Aciertos: {aciertos}/{total}")
    print(f"  Exactitud: {exactitud:.1%}")
    print(f"  -> {verdicto(exactitud >= UMBRAL_EXACTITUD_MIN)}")

    print(f"\nProject Charter - Error de clasificacion (bajo/medio/alto) <= {UMBRAL_ERROR_CLASIFICACION_MAX:.0%}")
    print(f"  Error de clasificacion: {error_clasificacion:.1%}")
    print(f"  -> {verdicto(error_clasificacion <= UMBRAL_ERROR_CLASIFICACION_MAX)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generar", action="store_true", help="Genera la muestra de frames para juzgar a mano")
    parser.add_argument("--calcular", action="store_true", help="Calcula las metricas con el CSV ya completado")
    args = parser.parse_args()

    if args.generar:
        generar_muestra()
    elif args.calcular:
        calcular_metricas()
    else:
        parser.print_help()
