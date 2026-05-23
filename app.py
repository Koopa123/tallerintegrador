from database import listar_analisis
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import shutil
import os
import cv2

from detector import (
    generar_stream_video,
    guardar_zonas_web,
    obtener_zonas_web,
    borrar_zonas_web
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CARPETA_VIDEOS = "videos_entrada"
CARPETA_FRAMES = "frames_referencia"

os.makedirs(CARPETA_VIDEOS, exist_ok=True)
os.makedirs(CARPETA_FRAMES, exist_ok=True)


class ZonasRequest(BaseModel):
    zonas: list


def limpiar_carpeta(carpeta, max_archivos=2):
    archivos = [
        os.path.join(carpeta, archivo)
        for archivo in os.listdir(carpeta)
        if os.path.isfile(os.path.join(carpeta, archivo))
    ]

    archivos.sort(key=os.path.getmtime)

    while len(archivos) > max_archivos:
        archivo_antiguo = archivos.pop(0)
        os.remove(archivo_antiguo)


@app.get("/")
def inicio():
    return {
        "mensaje": "API de detección de aglomeraciones funcionando"
    }


@app.post("/subir-video")
async def subir_video(file: UploadFile = File(...)):
    ruta_entrada = os.path.join(CARPETA_VIDEOS, file.filename)

    with open(ruta_entrada, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    limpiar_carpeta(CARPETA_VIDEOS, max_archivos=2)

    return {
        "mensaje": "Video subido correctamente",
        "nombre_video": file.filename,
        "stream_url": f"/stream-video/{file.filename}"
    }


@app.get("/stream-video/{nombre_video}")
def stream_video(nombre_video: str):
    ruta_video = os.path.join(CARPETA_VIDEOS, nombre_video)

    return StreamingResponse(
        generar_stream_video(ruta_video, nombre_video),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/subir-video-referencia")
async def subir_video_referencia(file: UploadFile = File(...)):
    nombre_video = f"referencia_{file.filename}"
    nombre_frame = f"frame_{file.filename}.jpg"

    ruta_video = os.path.join(CARPETA_VIDEOS, nombre_video)
    ruta_frame = os.path.join(CARPETA_FRAMES, nombre_frame)

    with open(ruta_video, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(ruta_video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "No se pudo extraer el primer frame"}

    cv2.imwrite(ruta_frame, frame)

    limpiar_carpeta(CARPETA_VIDEOS, max_archivos=2)
    limpiar_carpeta(CARPETA_FRAMES, max_archivos=2)

    return {
        "mensaje": "Frame de referencia generado",
        "frame_url": f"/frame-referencia/{file.filename}"
    }


@app.get("/frame-referencia/{nombre_video}")
def obtener_frame_referencia(nombre_video: str):
    ruta_frame = os.path.join(CARPETA_FRAMES, f"frame_{nombre_video}.jpg")
    return FileResponse(ruta_frame)


@app.post("/guardar-zonas")
def guardar_zonas(data: ZonasRequest):
    guardar_zonas_web(data.zonas)

    return {
        "mensaje": "Zonas guardadas correctamente",
        "zonas": data.zonas
    }


@app.get("/zonas")
def obtener_zonas():
    return {
        "zonas": obtener_zonas_web()
    }


@app.delete("/zonas")
def borrar_zonas():
    borrar_zonas_web()

    return {
        "mensaje": "Zonas eliminadas correctamente"
    }

@app.get("/analisis")
def obtener_analisis():
    datos = listar_analisis()

    return {
        "analisis": [
            {
                "id": fila[0],
                "nombre_video": fila[1],
                "personas_maximas": fila[2],
                "grupo_mayor_maximo": fila[3],
                "nivel_final": fila[4],
                "fecha": fila[5]
            }
            for fila in datos
        ]
    }