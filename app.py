from database import (
    listar_analisis,
    crear_preset,
    listar_presets,
    obtener_preset,
    actualizar_zonas_preset,
    eliminar_preset,
    crear_usuario,
    obtener_usuario_por_email,
)
from auth import (
    hashear_password,
    verificar_password,
    crear_token,
    requerir_auth,
)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import shutil
import os
import cv2
import uuid

from database import crear_usuario, obtener_usuario_por_email
from psycopg2.errors import UniqueViolation

from detector import generar_stream_video

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
    

class RegistroRequest(BaseModel):
    nombre: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str

def limpiar_carpeta(carpeta, max_archivos=5):
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
    return {"mensaje": "API de detección de aglomeraciones funcionando"}

# ============================================================
# AUTENTICACIÓN
# ============================================================

@app.post("/auth/registro")
def registro(data: RegistroRequest):
    if len(data.password) < 6:
        raise HTTPException(400, "La contraseña debe tener al menos 6 caracteres")

    if "@" not in data.email:
        raise HTTPException(400, "Email inválido")

    password_hash = hashear_password(data.password)

    try:
        usuario = crear_usuario(data.nombre, data.email, password_hash)
    except UniqueViolation:
        raise HTTPException(409, "Ese email ya está registrado")
    except Exception as e:
        raise HTTPException(500, f"Error creando usuario: {str(e)}")

    user_id, nombre, email = usuario
    token = crear_token(user_id, email)

    return {
        "token": token,
        "usuario": {"id": user_id, "nombre": nombre, "email": email}
    }


@app.post("/auth/login")
def login(data: LoginRequest):
    usuario = obtener_usuario_por_email(data.email)
    if not usuario:
        raise HTTPException(401, "Email o contraseña incorrectos")

    user_id, nombre, email, password_hash = usuario

    if not verificar_password(data.password, password_hash):
        raise HTTPException(401, "Email o contraseña incorrectos")

    token = crear_token(user_id, email)

    return {
        "token": token,
        "usuario": {"id": user_id, "nombre": nombre, "email": email}
    }


@app.get("/auth/me")
def yo(payload: dict = Depends(requerir_auth)):
    """Endpoint para que el frontend valide si el token sigue siendo válido."""
    return {
        "id": int(payload["sub"]),
        "email": payload["email"],
    }


# ============================================================
# PRESETS
# ============================================================

@app.post("/presets")
async def crear_preset_endpoint(
    nombre: str = Form(...),
    file: UploadFile = File(...),
    _: dict = Depends(requerir_auth)   # ← protegido
):
    preset_uuid = str(uuid.uuid4())
    ruta_video_temp = os.path.join(CARPETA_VIDEOS, f"ref_{preset_uuid}.mp4")
    nombre_frame = f"frame_{preset_uuid}.jpg"
    ruta_frame = os.path.join(CARPETA_FRAMES, nombre_frame)

    with open(ruta_video_temp, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(ruta_video_temp)
    ret, frame = cap.read()
    cap.release()

    if os.path.exists(ruta_video_temp):
        os.remove(ruta_video_temp)

    if not ret:
        raise HTTPException(status_code=400, detail="No se pudo extraer el primer frame del video")

    cv2.imwrite(ruta_frame, frame)

    try:
        preset_id = crear_preset(nombre=nombre, frame_path=nombre_frame, zonas=[])
    except Exception as e:
        if os.path.exists(ruta_frame):
            os.remove(ruta_frame)
        raise HTTPException(status_code=400, detail=f"Error al crear preset (¿nombre duplicado?): {str(e)}")

    return {
        "id": preset_id,
        "nombre": nombre,
        "frame_url": f"/presets/{preset_id}/frame",
        "zonas": []
    }


@app.get("/presets")
def listar_presets_endpoint(
    _: dict = Depends(requerir_auth)   # ← protegido
):
    presets = listar_presets()
    return {
        "presets": [
            {
                "id": p[0],
                "nombre": p[1],
                "frame_url": f"/presets/{p[0]}/frame",
                "zonas": p[3],
                "fecha_creacion": p[4].isoformat() if p[4] else None
            }
            for p in presets
        ]
    }


@app.get("/presets/{preset_id}")
def obtener_preset_endpoint(
    preset_id: int,
    _: dict = Depends(requerir_auth)   # ← protegido
):
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    return {
        "id": preset[0],
        "nombre": preset[1],
        "frame_url": f"/presets/{preset_id}/frame",
        "zonas": preset[3],
        "fecha_creacion": preset[4].isoformat() if preset[4] else None
    }


@app.put("/presets/{preset_id}/zonas")
def actualizar_zonas_endpoint(
    preset_id: int,
    data: ZonasRequest,
    _: dict = Depends(requerir_auth)   # ← protegido
):
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    actualizar_zonas_preset(preset_id, data.zonas)
    return {"mensaje": "Zonas actualizadas", "zonas": data.zonas}


@app.delete("/presets/{preset_id}")
def eliminar_preset_endpoint(
    preset_id: int,
    _: dict = Depends(requerir_auth)   # ← protegido
):
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    ruta_frame = os.path.join(CARPETA_FRAMES, preset[2])
    if os.path.exists(ruta_frame):
        os.remove(ruta_frame)

    eliminar_preset(preset_id)
    return {"mensaje": "Preset eliminado"}


@app.get("/presets/{preset_id}/frame")
def obtener_frame_preset(preset_id: int):
    # ← SIN proteger (se carga como <img src>, no acepta headers)
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    ruta_frame = os.path.join(CARPETA_FRAMES, preset[2])
    if not os.path.exists(ruta_frame):
        raise HTTPException(status_code=404, detail="Frame del preset no existe en disco")
    return FileResponse(ruta_frame)


# ============================================================
# ANÁLISIS
# ============================================================

@app.post("/analisis")
async def iniciar_analisis(
    preset_id: int = Form(...),
    file: UploadFile = File(...),
    _: dict = Depends(requerir_auth)   # ← protegido
):
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    nombre_unico = f"{uuid.uuid4()}_{file.filename}"
    ruta_entrada = os.path.join(CARPETA_VIDEOS, nombre_unico)

    with open(ruta_entrada, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    limpiar_carpeta(CARPETA_VIDEOS, max_archivos=5)

    return {
        "mensaje": "Video subido, iniciando análisis",
        "nombre_video": file.filename,
        "stream_url": f"/analisis/stream/{nombre_unico}?preset_id={preset_id}&nombre_original={file.filename}"
    }


@app.get("/analisis/stream/{nombre_video}")
def stream_analisis(nombre_video: str, preset_id: int, nombre_original: str = None):
    # ← SIN proteger (se carga como <img src> stream, no acepta headers)
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    ruta_video = os.path.join(CARPETA_VIDEOS, nombre_video)
    if not os.path.exists(ruta_video):
        raise HTTPException(status_code=404, detail="Video no encontrado")

    zonas = preset[3]
    preset_nombre = preset[1]

    return StreamingResponse(
        generar_stream_video(
            ruta_video,
            nombre_original or nombre_video,
            zonas=zonas,
            preset_id=preset_id,
            preset_nombre=preset_nombre
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/analisis")
def listar_analisis_endpoint(
    _: dict = Depends(requerir_auth)   # ← protegido
):
    datos = listar_analisis()
    return {
        "analisis": [
            {
                "id": fila[0],
                "nombre_video": fila[1],
                "personas_maximas": fila[2],
                "grupo_mayor_maximo": fila[3],
                "nivel_final": fila[4],
                "fecha": fila[5].isoformat() if fila[5] else None,
                "preset_nombre": fila[6]
            }
            for fila in datos
        ]
    }