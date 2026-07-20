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
    requerir_admin,
    crear_token_stream,
    verificar_token_stream,
)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, field_validator, model_validator
from pathlib import Path
from typing import List, Tuple
import shutil
import os
import re
import cv2
import uuid

from database import crear_usuario, obtener_usuario_por_email
from psycopg2.errors import UniqueViolation

from detector import generar_stream_video, STREAMS_ACTIVOS, STREAMS_CANCELADOS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # La autenticación es 100% por Bearer token, no por cookies, así que no
    # hace falta allow_credentials (y combinarlo con origin "*" es una
    # bandera típica de escáneres de seguridad sin beneficio real acá).
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CARPETA_VIDEOS = "videos_entrada"
CARPETA_FRAMES = "frames_referencia"
EXTENSIONES_VIDEO_VALIDAS = {".mp4", ".avi", ".mov", ".mkv"}

os.makedirs(CARPETA_VIDEOS, exist_ok=True)
os.makedirs(CARPETA_FRAMES, exist_ok=True)


EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
NOMBRE_MAX_LARGO = 80
BCRYPT_MAX_BYTES = 72  # bcrypt trunca (y passlib puede fallar) más allá de esto


class ZonasRequest(BaseModel):
    # Cada zona es (x1, y1, x2, y2). Tipado y validado acá porque detector.py
    # las desempaqueta como `for zx1, zy1, zx2, zy2 in zonas` sin chequeos:
    # una zona mal formada rompería el stream de video a mitad de análisis.
    zonas: List[Tuple[int, int, int, int]]
    # Umbrales de aglomeración propios de este pasillo (antes eran constantes
    # fijas en detector.py, iguales para todos los pasillos).
    umbral_medio: int = 4
    umbral_alto: int = 6

    @field_validator("zonas")
    @classmethod
    def validar_zonas(cls, zonas):
        for x1, y1, x2, y2 in zonas:
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                raise ValueError("Las coordenadas de una zona no pueden ser negativas")
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Cada zona debe cumplir x2 > x1 y y2 > y1")
        return zonas

    @field_validator("umbral_medio")
    @classmethod
    def validar_umbral_medio(cls, v):
        if v < 2:
            raise ValueError("El umbral medio debe ser al menos 2 personas")
        return v

    @model_validator(mode="after")
    def validar_orden_umbrales(self):
        if self.umbral_alto <= self.umbral_medio:
            raise ValueError("El umbral alto debe ser mayor que el umbral medio")
        return self


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


@app.get("/", tags=["inicio"])
def inicio():
    return {"mensaje": "API de detección de aglomeraciones funcionando"}

# ============================================================
# AUTENTICACIÓN
# ============================================================

@app.post("/auth/registro", tags=["auth"])
def registro(data: RegistroRequest):
    nombre = data.nombre.strip()
    email = data.email.strip().lower()

    if not nombre:
        raise HTTPException(400, "El nombre no puede estar vacío")
    if len(nombre) > NOMBRE_MAX_LARGO:
        raise HTTPException(400, f"El nombre no puede superar los {NOMBRE_MAX_LARGO} caracteres")
    if not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ' -]+", nombre):
        raise HTTPException(400, "El nombre solo puede contener letras y espacios")

    if not EMAIL_REGEX.match(email):
        raise HTTPException(400, "Email inválido")

    if len(data.password) < 6:
        raise HTTPException(400, "La contraseña debe tener al menos 6 caracteres")
    if len(data.password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise HTTPException(400, f"La contraseña no puede superar los {BCRYPT_MAX_BYTES} caracteres")

    password_hash = hashear_password(data.password)

    try:
        usuario = crear_usuario(nombre, email, password_hash)
    except UniqueViolation:
        raise HTTPException(409, "Ese email ya está registrado")
    except Exception as e:
        raise HTTPException(500, f"Error creando usuario: {str(e)}")

    user_id, nombre, email, rol = usuario
    token = crear_token(user_id, email, rol)

    return {
        "token": token,
        "usuario": {"id": user_id, "nombre": nombre, "email": email, "rol": rol}
    }


@app.post("/auth/login", tags=["auth"])
def login(data: LoginRequest):
    usuario = obtener_usuario_por_email(data.email.strip().lower())
    if not usuario:
        raise HTTPException(401, "Email o contraseña incorrectos")

    user_id, nombre, email, password_hash, rol = usuario

    if not verificar_password(data.password, password_hash):
        raise HTTPException(401, "Email o contraseña incorrectos")

    token = crear_token(user_id, email, rol)

    return {
        "token": token,
        "usuario": {"id": user_id, "nombre": nombre, "email": email, "rol": rol}
    }


@app.get("/auth/me", tags=["auth"])
def yo(payload: dict = Depends(requerir_auth)):
    """Endpoint para que el frontend valide si el token sigue siendo válido."""
    return {
        "id": int(payload["sub"]),
        "email": payload["email"],
        "rol": payload.get("rol", "vigilante"),
    }


# ============================================================
# PRESETS
# ============================================================

@app.post("/presets", tags=["presets"])
def crear_preset_endpoint(
    nombre: str = Form(...),
    file: UploadFile = File(...),   # ← ya viene como imagen (frame extraído en el navegador)
    payload: dict = Depends(requerir_admin)   # ← solo administrador (HU-3.1)
):
    nombre = nombre.strip()
    if not nombre:
        raise HTTPException(status_code=400, detail="El nombre del pasillo no puede estar vacío")
    if len(nombre) > NOMBRE_MAX_LARGO:
        raise HTTPException(status_code=400, detail=f"El nombre no puede superar los {NOMBRE_MAX_LARGO} caracteres")

    user_id = int(payload["sub"])
    preset_uuid = str(uuid.uuid4())
    nombre_frame = f"frame_{preset_uuid}.jpg"
    ruta_frame = os.path.join(CARPETA_FRAMES, nombre_frame)

    with open(ruta_frame, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # El navegador ya recorta el primer frame, así que acá no hace falta
    # tocar video: solo validamos que lo recibido sea una imagen legible.
    if cv2.imread(ruta_frame) is None:
        if os.path.exists(ruta_frame):
            os.remove(ruta_frame)
        raise HTTPException(status_code=400, detail="El archivo recibido no es una imagen válida")

    try:
        preset_id = crear_preset(nombre=nombre, frame_path=nombre_frame, zonas=[], user_id=user_id)
    except Exception as e:
        if os.path.exists(ruta_frame):
            os.remove(ruta_frame)
        raise HTTPException(status_code=400, detail=f"Error al crear preset (¿nombre duplicado?): {str(e)}")

    return {
        "id": preset_id,
        "nombre": nombre,
        "frame_url": f"/presets/{preset_id}/frame",
        "zonas": [],
        "umbral_medio": 4,
        "umbral_alto": 6,
    }


@app.get("/presets", tags=["presets"])
def listar_presets_endpoint(
    payload: dict = Depends(requerir_auth)   # ← cualquier vigilante logueado
):
    presets = listar_presets()
    return {
        "presets": [
            {
                "id": p[0],
                "nombre": p[1],
                "frame_url": f"/presets/{p[0]}/frame",
                "zonas": p[3],
                "fecha_creacion": p[4].isoformat() if p[4] else None,
                "umbral_medio": p[5],
                "umbral_alto": p[6],
            }
            for p in presets
        ]
    }


@app.get("/presets/{preset_id}", tags=["presets"])
def obtener_preset_endpoint(
    preset_id: int,
    payload: dict = Depends(requerir_auth)   # ← cualquier vigilante logueado
):
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    return {
        "id": preset[0],
        "nombre": preset[1],
        "frame_url": f"/presets/{preset_id}/frame",
        "zonas": preset[3],
        "fecha_creacion": preset[4].isoformat() if preset[4] else None,
        "umbral_medio": preset[5],
        "umbral_alto": preset[6],
    }


@app.put("/presets/{preset_id}/zonas", tags=["presets"])
def actualizar_zonas_endpoint(
    preset_id: int,
    data: ZonasRequest,
    payload: dict = Depends(requerir_admin)   # ← solo administrador (HU-3.1)
):
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    actualizar_zonas_preset(
        preset_id, data.zonas,
        umbral_medio=data.umbral_medio, umbral_alto=data.umbral_alto
    )
    return {
        "mensaje": "Zonas actualizadas",
        "zonas": data.zonas,
        "umbral_medio": data.umbral_medio,
        "umbral_alto": data.umbral_alto,
    }


@app.delete("/presets/{preset_id}", tags=["presets"])
def eliminar_preset_endpoint(
    preset_id: int,
    payload: dict = Depends(requerir_admin)   # ← solo administrador (HU-3.1)
):
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    ruta_frame = os.path.join(CARPETA_FRAMES, preset[2])
    if os.path.exists(ruta_frame):
        os.remove(ruta_frame)

    eliminar_preset(preset_id)
    return {"mensaje": "Preset eliminado"}


@app.get("/presets/{preset_id}/frame", tags=["presets"])
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

@app.post("/analisis", tags=["analisis"])
def iniciar_analisis(
    preset_id: int = Form(...),
    file: UploadFile = File(...),
    payload: dict = Depends(requerir_auth)   # ← protegido
):
    user_id = int(payload["sub"])
    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    nombre_seguro = Path(file.filename).name
    if Path(nombre_seguro).suffix.lower() not in EXTENSIONES_VIDEO_VALIDAS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de video no soportado. Usa: {', '.join(sorted(EXTENSIONES_VIDEO_VALIDAS))}"
        )

    nombre_unico = f"{uuid.uuid4()}_{nombre_seguro}"
    ruta_entrada = os.path.join(CARPETA_VIDEOS, nombre_unico)

    with open(ruta_entrada, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # El stream se consume como <img src>/stream, que no puede mandar el
    # header Authorization: usamos un token de corta duración atado a
    # este video, preset y usuario en vez de dejar la ruta sin protección.
    token_stream = crear_token_stream(nombre_unico, preset_id, user_id)

    return {
        "mensaje": "Video subido, iniciando análisis",
        "nombre_video": file.filename,
        "video_id": nombre_unico,
        "stream_url": f"/analisis/stream/{nombre_unico}?preset_id={preset_id}&nombre_original={file.filename}&token={token_stream}"
    }


@app.post("/analisis/stream/{nombre_video}/detener", tags=["analisis"])
def detener_stream_analisis(nombre_video: str, payload: dict = Depends(requerir_auth)):
    # Autenticado con Bearer normal (no con el token de un solo uso del
    # stream): esta llamada la hace el botón "Detener" del frontend, que
    # sí tiene el Bearer a mano, a diferencia del <img src>/stream.
    nombre_video = Path(nombre_video).name
    user_id = int(payload["sub"])

    dueño = STREAMS_ACTIVOS.get(nombre_video)
    if dueño is None:
        return {"mensaje": "El stream ya no está activo"}
    if dueño != user_id:
        raise HTTPException(status_code=403, detail="No autorizado para detener este análisis")

    STREAMS_CANCELADOS.add(nombre_video)
    return {"mensaje": "Solicitud de detener enviada"}


@app.get("/analisis/stream/{nombre_video}", tags=["analisis"])
def stream_analisis(nombre_video: str, preset_id: int, token: str, nombre_original: str = None):
    # nombre_video viene de la URL: se sanea para evitar path traversal.
    nombre_video = Path(nombre_video).name

    # No se puede usar requerir_auth (Bearer) porque esto se carga como
    # <img src>/stream, así que validamos el token de un solo uso emitido
    # por /analisis para este video+preset+usuario específicos.
    payload_stream = verificar_token_stream(token, nombre_video, preset_id)
    user_id = int(payload_stream["sub"])

    preset = obtener_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset no encontrado")

    ruta_video = os.path.join(CARPETA_VIDEOS, nombre_video)
    if not os.path.exists(ruta_video):
        raise HTTPException(status_code=404, detail="Video no encontrado")

    zonas = preset[3]
    preset_nombre = preset[1]
    umbral_medio = preset[5]
    umbral_alto = preset[6]

    return StreamingResponse(
        generar_stream_video(
            ruta_video,
            nombre_original or nombre_video,
            zonas=zonas,
            preset_id=preset_id,
            preset_nombre=preset_nombre,
            user_id=user_id,
            umbral_medio=umbral_medio,
            umbral_alto=umbral_alto,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/analisis", tags=["analisis"])
def listar_analisis_endpoint(
    payload: dict = Depends(requerir_auth)   # ← protegido
):
    datos = listar_analisis(user_id=int(payload["sub"]))
    return {
        "analisis": [
            {
                "id": fila[0],
                "nombre_video": fila[1],
                "personas_maximas": fila[2],
                "grupo_mayor_maximo": fila[3],
                "nivel_final": fila[4],
                "fecha": fila[5].isoformat() if fila[5] else None,
                "preset_nombre": fila[6],
                "zona_alerta": fila[7],
                "momento_alerta_seg": float(fila[8]) if fila[8] is not None else None,
            }
            for fila in datos
        ]
    }