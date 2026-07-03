import os
from datetime import datetime, timedelta

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Esquema de seguridad que Swagger SÍ reconoce
security_scheme = HTTPBearer()


def hashear_password(password: str) -> str:
    return pwd_context.hash(password)


def verificar_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def crear_token(user_id: int, email: str, rol: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "rol": rol,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verificar_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(401, "Token inválido o expirado")


def requerir_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
) -> dict:
    """
    Dependency de FastAPI: valida el token enviado como Bearer.
    Swagger muestra el botón Authorize gracias a HTTPBearer.
    """
    return verificar_token(credentials.credentials)


def requerir_admin(
    payload: dict = Depends(requerir_auth)
) -> dict:
    """
    Igual que requerir_auth, pero además exige rol "administrador" en el
    token. El rol viaja en el JWT (se fija al loguear/registrar), así que
    esto no necesita otra consulta a la base de datos.
    """
    if payload.get("rol") != "administrador":
        raise HTTPException(403, "Esta acción requiere una cuenta de administrador")
    return payload


STREAM_TOKEN_EXPIRE_MINUTOS = 5


def crear_token_stream(nombre_video: str, preset_id: int, user_id: int) -> str:
    """
    Token de corta duración para el endpoint /analisis/stream, que se
    consume como <img src>/stream y por lo tanto no puede mandar un
    header Authorization. Solo autoriza ESE video y ESE preset puntual.
    """
    payload = {
        "sub": str(user_id),
        "nombre_video": nombre_video,
        "preset_id": preset_id,
        "tipo": "stream",
        "exp": datetime.utcnow() + timedelta(minutes=STREAM_TOKEN_EXPIRE_MINUTOS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verificar_token_stream(token: str, nombre_video: str, preset_id: int) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(401, "Token de stream inválido o expirado")

    if payload.get("tipo") != "stream":
        raise HTTPException(401, "Token de stream inválido")

    if payload.get("nombre_video") != nombre_video or payload.get("preset_id") != preset_id:
        raise HTTPException(401, "Token de stream no corresponde a este recurso")

    return payload