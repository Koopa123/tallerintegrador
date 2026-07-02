import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def obtener_conexion():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


# ============================================================
# ANÁLISIS
# ============================================================

def guardar_analisis(nombre_video, personas_maximas, grupo_mayor_maximo, nivel_final, user_id, preset_id=None):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            INSERT INTO analisis (
                nombre_video,
                personas_maximas,
                grupo_mayor_maximo,
                nivel_final,
                preset_id,
                user_id
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            nombre_video,
            personas_maximas,
            grupo_mayor_maximo,
            nivel_final,
            preset_id,
            user_id
        ))
        conexion.commit()
    finally:
        cursor.close()
        conexion.close()


def listar_analisis(user_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            SELECT a.id, a.nombre_video, a.personas_maximas, a.grupo_mayor_maximo,
                   a.nivel_final, a.fecha, p.nombre
            FROM analisis a
            LEFT JOIN presets p ON a.preset_id = p.id
            WHERE a.user_id = %s
            ORDER BY a.fecha DESC
        """, (user_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conexion.close()


# ============================================================
# PRESETS DE ZONAS
# ============================================================

def crear_preset(nombre, frame_path, zonas, user_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            INSERT INTO presets (nombre, frame_path, zonas, user_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (nombre, frame_path, json.dumps(zonas), user_id))

        preset_id = cursor.fetchone()[0]
        conexion.commit()
        return preset_id
    finally:
        cursor.close()
        conexion.close()


def listar_presets(user_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            SELECT id, nombre, frame_path, zonas, fecha_creacion
            FROM presets
            WHERE user_id = %s
            ORDER BY fecha_creacion DESC
        """, (user_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conexion.close()


def obtener_preset(preset_id, user_id):
    """Devuelve el preset solo si pertenece a user_id (para rutas autenticadas)."""
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            SELECT id, nombre, frame_path, zonas, fecha_creacion
            FROM presets
            WHERE id = %s AND user_id = %s
        """, (preset_id, user_id))
        return cursor.fetchone()
    finally:
        cursor.close()
        conexion.close()


def obtener_preset_publico(preset_id):
    """
    Devuelve el preset sin filtrar por dueño. Uso exclusivo del endpoint
    de frame (/presets/{id}/frame), que se carga como <img src> y por lo
    tanto no puede mandar el header Authorization.
    """
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            SELECT id, nombre, frame_path, zonas, fecha_creacion
            FROM presets
            WHERE id = %s
        """, (preset_id,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conexion.close()


def actualizar_zonas_preset(preset_id, zonas, user_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            UPDATE presets
            SET zonas = %s
            WHERE id = %s AND user_id = %s
        """, (json.dumps(zonas), preset_id, user_id))
        conexion.commit()
    finally:
        cursor.close()
        conexion.close()


def eliminar_preset(preset_id, user_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("DELETE FROM presets WHERE id = %s AND user_id = %s", (preset_id, user_id))
        conexion.commit()
    finally:
        cursor.close()
        conexion.close()

# ============================================================
# USUARIOS
# ============================================================

def crear_usuario(nombre, email, password_hash):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            INSERT INTO usuarios (nombre, email, password_hash)
            VALUES (%s, %s, %s)
            RETURNING id, nombre, email
        """, (nombre, email, password_hash))
        usuario = cursor.fetchone()
        conexion.commit()
        return usuario  # (id, nombre, email)
    finally:
        cursor.close()
        conexion.close()


def obtener_usuario_por_email(email):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            SELECT id, nombre, email, password_hash
            FROM usuarios
            WHERE email = %s
        """, (email,))
        return cursor.fetchone()  # (id, nombre, email, password_hash) o None
    finally:
        cursor.close()
        conexion.close()
