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

def guardar_analisis(nombre_video, personas_maximas, grupo_mayor_maximo, nivel_final, user_id,
                      preset_id=None, zona_alerta=None, momento_alerta_seg=None):
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
                user_id,
                zona_alerta,
                momento_alerta_seg
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            nombre_video,
            personas_maximas,
            grupo_mayor_maximo,
            nivel_final,
            preset_id,
            user_id,
            zona_alerta,
            momento_alerta_seg
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
                   a.nivel_final, a.fecha, p.nombre, a.zona_alerta, a.momento_alerta_seg
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
#
# Los pasillos son un recurso COMPARTIDO, no privado por usuario: los
# configura el administrador (único rol que puede crear/editar/borrar,
# ver requerir_admin en auth.py) y cualquier vigilante logueado los usa
# para monitorear. user_id se sigue guardando en la fila como metadato
# de quién lo creó, pero ya no se usa para filtrar el acceso.
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


def listar_presets():
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            SELECT id, nombre, frame_path, zonas, fecha_creacion, umbral_medio, umbral_alto
            FROM presets
            ORDER BY fecha_creacion DESC
        """)
        return cursor.fetchall()
    finally:
        cursor.close()
        conexion.close()


def obtener_preset(preset_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            SELECT id, nombre, frame_path, zonas, fecha_creacion, umbral_medio, umbral_alto
            FROM presets
            WHERE id = %s
        """, (preset_id,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conexion.close()


def actualizar_zonas_preset(preset_id, zonas, umbral_medio, umbral_alto):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            UPDATE presets
            SET zonas = %s, umbral_medio = %s, umbral_alto = %s
            WHERE id = %s
        """, (json.dumps(zonas), umbral_medio, umbral_alto, preset_id))
        conexion.commit()
    finally:
        cursor.close()
        conexion.close()


def eliminar_preset(preset_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("DELETE FROM presets WHERE id = %s", (preset_id,))
        conexion.commit()
    finally:
        cursor.close()
        conexion.close()

# ============================================================
# USUARIOS
# ============================================================

def crear_usuario(nombre, email, password_hash, rol="vigilante"):
    """
    rol por defecto es "vigilante": el registro público (/auth/registro)
    nunca pasa un rol distinto. La cuenta de administrador se crea aparte,
    con crear_admin.py, para que no sea posible autoasignarse el rol
    desde el formulario de registro normal.
    """
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            INSERT INTO usuarios (nombre, email, password_hash, rol)
            VALUES (%s, %s, %s, %s)
            RETURNING id, nombre, email, rol
        """, (nombre, email, password_hash, rol))
        usuario = cursor.fetchone()
        conexion.commit()
        return usuario  # (id, nombre, email, rol)
    finally:
        cursor.close()
        conexion.close()


def obtener_usuario_por_email(email):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        # Case-insensitive: cubre usuarios registrados antes de normalizar
        # el email a minúsculas en /auth/registro.
        cursor.execute("""
            SELECT id, nombre, email, password_hash, rol
            FROM usuarios
            WHERE LOWER(email) = LOWER(%s)
        """, (email,))
        return cursor.fetchone()  # (id, nombre, email, password_hash, rol) o None
    finally:
        cursor.close()
        conexion.close()
