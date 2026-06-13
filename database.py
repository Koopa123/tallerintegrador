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

def guardar_analisis(nombre_video, personas_maximas, grupo_mayor_maximo, nivel_final, preset_id=None):
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("""
        INSERT INTO analisis (
            nombre_video,
            personas_maximas,
            grupo_mayor_maximo,
            nivel_final,
            preset_id
        )
        VALUES (%s, %s, %s, %s, %s)
    """, (
        nombre_video,
        personas_maximas,
        grupo_mayor_maximo,
        nivel_final,
        preset_id
    ))

    conexion.commit()
    cursor.close()
    conexion.close()


def listar_analisis():
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT a.id, a.nombre_video, a.personas_maximas, a.grupo_mayor_maximo,
               a.nivel_final, a.fecha, p.nombre
        FROM analisis a
        LEFT JOIN presets p ON a.preset_id = p.id
        ORDER BY a.fecha DESC
    """)

    datos = cursor.fetchall()
    cursor.close()
    conexion.close()
    return datos


# ============================================================
# PRESETS DE ZONAS
# ============================================================

def crear_preset(nombre, frame_path, zonas):
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("""
        INSERT INTO presets (nombre, frame_path, zonas)
        VALUES (%s, %s, %s)
        RETURNING id
    """, (nombre, frame_path, json.dumps(zonas)))

    preset_id = cursor.fetchone()[0]
    conexion.commit()
    cursor.close()
    conexion.close()
    return preset_id


def listar_presets():
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT id, nombre, frame_path, zonas, fecha_creacion
        FROM presets
        ORDER BY fecha_creacion DESC
    """)

    datos = cursor.fetchall()
    cursor.close()
    conexion.close()
    return datos


def obtener_preset(preset_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT id, nombre, frame_path, zonas, fecha_creacion
        FROM presets
        WHERE id = %s
    """, (preset_id,))

    fila = cursor.fetchone()
    cursor.close()
    conexion.close()
    return fila


def actualizar_zonas_preset(preset_id, zonas):
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("""
        UPDATE presets
        SET zonas = %s
        WHERE id = %s
    """, (json.dumps(zonas), preset_id))

    conexion.commit()
    cursor.close()
    conexion.close()


def eliminar_preset(preset_id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("DELETE FROM presets WHERE id = %s", (preset_id,))
    conexion.commit()
    cursor.close()
    conexion.close()