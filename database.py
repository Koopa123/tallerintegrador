import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def obtener_conexion():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def guardar_analisis(nombre_video, personas_maximas, grupo_mayor_maximo, nivel_final):
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("""
        INSERT INTO analisis (
            nombre_video,
            personas_maximas,
            grupo_mayor_maximo,
            nivel_final
        )
        VALUES (%s, %s, %s, %s)
    """, (
        nombre_video,
        personas_maximas,
        grupo_mayor_maximo,
        nivel_final
    ))

    conexion.commit()
    cursor.close()
    conexion.close()


def listar_analisis():
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT id, nombre_video, personas_maximas, grupo_mayor_maximo, nivel_final, fecha
        FROM analisis
        ORDER BY fecha DESC
    """)

    datos = cursor.fetchall()

    cursor.close()
    conexion.close()

    return datos