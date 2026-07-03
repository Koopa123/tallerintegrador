"""
Crea (o promueve a) una cuenta de administrador.

No hay forma de crear un administrador desde el formulario de registro
publico de la app (/auth/registro siempre crea rol "vigilante") a
proposito: la unica forma de tener un admin es correr este script a
mano, con acceso directo a la base de datos.

Uso:
    python crear_admin.py
    (pide nombre, email y contrasena de forma interactiva)
"""
import getpass
import re

import psycopg2
from psycopg2.errors import UniqueViolation

from auth import hashear_password
from database import obtener_conexion

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def promover_a_admin(email):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute(
            "UPDATE usuarios SET rol = 'administrador' WHERE LOWER(email) = LOWER(%s) RETURNING id, nombre",
            (email,),
        )
        fila = cursor.fetchone()
        conexion.commit()
        return fila
    finally:
        cursor.close()
        conexion.close()


def crear_admin(nombre, email, password):
    password_hash = hashear_password(password)
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("""
            INSERT INTO usuarios (nombre, email, password_hash, rol)
            VALUES (%s, %s, %s, 'administrador')
            RETURNING id
        """, (nombre, email.strip().lower(), password_hash))
        admin_id = cursor.fetchone()[0]
        conexion.commit()
        return admin_id
    finally:
        cursor.close()
        conexion.close()


def main():
    print("=== Crear cuenta de administrador ===\n")
    email = input("Email: ").strip().lower()
    if not EMAIL_REGEX.match(email):
        print("Email invalido.")
        return

    # Si el email ya existe como usuario normal, ofrecer promoverlo en
    # vez de fallar por email duplicado.
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    try:
        cursor.execute("SELECT id, rol FROM usuarios WHERE LOWER(email) = LOWER(%s)", (email,))
        existente = cursor.fetchone()
    finally:
        cursor.close()
        conexion.close()

    if existente:
        _, rol_actual = existente
        if rol_actual == "administrador":
            print(f"'{email}' ya es administrador. No hay nada que hacer.")
            return
        respuesta = input(
            f"'{email}' ya existe como '{rol_actual}'. ¿Promoverlo a administrador? (s/n): "
        ).strip().lower()
        if respuesta != "s":
            print("Cancelado.")
            return
        fila = promover_a_admin(email)
        print(f"Listo: {fila[1]} (id={fila[0]}) ahora es administrador.")
        return

    nombre = input("Nombre: ").strip()
    if not nombre:
        print("El nombre no puede estar vacío.")
        return

    password = getpass.getpass("Contraseña (mínimo 6 caracteres): ")
    if len(password) < 6:
        print("La contraseña debe tener al menos 6 caracteres.")
        return
    confirmacion = getpass.getpass("Confirmar contraseña: ")
    if password != confirmacion:
        print("Las contraseñas no coinciden.")
        return

    try:
        admin_id = crear_admin(nombre, email, password)
    except UniqueViolation:
        print("Ese email ya está registrado (carrera con otro proceso). Volvé a correr el script.")
        return
    except psycopg2.Error as e:
        print(f"Error de base de datos: {e}")
        return

    print(f"\nAdministrador creado: id={admin_id}, email={email}")
    print("Ya puede iniciar sesión normalmente desde la app con estas credenciales.")


if __name__ == "__main__":
    main()
