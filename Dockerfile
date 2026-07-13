FROM python:3.10.4-slim

# Dependencias del sistema para OpenCV y YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear carpetas necesarias
RUN mkdir -p videos_entrada videos_salida frames_referencia

EXPOSE 8000

# Forma shell (no exec/JSON) a propósito: Railway inyecta el puerto real
# en la variable $PORT en tiempo de ejecución, y solo la forma shell
# expande variables de entorno en el CMD. El fallback a 8000 es para
# correr el contenedor en local sin definir PORT.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}