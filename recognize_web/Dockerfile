FROM python:3.11-slim-bookworm

ARG PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best.pt \
    UPLOAD_DIR=/app/web_data/uploads \
    RESULT_DIR=/app/web_data/results \
    YOLO_DEVICE=cpu \
    PORT=3005

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/cpu.txt requirements/base.txt requirements/web.txt ./requirements/

# Install CPU PyTorch separately so the CUDA/WSL setup is not pulled into this image.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    -r requirements/cpu.txt \
    && python -m pip install --no-cache-dir \
    --index-url "${PYPI_MIRROR}" \
    -r requirements/base.txt \
    -r requirements/web.txt

COPY recognize_web ./recognize_web

RUN mkdir -p /app/models /app/web_data/uploads /app/web_data/results

EXPOSE 3005

CMD ["sh", "-c", "exec python -m uvicorn recognize_web.app.main:app --host 0.0.0.0 --port ${PORT}"]
