FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-lc"]

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY server /app

RUN conda create -y -n osl python=3.12 && \
    conda run -n osl pip install --no-cache-dir /app && \
    conda clean -afy

RUN conda run -n osl python /app/scripts/generate_flat_configs.py && \
    chmod +x /app/scripts/serverctl /app/scripts/docker_entrypoint.sh

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
