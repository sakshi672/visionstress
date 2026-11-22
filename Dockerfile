FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY src ./src
RUN pip install -e .
COPY alembic.ini ./
COPY migrations ./migrations
COPY configs ./configs
CMD ["uvicorn", "visionstress.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
