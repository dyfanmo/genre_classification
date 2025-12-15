FROM python:3.10-slim

WORKDIR /app

COPY requirements-vertex.txt .
RUN pip install --no-cache-dir -r requirements-vertex.txt

COPY src/ src/
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.vertex_train"]