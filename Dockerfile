FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[server]'
COPY configs ./configs
COPY data ./data
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
EXPOSE 8765
CMD ["looped-transformers", "serve", "--host", "0.0.0.0", "--configs", "configs", "--runs", "/runs"]
