FROM python:3.12-slim-bookworm AS builder

RUN pip install uv

ENV UV_CACHE_DIR=/root/.cache/uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME=/app/huggingface_cache

RUN uv venv $VIRTUAL_ENV

WORKDIR /app

COPY pyproject.toml .
RUN uv pip install --no-cache .
RUN uv pip install torch --no-cache --extra-index-url https://download.pytorch.org/whl/cpu

RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

FROM python:3.12-slim-bookworm AS production

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME="/app/huggingface_cache"

WORKDIR /app

RUN groupadd --system appuser && useradd --system --gid appuser appuser

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=builder /app/huggingface_cache/ /app/huggingface_cache/
COPY --chown=appuser:appuser ./app ./app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]