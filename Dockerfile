# syntax=docker/dockerfile:1.6

# Stage 1: build the Next.js frontend as a static export
FROM node:20-alpine AS frontend
WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Empty base URL => API calls are same-origin (served by FastAPI)
ENV NEXT_PUBLIC_AI_TUTOR_API_BASE_URL=""
RUN npm run build


# Stage 2: Python runtime + FastAPI + bundled static frontend
FROM python:3.13

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN useradd -m -u 1000 user
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY . .
COPY --from=frontend /frontend/out ./frontend/out

ENV HOME=/home/user \
    PATH="/root/.local/bin:/home/user/.local/bin:${PATH}" \
    PORT=7860
RUN chown -R user:user /app
USER user

EXPOSE 7860
CMD ["sh", "-c", "uv run uvicorn scripts.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
