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

# ripgrep backs run_kb_command's `rg`; without it the model falls back to find+cat.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install uv into /usr/local/bin (on PATH for every user, including HF's uid 1000).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /root/.local/bin/uvx /usr/local/bin/ \
    && rm -rf /root/.local /root/.cache

RUN useradd -m -u 1000 user
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY . .
COPY --from=frontend /frontend/out ./frontend/out

ENV HOME=/home/user \
    PORT=7860
RUN chown -R user:user /app
USER user

EXPOSE 7860
CMD ["uv", "run", "-m", "scripts.api"]
