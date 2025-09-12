# Dockerfile
FROM python:3.13-slim

# (Recommended) Install curl & certs on -slim base
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install uv (installs to /root/.local/bin by default)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Make sure uv is on PATH for all later steps
ENV PATH="/root/.local/bin:${PATH}"

# Create the HF-required user (UID 1000) before copying files
RUN useradd -m -u 1000 user

# Work in the user's home to avoid permissions issues
ENV HOME=/home/user \
    PATH="/root/.local/bin:/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

# Copy only deps first for better caching; make them owned by 'user'
COPY --chown=user:user pyproject.toml uv.lock ./

# Use the non-root user for env creation so .venv belongs to them
USER user
RUN uv sync --locked

# Copy the rest of your app
COPY --chown=user:user . .

# Spaces expects your app to listen on this port; keep README app_port in sync
EXPOSE 7860

# Start the app via uv
CMD ["uv", "run", "scripts/main.py"]
