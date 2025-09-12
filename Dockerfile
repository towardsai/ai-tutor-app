FROM python:3.13

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# uv was installed to /root/.local/bin – add it to PATH
ENV PATH="/root/.local/bin:${PATH}"

RUN useradd -m -u 1000 user
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY . .


ENV HOME=/home/user PATH="/root/.local/bin:/home/user/.local/bin:${PATH}"
RUN chown -R user:user /app
USER user

EXPOSE 7860
CMD ["uv", "run", "scripts/main.py"]
