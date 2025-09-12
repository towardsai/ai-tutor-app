# Use a suitable base image
FROM python:3.13

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy the dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Use uv sync to install dependencies from the lock file
# The --locked flag ensures the lock file is not modified
RUN uv sync --locked

# Copy the rest of your application code
COPY . .

# Grant permissions and switch user
RUN chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the application port
EXPOSE 7860

# Set the command to run your script using `uv run`
CMD ["uv", "run", "scripts/main.py"]
