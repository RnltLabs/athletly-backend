FROM python:3.12-slim

# Install uv — fast Python package manager from Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first so Docker can cache the install layer.
COPY pyproject.toml uv.lock* ./

# Install production dependencies only.
# --no-dev: skip dev/test dependencies
# --frozen: treat the lockfile as authoritative (fail if out of sync)
RUN uv sync --no-dev --frozen

# Copy application source.
COPY src/ src/

# Expose the FastAPI port.
EXPOSE 8000

# Run the FastAPI app via uvicorn.
# host 0.0.0.0 binds to all interfaces inside the container.
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
