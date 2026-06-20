FROM python:3.13-slim AS deps

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY src/ src/
COPY tests/ tests/

# FastStream is not an HTTP server, so there is no port to probe.
# Liveness is signalled by the "agent up ..." log line on startup and the
# container restart policy in docker-compose.yml.
CMD ["python", "-m", "src.main"]
