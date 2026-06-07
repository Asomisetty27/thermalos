# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY theta/ ./theta/

RUN pip install --upgrade pip --quiet \
 && pip install build --quiet \
 && python -m build --wheel --outdir /dist

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Theta"
LABEL org.opencontainers.image.description="GPU thermal-power forensics agent"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/Asomisetty27/theta"

# Non-root user
RUN useradd --create-home --shell /bin/bash theta

WORKDIR /app
COPY --from=builder /dist/*.whl .
RUN pip install --quiet *.whl && rm *.whl

# Config and log dirs (writable by theta user)
RUN mkdir -p /home/theta/.theta /var/log/theta \
 && chown -R theta:theta /home/theta/.theta /var/log/theta

USER theta

# Prometheus metrics
EXPOSE 9101

# Defaults — override via env vars or command args
ENV THETA_INTERVAL=5 \
    THETA_PROMETHEUS_PORT=9101 \
    THETA_LOG=/var/log/theta/alerts.jsonl

ENTRYPOINT ["theta"]
CMD ["monitor", \
     "--interval", "5", \
     "--port",     "9101", \
     "--log",      "/var/log/theta/alerts.jsonl"]
