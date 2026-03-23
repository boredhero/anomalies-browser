FROM pdal/pdal:latest

# pdal/pdal already has: PDAL 2.10, GDAL 3.12, Python 3.13, Ubuntu 24.04
# Just need: pip/uv, libspatialindex (for rtree), and our Python deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python deps
COPY pyproject.toml uv.lock ./
COPY README.md ./
RUN uv sync --frozen --no-dev --no-editable

# Force WhiteboxTools binary download (the whitebox Python package downloads
# the WBT Rust binary on first import — do it at build time, not runtime)
RUN uv run python -c "import whitebox; wbt = whitebox.WhiteboxTools(); print('WBT version:', wbt.version().split(chr(10))[0])"

# Copy source
COPY src/ src/
COPY configs/ configs/
COPY alembic/ alembic/
COPY alembic.ini ./
COPY scripts/ scripts/
COPY tests/fixtures/known_sites.json tests/fixtures/known_sites.json

# Copy built frontend (injected by CI)
COPY frontend/dist/ static/

# Version info
COPY info.yml ./

# Log directory
RUN mkdir -p /app/logs
VOLUME /app/logs

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "hole_finder.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
