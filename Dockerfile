# DictForge — reproducible CPU benchmark image.
FROM python:3.13-slim

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY dictforge ./dictforge
COPY tests ./tests

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

# Run the invariant self-test and the deterministic benchmark.
CMD ["sh", "-c", "python -m dictforge.cli selftest && python -m dictforge.cli demo --out benchmark.json"]
