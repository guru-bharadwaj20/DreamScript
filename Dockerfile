# Phase 15.10 - the model server image.
#
# CUDA base, because the pinned torch is `2.5.1+cu124` and an image that cannot use the card is an
# image that answers /predict in minutes rather than seconds. The runtime image is `-runtime` and
# not `-devel`: nothing here compiles CUDA kernels, and devel is roughly three times the size.
#
# Two stages, and the split is for layer caching rather than size. Requirements change rarely and
# source changes constantly, so the dependency install is its own layer and a source edit does not
# reinstall torch.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgl1 and libglib2.0-0 are OpenCV's runtime shared objects. Without them `import cv2` fails at
# load with an error that names neither OpenCV nor the missing library, which is a memorable
# afternoon.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements/ requirements/

# Installed in layers that mirror requirements/, and torch from the CUDA index so the image gets
# the same `+cu124` build the pins were measured against. 15.12 checks that at runtime; getting it
# right here is what makes that check pass.
RUN pip install --upgrade pip \
    && pip install -r requirements/base.txt \
    && pip install -r requirements/torch.txt --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r requirements/classical.txt \
    && pip install -r requirements/cv.txt \
    && pip install -r requirements/serve.txt

# --- the served image -------------------------------------------------------------------------

FROM deps AS serve

WORKDIR /app
COPY src/ src/
COPY schemas/ schemas/
COPY pyproject.toml ./

# Not root. The service reads an upload and writes a temporary file, and needs nothing else.
RUN useradd --create-home --uid 10001 dreamscript \
    && chown -R dreamscript:dreamscript /app
USER dreamscript

# Weights and the DVC payload are mounted, never baked. A 1.3 GB checkpoint in an image layer
# makes every pull carry it and every retrain invalidate the layer beneath.
VOLUME ["/app/experiments", "/app/data", "/app/mlruns"]

EXPOSE 8000

# `/health` deliberately does not load the model (15.11), which is what makes it usable here:
# a healthcheck that pulled 1.3 GB of weights would fail its own start period and loop the
# container forever.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
