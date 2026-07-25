# Container for a Hugging Face Docker Space (or any plain Docker host).
#
# Why a Docker Space rather than the Gradio SDK: this app is FastAPI serving a
# static front end. The Gradio SDK expects a Gradio app object and would ignore
# server.py entirely.
#
# Build/run locally exactly as the Space will:
#   docker build -t pdb2print .
#   docker run --rm -p 7860:7860 pdb2print
FROM python:3.12-slim

# libGL and libglib are needed by trimesh/pymeshlab's binary wheels; without them
# the import fails at runtime with a bare "libGL.so.1: cannot open shared object
# file", which is an unhelpful way to discover a missing system package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as UID 1000. Creating that user here (and
# owning the app directory) keeps file permissions the same locally and on the
# Space, so "works in Docker, fails on the Space" cannot happen for that reason.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Requirements first, so a source edit does not invalidate the dependency layer.
COPY --chown=user requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt -r requirements-server.txt

COPY --chown=user . .

# The cache fills with whatever people actually build. The Space's disk resets
# when the container restarts, so the cache is warm for as long as the Space
# stays up and starts empty after a rebuild. That is fine: the structures people
# ask for repeatedly are the ones worth keeping, and nobody has to guess which
# those are in advance.
#
# Set PDB2PRINT_CACHE_RO=1 to make it read-only instead.

# Spaces route to 7860.
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
