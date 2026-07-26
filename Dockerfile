# The pdb2print container: FastAPI serving the static front end in frontend/.
#
# In production it sits behind Caddy — see deploy/docker-compose.yml, which is
# what runs pdb2print.org. Standalone, for a local check:
#
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

# Run as an unprivileged user rather than root. UID 1000 specifically, because
# the cache is bind-mounted from the host in production — the host directory has
# to be chowned to the same UID or the app cannot write to it.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Requirements first, so a source edit does not invalidate the dependency layer.
COPY --chown=user requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt -r requirements-server.txt

COPY --chown=user . .

# The cache fills with whatever people actually build, so the structures asked
# for repeatedly become instant without anyone guessing which those are in
# advance. In production it is bind-mounted from the host and survives restarts.
#
# Set PDB2PRINT_CACHE_RO=1 to make it read-only — worth doing on a host with an
# ephemeral disk, where writes would be thrown away anyway.

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
