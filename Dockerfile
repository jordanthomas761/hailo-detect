# syntax=docker/dockerfile:1

# HailoRT must match the host's PCIe driver exactly. ml.k8s.internal runs
# hailort-pcie-driver 4.23.0, so both of these are pinned and must be bumped
# in the same change as the host -- a mismatch fails at device-open with a
# version error, not at build time.
ARG HAILORT_VERSION=4.23.0
ARG PYHAILORT_VERSION=4.23.0-1

# The HEFs come from the Raspberry Pi archive's hailo-models package rather
# than the repo: they are tens of MB each and are gitignored.
ARG HEF_FILES="yolov8s_h8.hef yolov6n_h8.hef"


# ---------------------------------------------------------------------------
# Common base: Debian trixie with the Raspberry Pi archive wired up, which is
# where every hailo package lives.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS hailo-apt

# hailort has no amd64 candidate, and the failure otherwise arrives 200MB
# later as an unhelpful "has no installation candidate". Every node in this
# cluster is arm64, and so is the ARC runner that builds this.
RUN test "$(dpkg --print-architecture)" = "arm64" \
    || { echo "hailo-detect builds arm64 only (got $(dpkg --print-architecture))" >&2; exit 1; }

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
       | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg]" \
            "http://archive.raspberrypi.com/debian trixie main" \
       > /etc/apt/sources.list.d/raspi.list


# ---------------------------------------------------------------------------
# Models: install the whole hailo-models package, ship only the HEFs we use.
# ---------------------------------------------------------------------------
FROM hailo-apt AS models
ARG HEF_FILES

# ~326MB installed: every model, for all three Hailo architectures.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends hailo-models

# Only the _h8 variants are usable here: _h8l is Hailo-8L and _h10 is
# Hailo-10, and loading either fails at configure with an architecture
# mismatch. Copying a subset rather than the directory keeps the image from
# carrying three architectures' worth of weights.
RUN mkdir -p /models \
    && for hef in ${HEF_FILES}; do cp "/usr/share/hailo-models/${hef}" /models/; done \
    && ls -la /models


# ---------------------------------------------------------------------------
# Python dependencies, built into a venv we then copy wholesale.
# ---------------------------------------------------------------------------
FROM hailo-apt AS builder

# python3-numpy is here on purpose, and it is not just a build dependency:
# python3-hailort is compiled against Debian's numpy. Installing it before the
# venv means pip sees the requirement already satisfied through
# --system-site-packages and leaves it alone, so the runtime ends up with the
# one numpy those bindings were built against instead of a PyPI build layered
# over it.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-numpy build-essential

# --system-site-packages so the venv can see python3-hailort, which apt
# installs into /usr/lib/python3/dist-packages in the runtime stage. There is
# no PyPI distribution of the HailoRT bindings to install here instead.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile .


# ---------------------------------------------------------------------------
# Runtime.
# ---------------------------------------------------------------------------
FROM hailo-apt AS runtime
ARG HAILORT_VERSION
ARG PYHAILORT_VERSION

# Note what is *not* installed: hailo-all. That metapackage pulls
# hailort-pcie-driver, a DKMS package that would try to build a kernel module
# inside the image -- the driver belongs to the host -- along with tappas and
# the rpicam postprocess plugins, none of which this service uses. hailort and
# python3-hailort are the two packages that matter, so they are named
# directly.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        "hailort=${HAILORT_VERSION}" \
        "python3-hailort=${PYHAILORT_VERSION}" \
        python3 \
        ffmpeg \
        fonts-dejavu-core \
    && apt-mark hold hailort python3-hailort

COPY --from=builder /opt/venv /opt/venv
COPY --from=models /models /usr/share/hailo-models

# /dev/hailo0 is mode 666 on the host, so an unprivileged uid can open it.
# This image must never need `privileged` -- the cluster's
# policy/workloads.rego rejects that for a container not in its allowlist.
RUN useradd --system --uid 10001 --create-home --home-dir /home/app app
USER 10001:10001
WORKDIR /home/app

# HAILORT_LOGGER_PATH: HailoRT writes hailort.log into the process's working
# directory when it cannot reach its own log dir. Pointing it at /tmp keeps
# the image runnable with a read-only root filesystem and an emptyDir /tmp.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HAILORT_LOGGER_PATH=/tmp \
    HEF_PATH=/usr/share/hailo-models/yolov8s_h8.hef \
    PORT=8000

EXPOSE 8000
CMD ["hailo-detect"]
