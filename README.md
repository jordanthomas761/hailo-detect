# hailo-detect

Inference service for the Hailo-8 accelerator on the homelab cluster. Serves a
small web UI, reachable on the LAN only.

This repo holds **the application and its image**. It does not hold any
Kubernetes manifests — those live in `homelab-infra` under `apps/hailo-detect/`,
because `root-app.yaml` there globs `apps/*/application.yaml` in that repo only.
The seam between the two is the image tag in the internal registry.

```
hailo-detect (here)          homelab-infra
  src/ + Dockerfile   -->   apps/hailo-detect/manifests/deployment.yaml
        |                            (image: registry.jordanthomas.site/hailo-detect:<sha>)
        +--> build on ARM64 ARC runner
        +--> push to zot at registry.jordanthomas.site
```

## Target hardware

`ml.k8s.internal` (192.168.30.101), a Raspberry Pi 5 / 8GB with an
**M.2 Hailo-8** — the 26 TOPS part, not the 8L:

```
0001:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01)
Device Architecture: HAILO8
Firmware Version: 4.23.0
```

The node carries a `workload-type: ml` label; that is the `nodeSelector`.

## Things that will bite

**HailoRT must match the host driver exactly.** The host runs
`hailort-pcie-driver 4.23.0`, so the image must install `hailort=4.23.0` and
`python3-hailort=4.23.0-1` from `http://archive.raspberrypi.com/debian trixie`.
A mismatch fails at device-open with a version error, not at build time. Pin
both, and bump them in the same commit as the host.

**No `privileged` needed, and don't ask for it.** `/dev/hailo0` is mode 666, so
mounting the device node is enough. The cluster's `policy/workloads.rego`
rejects a privileged container whose name isn't in its allowlist, so asking for
it fails CI over there rather than failing at runtime here.

**The Raspberry Pi archive key is SHA-1, and trixie's apt rejects it.**
Debian 13 verifies repository signatures with Sequoia's `sqv` rather than
`gpgv`, and `sqv`'s default policy stopped accepting SHA-1 on 2026-02-01. The
archive key's own binding signature predates that and has not been re-signed,
so `apt-get update` reports the repo as *"not signed"* and the image build
fails outright. Nothing about the archive changed — the verifier did. The
Dockerfile derives `/etc/crypto-policies/back-ends/apt-sequoia.config` from
apt's default and re-allows SHA-1 only where second pre-image resistance is
sufficient (binding a key to its certificate), leaving it rejected where
collision resistance is what matters (the signature over `Release`). The date
is finite on purpose: the real fix is upstream re-signing, and a permanent
exemption would outlive it silently.

**`hailort.service` is active on the host.** Whether it holds the device in a
way that blocks a container opening `/dev/hailo0` directly is untested — first
thing to check if device-open fails.

**No camera is attached.** `rpicam-hello --list-cameras` reports none; the
`/dev/video*` nodes are the Pi's codec units, not a sensor. So input is uploaded
media or a pulled RTSP stream until a Camera Module goes in.

## Models

Precompiled HEFs ship in the `hailo-models` package at `/usr/share/hailo-models/`.
Use the `_h8` variants — `_h8l` is Hailo-8L and `_h10` is Hailo-10:

    yolov8s_h8.hef  yolov6n_h8.hef  yolov8s_pose_h8.hef  yolov5n_seg_h8.hef

## What it does

A FastAPI service with a small web UI at `/`:

- **Upload an image** — `POST /api/detect` returns JSON boxes,
  `POST /api/detect/annotated` returns the same image with the boxes drawn on.
- **Watch an RTSP source** — set `RTSP_URL` and the service pulls the stream
  with ffmpeg, detects continuously, and serves an annotated MJPEG preview at
  `/api/stream/mjpeg`.
- `/healthz`, `/readyz` and `/api/status` for the cluster and for you.

```
src/hailo_detect/
  engine.py       the device, owned by one thread
  postprocess.py  HailoRT's NMS output -> detections
  imaging.py      letterbox, map boxes back, draw, encode
  stream.py       ffmpeg -> latest-frame-wins -> engine
  app.py          routes
  static/         the UI
```

### One thread owns the device

There is one accelerator and HailoRT's Python objects are not thread-safe, so
every caller — an HTTP upload or a stream frame — posts onto a queue that a
single worker thread drains. The device is opened and the network group
configured **once**, at startup, and the vstream pipeline stays alive for the
life of the process.

That has two visible consequences, both deliberate:

- The queue is short (`INFER_QUEUE_SIZE`, default 8). Past that, uploads get a
  `503` with `Retry-After` rather than a slowly growing latency.
- The RTSP path drops frames instead of queueing them. A reader thread pulls
  from ffmpeg into a one-slot buffer that overwrites; the detect thread takes
  whatever is in it. On a live camera an old frame has no value.

### The device failing is not a crash

If the device cannot be opened — wrong HailoRT version, missing HEF, something
else holding `/dev/hailo0` — the process still starts, `/readyz` returns 503
and names the reason, and `/api/status` and the UI show it. A CrashLoopBackOff
would hide that message behind a restart count.

## Configuration

All environment, all optional.

| Variable | Default | |
|---|---|---|
| `HEF_PATH` | `/usr/share/hailo-models/yolov8s_h8.hef` | Also in the image: `yolov6n_h8.hef` |
| `SCORE_THRESHOLD` | `0.4` | |
| `MAX_DETECTIONS` | `50` | Per frame, highest scoring first |
| `PORT` | `8000` | |
| `MAX_UPLOAD_BYTES` | `16777216` | Refused before the body is buffered |
| `JPEG_QUALITY` | `80` | |
| `INFER_QUEUE_SIZE` | `8` | |
| `INFER_TIMEOUT_S` | `15` | |
| `RTSP_URL` | *unset* | Unset means the stream half stays dormant |
| `RTSP_TRANSPORT` | `tcp` | `tcp` or `udp` |
| `RTSP_FPS` | `5` | What ffmpeg is asked for, not what the device sustains |
| `LOG_LEVEL` | `INFO` | |

`RTSP_URL` usually carries credentials. Deliver it as a SealedSecret and load
it with `envFrom`, not as a literal in the Deployment — and note the service
redacts the userinfo before the URL reaches a log line or `/api/status`.

## Working on it without a Hailo

`hailo_platform` only exists inside the image, so `engine.py` imports it
softly and records the failure instead of raising. Everything else — the
letterbox maths, the NMS decode, the routes — runs and is tested on a laptop:

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python -m hailo_detect   # serves on :8000, /readyz red
```

The tests cover both container shapes HailoRT wraps its NMS output in, the
letterbox round trip, and each engine failure's HTTP status — with a stub in
place of the device.

## Build and push

CI does this on every push to `main` (`.github/workflows/ci.yml`): lint, test,
then build and push `registry.jordanthomas.site/hailo-detect:<short-sha>`. No
`:latest` and no branch tag — `policy/workloads.rego` in homelab-infra rejects
both, and a floating tag leaves Argo nothing to roll back to.

The workflow needs two things that do not exist yet:

1. **A repo-scoped ARC scale set**, `apps/arc-runners-hailo-detect/` in
   homelab-infra, with `runnerScaleSetName: hailo-detect-runner` — that name
   is the `runs-on:` label here, and the two must be changed together. Copy
   `apps/arc-runners-homelab-infra/`, which already pins runners to worker
   nodes.
2. **`REGISTRY_USERNAME` / `REGISTRY_PASSWORD`** repo secrets — the zot
   htpasswd account from `apps/registry/README.md`. Pull is anonymous; only
   the push needs them.

Locally, on an arm64 machine:

```sh
docker build -t hailo-detect:dev .
```

The build refuses to run on amd64. `hailort` has no amd64 candidate, and
without the check that surfaces 200MB later as "no installation candidate".

## Deploying it

The manifests live in **homelab-infra**, under `apps/hailo-detect/`, because
`root-app.yaml` globs `apps/*/application.yaml` in that repo. What that side
needs:

- `manifests/namespace.yaml` with `gateway-access: "true"`, or the HTTPRoute
  is rejected by the Gateway with "Not Permitted".
- A Deployment with `nodeSelector: workload-type: ml`, one replica, and the
  device node mounted. The Hailo-8 is an M.2 card in one Pi, so this is what
  puts the pod on the node that has it — confirmed present on
  `ml.k8s.internal` on 2026-08-23, after the workers were re-joined under
  their current names (`inventory/host_vars/pi5-ml.yml` in bootstrap-device
  still documents applying it to `pi5-ml`, which no longer exists as a node
  object). The selector fails closed: a missing label leaves the pod Pending
  rather than landing it somewhere with no accelerator.

  ```yaml
  volumeMounts:
    - name: hailo
      mountPath: /dev/hailo0
  volumes:
    - name: hailo
      hostPath:
        path: /dev/hailo0
        type: CharDevice
  ```

  **No `privileged`.** `/dev/hailo0` is mode 666, the image runs as uid 10001,
  and `policy/workloads.rego` fails CI over there for a privileged container
  whose name is not in its allowlist.
- `readinessProbe` on `/readyz` and `livenessProbe` on `/healthz`. They are
  deliberately different: restarting the pod does not re-seat a PCIe card, so
  a device fault must not turn into a restart loop.
- One replica, `strategy: Recreate`. Two pods would fight over one
  accelerator, and a rolling update would briefly try exactly that.
- If you set `readOnlyRootFilesystem: true`, mount an `emptyDir` at `/tmp` —
  the image points `HAILORT_LOGGER_PATH` there.
- A Service, and an HTTPRoute with a `parentRef` to `nginx-gateway` in the
  `gateway` namespace under `*.jordanthomas.site`.

## Status

The application, the image and CI are here and the test suite passes. What has
**not** happened yet:

- **No inference has run on the Hailo-8.** The first CI build reached the
  Raspberry Pi archive and failed on its SHA-1 signing key (see above); the
  package pins themselves were checked against the archive index —
  `hailort=4.23.0`, `python3-hailort=4.23.0-1`, and
  `yolov8s_h8.hef`/`yolov6n_h8.hef` in `hailo-models`.
- **The NMS output format is decoded, not confirmed.** `postprocess.py`
  handles both containers HailoRT is known to use and raises with the shape it
  actually saw for anything else, rather than quietly returning no detections.
  The first real inference is what settles it.
- **`hailort.service` on the host is still untested** against a container
  opening `/dev/hailo0`. If device-open fails, `/readyz` will say so — check
  this first.
- The homelab-infra half does not exist: no `apps/hailo-detect/`, no
  `apps/arc-runners-hailo-detect/`, and no registry secrets on the repo.
