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

**Mounting `/dev/hailo0` into a pod does not work, and `privileged` is not
the answer.** The device node is mode 666, so this *looks* like it should be
enough — and under `docker --device` it is. Under Kubernetes it is not: a
`hostPath` mount hands the container the device inode but adds no
device-cgroup rule, and `open()` returns **`EPERM`** (not `EACCES` — the mode
bits were never the problem). Measured from inside the pod:

    node    : 0o666  chardev=True  major/minor=237/0
    uid/gid : 10001 10001
    open    : FAILED errno=1 Operation not permitted

The two ways out are a device plugin, which is itself a privileged DaemonSet,
or going through `hailort_service` — which is what this service does. Keep it
that way: `policy/workloads.rego` rejects a privileged container whose name is
not in its allowlist, so asking for it fails CI over there rather than at
runtime here.

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

**`hailort.service` on the host is load-bearing, not an obstacle.** It was
suspected of holding the device and blocking a container; it does not —
`sudo fuser -v /dev/hailo0` shows nothing holding it, because the service only
opens the device once a client connects. It is HailoRT's multi-process
service, and this app is one of its clients: `multi_process_service = True`,
`group_id = "SHARED"`, talking to `/tmp/hailort_uds.sock` mounted in from the
host.

Two consequences. Several pods can share the one accelerator, which direct
device access cannot do. And the app now depends on a **systemd unit that
GitOps does not manage** — it is `enabled`, so it survives reboots, but a
rebuilt Pi without it leaves this app unable to start until it is back. That
belongs in `bootstrap-device` if this sticks around.

Using the service also forces the scheduler on (`ROUND_ROBIN`), which makes
`network_group.activate()` illegal. Reverting to direct device access means
restoring that call *and* solving the cgroup problem above.

**`HAILORT_SERVICE_ADDRESS` must be set, and its exact spelling decides
whether any of this works.** `multi_process_service = True` on its own is
ignored — libhailort goes to `/dev/hailo0` anyway. The address is handed
straight to gRPC as a channel target, and the three spellings a human would
call equivalent are not:

| value | result |
|---|---|
| `unix:/tmp/hailort_uds.sock` | connects |
| `unix:///tmp/hailort_uds.sock` | silently falls back to opening `/dev/hailo0` |
| `/tmp/hailort_uds.sock` | reaches gRPC, fails `UNAVAILABLE` (code 14) |

The Dockerfile sets the working form. The failure mode of the second is the
nasty one: it looks exactly like the service not being configured at all.

**A USB camera is attached, and it takes `/dev/video0`.** No CSI Camera
Module — `v4l2-ctl --list-devices` reports a UVC device on `/dev/video0`,
`/dev/video1` (the metadata node UVC devices expose alongside capture) and
`/dev/media3`. So the older note that the low-numbered `/dev/video*` nodes are
the Pi's codec units does not hold while a USB camera is plugged in.

**Node numbers are recycled, so do not hardcode them.** Two different devices
— a Logitech MX Brio and an HDMI capture dongle — both enumerated as exactly
`/dev/video0`, `/dev/video1`, `/dev/media3`. Reference a
`/dev/v4l/by-id/usb-...-video-index0` path instead, which is stable across
replug and unambiguous about which device you meant.

**A container cannot read it, for the same reason it cannot read
`/dev/hailo0`.** It is a device node, so `hostPath` gives `EPERM` from the
device cgroup. The pattern that worked for the accelerator applies unchanged:
the host owns the hardware and publishes it, the container consumes a URL.
`ustreamer` (MJPEG over HTTP) is the simpler publisher for a UVC device;
MediaMTX with an ffmpeg `v4l2` input gives you RTSP. Pass
`-input_format mjpeg` either way — otherwise ffmpeg negotiates raw YUYV and
USB bandwidth caps you at a few frames a second — and pick a resolution
explicitly, since a 4K camera will otherwise hand you far more pixels than a
640x640 model wants.

## Models

Precompiled HEFs ship in the `hailo-models` package at `/usr/share/hailo-models/`.
Use the `_h8` variants — `_h8l` is Hailo-8L and `_h10` is Hailo-10:

    yolov8s_h8.hef  yolov6n_h8.hef  yolov8s_pose_h8.hef  yolov5n_seg_h8.hef

## What it does

A FastAPI service with a small web UI at `/`:

- **Upload an image** — `POST /api/detect` returns JSON boxes,
  `POST /api/detect/annotated` returns the same image with the boxes drawn on.
- **Watch a stream** — set `STREAM_URL` to anything ffmpeg can read (`rtsp://`,
  an `http://` MJPEG stream, a file) and the service pulls it, detects
  continuously, and serves an annotated MJPEG preview at `/api/stream/mjpeg`.
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
| `STREAM_URL` | *unset* | Any URL ffmpeg can read. Unset means the stream half stays dormant |
| `RTSP_TRANSPORT` | `tcp` | `tcp` or `udp`, and passed only for `rtsp://` URLs |
| `STREAM_FPS` | `5` | What ffmpeg is asked for, not what the source sustains |
| `LOG_LEVEL` | `INFO` | |

`STREAM_URL` is not RTSP-specific. The RTSP-only ffmpeg flags are passed only
for `rtsp://` URLs, because ffmpeg exits on an input option the demuxer does
not recognise rather than ignoring it — which is what made this RTSP-only
before. `RTSP_URL` and `RTSP_FPS` are still accepted as the older names.

If the URL carries credentials, deliver it as a SealedSecret and load it with
`envFrom` rather than as a literal in the Deployment — and note the service
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

**It runs on the hardware.** Measured in-cluster, from an unprivileged pod with
no device node mounted, talking to `hailort_service`:

    model      : yolov8s input 640 x 640
    inference  : 25.4 ms
    raw type   : list
    decode_nms : OK

That settles two things that were assumptions for most of this repo's life:
the accelerator is reachable this way at all, and the NMS output arrives in
the ragged per-class `list` form `postprocess.py` handles.

Still open:

- **No detection has been run on a real image.** The inference above was on
  random noise, which correctly produced nothing. Boxes on a real photo, and
  whether the labels line up with COCO's class order, are unverified.
- **The stream path has never run.** Nothing has fed it a URL, so the reader
  loop, the reconnect backoff and the MJPEG endpoint are all unexercised. A
  camera is attached now, but nothing publishes it yet.
- **Nothing is measured under load.** The queue depth, the timeout and the
  memory request are all estimates; 25.4 ms for one frame is not a throughput
  figure.
