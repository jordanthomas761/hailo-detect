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

## Status

Scaffold only — no application code yet.
