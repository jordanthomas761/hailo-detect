"""Entry point: `python -m hailo_detect`, or the `hailo-detect` console script."""

from __future__ import annotations

import logging
import os

import uvicorn

from .app import create_app
from .config import Settings


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        # The gateway and the Service both sit in front of this; uvicorn's own
        # access log would duplicate what NGINX already records.
        access_log=False,
        # One worker, always: the process owns a single accelerator, and a
        # second worker would fight it for the device.
        workers=1,
    )


if __name__ == "__main__":
    main()
