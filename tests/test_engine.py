"""The pre-flight check for hailort_service, which needs no accelerator."""

from __future__ import annotations

import os
import shutil
import socket
import tempfile

import pytest

from hailo_detect.engine import check_service_socket, service_socket_path


def test_missing_socket_names_both_likely_causes(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        check_service_socket(str(tmp_path / "absent.sock"))

    message = str(excinfo.value)
    assert "hailort.service is not running" in message
    assert "hostPath mount" in message


def test_a_directory_in_place_of_the_socket_is_diagnosed(tmp_path):
    # What a hostPath without `type: Socket` leaves behind when the host path
    # does not exist -- the failure mode the manifest's type guards against.
    stand_in = tmp_path / "hailort_uds.sock"
    stand_in.mkdir()

    with pytest.raises(RuntimeError, match="not a socket"):
        check_service_socket(str(stand_in))


def test_a_real_socket_passes():
    # Bound under a short directory rather than pytest's tmp_path: AF_UNIX
    # paths are capped near 104 bytes, and macOS's per-test temp directories
    # are long enough to blow past that.
    directory = tempfile.mkdtemp(prefix="hd", dir="/tmp")
    path = os.path.join(directory, "s.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(path)
        check_service_socket(path)  # does not raise
    finally:
        server.close()
        shutil.rmtree(directory, ignore_errors=True)


def test_service_address_forms_resolve_to_the_same_socket():
    # The two spellings gRPC accepts both name one file; only `unix:/path`
    # actually connects, but the pre-flight check has to locate either.
    assert service_socket_path("unix:/tmp/hailort_uds.sock") == "/tmp/hailort_uds.sock"
    assert service_socket_path("unix:///tmp/hailort_uds.sock") == "/tmp/hailort_uds.sock"


def test_a_tcp_service_address_has_no_socket_to_check():
    assert service_socket_path("localhost:50051") is None


def test_a_tcp_address_skips_the_preflight_rather_than_failing(monkeypatch):
    # Nothing local to stat, so the check must pass rather than invent a
    # failure about a socket that was never meant to exist.
    monkeypatch.setenv("HAILORT_SERVICE_ADDRESS", "1.2.3.4:9999")

    check_service_socket()  # does not raise
