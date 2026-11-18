"""Keep the real-cell kit's tests off the host's cell: a private lock file and quiet fake buses."""

from __future__ import annotations

import pytest

from tests.support.cell.fake import FakeAdapter, make_cell, redirect_lock


@pytest.fixture(autouse=True)
def cell_lock_path(tmp_path, monkeypatch):
    """The cell lock of every test here is a file under its own ``tmp_path``, never ``/tmp/ora-cell.lock``."""

    return redirect_lock(monkeypatch, tmp_path / "ora-cell.lock")


@pytest.fixture
def quiet_buses(monkeypatch):
    """No other reader and no traffic on any bus: the kit's bus checks open no CAN socket."""

    from ora.cell import occupancy

    monkeypatch.setattr(occupancy, "can_receivers", lambda channel: 0)
    monkeypatch.setattr(occupancy, "can_traffic", lambda channel, seconds=0.3: 0)


@pytest.fixture
def adapter():
    return FakeAdapter()


@pytest.fixture
def cell(tmp_path, adapter, quiet_buses):
    """A cell on the fake adapter, its rest pose recorded at zero."""

    return make_cell(tmp_path, adapter)
