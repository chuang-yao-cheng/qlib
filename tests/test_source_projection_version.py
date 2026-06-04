from __future__ import annotations

import qlib


def test_qlib_version_falls_back_to_installed_distribution(monkeypatch) -> None:
    def missing_git_metadata(*args, **kwargs):
        raise LookupError("missing source control metadata")

    monkeypatch.setattr(qlib, "get_version", missing_git_metadata)
    monkeypatch.setattr(
        qlib,
        "installed_distribution_version",
        lambda distribution: "0.9.8.dev29",
    )

    assert qlib._resolve_qlib_version() == "0.9.8.dev29"

