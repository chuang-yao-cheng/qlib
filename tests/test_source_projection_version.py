from __future__ import annotations

import builtins

import qlib


def test_qlib_version_falls_back_to_installed_distribution(monkeypatch) -> None:
    real_import = builtins.__import__

    def missing_generated_version(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "_version":
            raise ImportError("missing generated source projection version")
        return real_import(name, globals, locals, fromlist, level)

    def missing_git_metadata(*args, **kwargs):
        raise LookupError("missing source control metadata")

    monkeypatch.setattr(builtins, "__import__", missing_generated_version)
    monkeypatch.setattr(qlib, "get_version", missing_git_metadata)
    monkeypatch.setattr(
        qlib,
        "installed_distribution_version",
        lambda distribution: "0.9.8.dev29",
    )

    assert qlib._resolve_qlib_version() == "0.9.8.dev29"
