# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import importlib


OPTIONAL_MODEL_IMPORT_FAILURES: dict[str, dict[str, str]] = {}
_OPTIONAL_MODEL_CACHE: dict[str, object | None] = {}
_OPTIONAL_MODEL_MESSAGES_EMITTED: set[str] = set()


def _record_optional_import_failure(group: str, exc: ImportError) -> None:
    OPTIONAL_MODEL_IMPORT_FAILURES[group] = {
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _emit_optional_import_message_once(group: str, message: str) -> None:
    if group in _OPTIONAL_MODEL_MESSAGES_EMITTED:
        return
    _OPTIONAL_MODEL_MESSAGES_EMITTED.add(group)
    print(message)


def _is_optional_pytorch_import_error(exc: ImportError) -> bool:
    missing_name = str(getattr(exc, "name", "") or "")
    if isinstance(exc, ModuleNotFoundError):
        return (
            not missing_name
            or missing_name == "torch"
            or missing_name.startswith("torch.")
            or missing_name.startswith("qlib.contrib.model.pytorch_")
        )

    message = str(exc)
    return (
        missing_name == "torch"
        or missing_name.startswith("torch.")
        or "libc10.so" in message
        or "static TLS block" in message
    )


_OPTIONAL_MODEL_EXPORTS = {
    "CatBoostModel": {
        "module": ".catboost_model",
        "attribute": "CatBoostModel",
        "group": "catboost",
        "failure_message": (
            "ModuleNotFoundError. CatBoostModel are skipped. "
            "(optional: maybe installing CatBoostModel can fix it.)"
        ),
    },
    "LinearModel": {
        "module": ".linear",
        "attribute": "LinearModel",
        "group": "linear",
        "failure_message": (
            "ModuleNotFoundError. LinearModel is skipped"
            "(optional: maybe installing scipy and sklearn can fix it)."
        ),
    },
    "XGBModel": {
        "module": ".xgboost",
        "attribute": "XGBModel",
        "group": "xgboost",
        "failure_message": (
            "ModuleNotFoundError. XGBModel is skipped"
            "(optional: maybe installing xgboost can fix it)."
        ),
    },
    "DEnsembleModel": {
        "module": ".double_ensemble",
        "attribute": "DEnsembleModel",
        "group": "lightgbm",
        "failure_message": (
            "ModuleNotFoundError. DEnsembleModel and LGBModel are skipped. "
            "(optional: maybe installing lightgbm can fix it.)"
        ),
    },
    "LGBModel": {
        "module": ".gbdt",
        "attribute": "LGBModel",
        "group": "lightgbm",
        "failure_message": (
            "ModuleNotFoundError. DEnsembleModel and LGBModel are skipped. "
            "(optional: maybe installing lightgbm can fix it.)"
        ),
    },
    "ALSTM": {
        "module": ".pytorch_alstm",
        "attribute": "ALSTM",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
    "GATs": {
        "module": ".pytorch_gats",
        "attribute": "GATs",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
    "GRU": {
        "module": ".pytorch_gru",
        "attribute": "GRU",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
    "LSTM": {
        "module": ".pytorch_lstm",
        "attribute": "LSTM",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
    "DNNModelPytorch": {
        "module": ".pytorch_nn",
        "attribute": "DNNModelPytorch",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
    "TabnetModel": {
        "module": ".pytorch_tabnet",
        "attribute": "TabnetModel",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
    "SFM_Model": {
        "module": ".pytorch_sfm",
        "attribute": "SFM_Model",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
    "TCN": {
        "module": ".pytorch_tcn",
        "attribute": "TCN",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
    "ADD": {
        "module": ".pytorch_add",
        "attribute": "ADD",
        "group": "pytorch",
        "failure_message": (
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        ),
    },
}

_PYTORCH_MODEL_NAMES = (
    "ALSTM",
    "GATs",
    "GRU",
    "LSTM",
    "DNNModelPytorch",
    "TabnetModel",
    "SFM_Model",
    "TCN",
    "ADD",
)
_ALL_MODEL_NAMES = (
    "CatBoostModel",
    "DEnsembleModel",
    "LGBModel",
    "XGBModel",
    "LinearModel",
    *_PYTORCH_MODEL_NAMES,
)


def _load_optional_model(name: str):
    if name in _OPTIONAL_MODEL_CACHE:
        return _OPTIONAL_MODEL_CACHE[name]

    export = _OPTIONAL_MODEL_EXPORTS[name]
    try:
        module = importlib.import_module(str(export["module"]), __name__)
        value = getattr(module, str(export["attribute"]))
    except ImportError as exc:
        group = str(export["group"])
        if group == "pytorch":
            if not _is_optional_pytorch_import_error(exc):
                raise
            _record_optional_import_failure(group, exc)
            _emit_optional_import_message_once(group, str(export["failure_message"]))
            value = None
        elif isinstance(exc, ModuleNotFoundError):
            _record_optional_import_failure(group, exc)
            _emit_optional_import_message_once(group, str(export["failure_message"]))
            value = None
        else:
            raise

    _OPTIONAL_MODEL_CACHE[name] = value
    return value


class LazyModelClass:
    __slots__ = ("_name",)

    def __init__(self, name: str):
        self._name = name

    @property
    def __name__(self) -> str:
        return self._name

    def _resolve(self):
        resolved = _load_optional_model(self._name)
        if resolved is None:
            raise ImportError(
                f"{self._name} is unavailable; inspect OPTIONAL_MODEL_IMPORT_FAILURES for details."
            )
        return resolved

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:
        return f"<LazyModelClass {self._name}>"


def _lazy_model_classes():
    return tuple(LazyModelClass(name) for name in _ALL_MODEL_NAMES)


def _resolved_pytorch_classes():
    return tuple(
        resolved
        for resolved in (_load_optional_model(name) for name in _PYTORCH_MODEL_NAMES)
        if resolved is not None
    )


def __getattr__(name: str):
    if name in _OPTIONAL_MODEL_EXPORTS:
        value = _load_optional_model(name)
        globals()[name] = value
        return value
    if name == "pytorch_classes":
        value = _resolved_pytorch_classes()
        globals()[name] = value
        return value
    if name == "all_model_classes":
        value = _lazy_model_classes()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | set(_OPTIONAL_MODEL_EXPORTS)
        | {"all_model_classes", "pytorch_classes"}
    )


__all__ = [
    "OPTIONAL_MODEL_IMPORT_FAILURES",
    "all_model_classes",
    "pytorch_classes",
    *_ALL_MODEL_NAMES,
]
