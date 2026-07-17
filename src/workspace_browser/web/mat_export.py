from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
import re
from typing import Any, BinaryIO, Iterable

import numpy as np
from scipy.io import savemat


def write_mat_export(
    stream: BinaryIO,
    *,
    workspace_id: str,
    item_id: str,
    playback: object,
    refresh: object,
    parameters: dict[str, object],
    controls: object,
    delivered_data: object,
    layout: object,
    metadata: object,
    statistics: object,
    views: Iterable[tuple[str, object]],
) -> None:
    """Write one MATLAB structure containing data and every workspace view."""
    all_views = {
        f"view_{index:03d}_{_field(name)}": _view_value(name, value)
        for index, (name, value) in enumerate(views, start=1)
    }
    payload = {
        "workspace_id": workspace_id,
        "item_id": item_id,
        "playback": _mat_value(playback),
        "refresh": _mat_value(refresh),
        "parameters": _mat_value(parameters),
        "controls": _mat_value(controls),
        "data": _mat_value(delivered_data),
        "layout": _mat_value(layout),
        "metadata": _mat_value(metadata),
        "statistics": _mat_value(statistics),
        "views": all_views,
    }
    savemat(stream, {"workspace_export": payload}, do_compression=False, long_field_names=True, oned_as="row")


def _view_value(name: str, value: object) -> dict[str, object]:
    try:
        import plotly.graph_objects as go

        if isinstance(value, go.Figure):
            figure = value.to_plotly_json()
            return {
                "name": name,
                "kind": "plotly",
                "traces": _indexed_structs("trace", figure.get("data", [])),
                "layout": _mat_value(figure.get("layout", {})),
            }
    except ImportError:  # pragma: no cover - Plotly is a framework dependency
        pass

    try:
        from matplotlib.figure import Figure

        if isinstance(value, Figure):
            return {"name": name, "kind": "matplotlib", "axes": _matplotlib_axes(value)}
    except ImportError:  # pragma: no cover - Matplotlib is a framework dependency
        pass

    return {"name": name, "kind": "value", "value": _mat_value(value)}


def _matplotlib_axes(figure: Any) -> dict[str, object]:
    axes: dict[str, object] = {}
    for axis_index, axis in enumerate(figure.axes, start=1):
        lines = {
            f"line_{index:03d}": {
                "label": line.get_label(),
                "x": np.asarray(line.get_xdata()),
                "y": np.asarray(line.get_ydata()),
            }
            for index, line in enumerate(axis.lines, start=1)
        }
        images = {
            f"image_{index:03d}": np.asarray(image.get_array())
            for index, image in enumerate(axis.images, start=1)
        }
        collections = {
            f"collection_{index:03d}": {
                "offsets": np.asarray(collection.get_offsets()),
                "values": np.asarray(collection.get_array()) if collection.get_array() is not None else np.empty((0, 0)),
            }
            for index, collection in enumerate(axis.collections, start=1)
        }
        axes[f"axis_{axis_index:03d}"] = {
            "title": axis.get_title(),
            "x_label": axis.get_xlabel(),
            "y_label": axis.get_ylabel(),
            "lines": lines,
            "images": images,
            "collections": collections,
        }
    return axes


def _indexed_structs(prefix: str, values: Iterable[object]) -> dict[str, object]:
    return {f"{prefix}_{index:03d}": _mat_value(value) for index, value in enumerate(values, start=1)}


def _mat_value(value: object) -> object:
    if value is None:
        return np.empty((0, 0))
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {_field(field.name): _mat_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {_field(str(key)): _mat_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return np.empty((0, 0))
        if all(isinstance(item, (str, bytes, bool, int, float, complex, np.generic)) for item in value):
            return np.asarray(value)
        array = np.empty((1, len(value)), dtype=object)
        for index, item in enumerate(value):
            array[0, index] = _mat_value(item)
        return array
    return str(value)


def _field(value: str) -> str:
    field = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or "value"
    if not field[0].isalpha():
        field = f"value_{field}"
    return field[:63]
