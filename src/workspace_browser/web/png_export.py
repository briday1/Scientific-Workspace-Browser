from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import BinaryIO, Iterable
from zipfile import ZIP_DEFLATED, ZipFile


def write_png_bundle(
    stream: BinaryIO,
    views: Iterable[tuple[str, object]],
    *,
    filename_prefix: str | None = None,
) -> int:
    """Render every native plot view into one ZIP archive."""
    count = 0
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for index, (name, value) in enumerate(views, start=1):
            png = _render_png(value)
            if png is None:
                continue
            count += 1
            prefix = f"{_filename(filename_prefix)}-" if filename_prefix else ""
            archive.writestr(f"{index:03d}-{prefix}{_filename(name)}.png", png)
        archive.writestr("README.txt", f"Rendered {count} plot view(s).\n")
    return count


def _render_png(value: object) -> bytes | None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        if isinstance(value, go.Figure):
            return pio.to_image(value, format="png", width=1600, height=900, scale=1)
    except ImportError:  # pragma: no cover
        pass

    try:
        from matplotlib.figure import Figure

        if isinstance(value, Figure):
            output = BytesIO()
            value.savefig(output, format="png", dpi=150, bbox_inches="tight")
            return output.getvalue()
    except ImportError:  # pragma: no cover
        pass
    return None


def _filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", value).strip("-.") or "plot"
    return Path(name).name[:100]
