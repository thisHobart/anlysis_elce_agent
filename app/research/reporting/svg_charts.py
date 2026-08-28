"""Hand-built inline SVG charts for the research report.

Every chart is a self-contained ``<svg>`` string with no external fonts, scripts,
or images, so the same markup works inline in the HTML report and as a
standalone ``.svg`` file inside the research package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from html import escape
from typing import Any

INK = "#1F2430"
MUTED = "#667085"
AXIS = "#98A2B3"
GRID = "#E7EAEF"
PANEL = "#FFFFFF"
ACCENT = "#3F51B5"
ACCENT_SOFT = "#E5E8F7"
POSITIVE = "#0F766E"
NEGATIVE = "#BE123C"
SERIES_COLORS = ("#3F51B5", "#0F766E", "#B45309", "#BE123C", "#7C3AED", "#475569")

# Widths for laying out value labels without measuring text: one 10.5px digit is
# about 6px wide, and a label needs this much clearance from the bar end.
LABEL_CHARACTER_WIDTH = 6.2
LABEL_GAP = 6.0


def elide(name: str, limit: int) -> str:
    """Shorten a long series name from the middle, keeping the part that tells it apart.

    Cutting the tail off ``fcst_new_energy_type_13`` and ``fcst_new_energy_type_1``
    prints two different variables under one identical label.
    """

    if len(name) <= limit or limit < 5:
        return name[:limit]
    return f"{name[: limit - 4]}…{name[-3:]}"

# QtSvg does not reliably resolve CSS font-family lists for CJK glyphs. A
# single CJK-capable family keeps its glyph metrics and synthetic weight
# aligned; platforms without this font still use Qt's normal font fallback.
FONT = "'Microsoft YaHei'"


def format_number(value: float | None, *, unit: str = "") -> str:
    """Format a statistic compactly without losing the reader's sense of scale."""

    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        text = f"{value / 1_000_000:,.2f}M"
    elif magnitude >= 10_000:
        text = f"{value / 1_000:,.1f}k"
    elif magnitude >= 100:
        text = f"{value:,.0f}"
    elif magnitude >= 1:
        text = f"{value:,.2f}"
    elif magnitude > 0:
        text = f"{value:,.4f}".rstrip("0").rstrip(".")
    else:
        text = "0"
    return f"{text} {unit}".strip() if unit else text


def _nice_step(span: float, target: int) -> float:
    if span <= 0:
        return 1.0
    rough = span / max(1, target)
    exponent = math.floor(math.log10(rough))
    base = 10**exponent
    for multiple in (1, 2, 2.5, 5, 10):
        if rough <= base * multiple:
            return base * multiple
    return base * 10


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0]
    if math.isclose(low, high):
        return [low]
    step = _nice_step(high - low, count)
    start = math.floor(low / step) * step
    values: list[float] = []
    current = start
    while current <= high + step * 0.5 and len(values) < 40:
        if current >= low - step * 0.5:
            values.append(round(current, 10))
        current += step
    return values or [low, high]


class Canvas:
    """Minimal cartesian canvas with a left value axis and a bottom category axis."""

    def __init__(
        self,
        *,
        width: int = 760,
        height: int = 300,
        left: int = 62,
        right: int = 18,
        top: int = 26,
        bottom: int = 40,
    ) -> None:
        self.width = width
        self.height = height
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.parts: list[str] = []
        self.low = 0.0
        self.high = 1.0

    @property
    def plot_width(self) -> float:
        return self.width - self.left - self.right

    @property
    def plot_height(self) -> float:
        return self.height - self.top - self.bottom

    def set_value_domain(self, values: Sequence[float], *, include_zero: bool = False) -> None:
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not finite:
            finite = [0.0, 1.0]
        low, high = min(finite), max(finite)
        if include_zero:
            low, high = min(low, 0.0), max(high, 0.0)
        if math.isclose(low, high):
            padding = abs(low) * 0.1 or 1.0
            low, high = low - padding, high + padding
        else:
            padding = (high - low) * 0.08
            low, high = low - padding, high + padding
        self.low, self.high = low, high

    def y(self, value: float) -> float:
        ratio = (float(value) - self.low) / (self.high - self.low)
        return self.top + self.plot_height * (1 - ratio)

    def x(self, ratio: float) -> float:
        return self.left + self.plot_width * ratio

    def band(self, index: int, count: int) -> tuple[float, float]:
        """Return the (start, width) of one categorical band."""

        width = self.plot_width / max(1, count)
        return self.left + width * index, width

    def value_axis(self, *, unit: str = "", ticks: int = 5) -> None:
        for value in _ticks(self.low, self.high, ticks):
            position = self.y(value)
            if position < self.top - 1 or position > self.height - self.bottom + 1:
                continue
            self.parts.append(
                f'<line x1="{self.left:.1f}" y1="{position:.1f}" x2="{self.width - self.right:.1f}" '
                f'y2="{position:.1f}" stroke="{GRID}" stroke-width="1"/>'
            )
            self.parts.append(
                f'<text x="{self.left - 8:.1f}" y="{position + 3.5:.1f}" text-anchor="end" '
                f'font-size="11" fill="{MUTED}">{escape(format_number(value))}</text>'
            )
        if unit:
            self.parts.append(
                f'<text x="{self.left - 8:.1f}" y="{self.top - 10:.1f}" text-anchor="end" '
                f'font-size="10" fill="{AXIS}">{escape(unit)}</text>'
            )

    def zero_line(self) -> None:
        if self.low < 0 < self.high:
            position = self.y(0)
            self.parts.append(
                f'<line x1="{self.left:.1f}" y1="{position:.1f}" x2="{self.width - self.right:.1f}" '
                f'y2="{position:.1f}" stroke="{AXIS}" stroke-width="1" stroke-dasharray="4 3"/>'
            )

    def category_labels(self, labels: Sequence[str], *, rotate: bool = False) -> None:
        count = len(labels)
        if not count:
            return
        stride = max(1, math.ceil(count / 14))
        for index, label in enumerate(labels):
            if index % stride:
                continue
            start, width = self.band(index, count)
            center = start + width / 2
            baseline = self.height - self.bottom + 16
            if rotate:
                self.parts.append(
                    f'<text transform="translate({center:.1f} {baseline:.1f}) rotate(-38)" text-anchor="end" '
                    f'font-size="10" fill="{MUTED}">{escape(label)}</text>'
                )
            else:
                self.parts.append(
                    f'<text x="{center:.1f}" y="{baseline:.1f}" text-anchor="middle" '
                    f'font-size="10" fill="{MUTED}">{escape(label)}</text>'
                )

    def axis_caption(self, text: str) -> None:
        self.parts.append(
            f'<text x="{self.left:.1f}" y="{self.height - 6:.1f}" font-size="10" '
            f'fill="{AXIS}">{escape(text)}</text>'
        )

    def render(self, *, title: str, subtitle: str = "") -> str:
        header = [
            (
                f'<text x="{self.left - 46:.1f}" y="16" font-size="13" font-weight="600" '
                f'fill="{INK}">{escape(title)}</text>'
            )
        ]
        if subtitle:
            header.append(
                f'<text x="{self.width - self.right:.1f}" y="16" text-anchor="end" font-size="10.5" '
                f'fill="{MUTED}">{escape(subtitle)}</text>'
            )
        body = "".join(header + self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'role="img" aria-label="{escape(title)}" '
            f'style="width:100%;height:auto;font-family:{FONT}">'
            f'<rect width="{self.width}" height="{self.height}" fill="{PANEL}"/>{body}</svg>'
        )


def empty_chart(title: str, message: str) -> str:
    canvas = Canvas(height=140)
    canvas.parts.append(
        f'<text x="{canvas.width / 2:.1f}" y="{canvas.height / 2:.1f}" text-anchor="middle" '
        f'font-size="12" fill="{MUTED}">{escape(message)}</text>'
    )
    return canvas.render(title=title)


def line_chart(
    points: Sequence[tuple[float, float]],
    *,
    title: str,
    subtitle: str = "",
    unit: str = "",
    caption: str = "",
    fill: bool = False,
) -> str:
    """Draw one continuous series indexed by an already-normalised x position."""

    values = [value for _position, value in points]
    if not values:
        return empty_chart(title, "本轮没有可绘制的数据点")
    canvas = Canvas()
    canvas.set_value_domain(values, include_zero=True)
    canvas.value_axis(unit=unit)
    canvas.zero_line()
    coordinates = [(canvas.x(position), canvas.y(value)) for position, value in points]
    path = " ".join(
        f"{'M' if index == 0 else 'L'}{x:.2f} {y:.2f}" for index, (x, y) in enumerate(coordinates)
    )
    if fill:
        baseline = canvas.y(max(canvas.low, min(canvas.high, 0.0)))
        area = f"{path} L{coordinates[-1][0]:.2f} {baseline:.2f} L{coordinates[0][0]:.2f} {baseline:.2f} Z"
        canvas.parts.append(f'<path d="{area}" fill="{ACCENT_SOFT}"/>')
    canvas.parts.append(f'<path d="{path}" fill="none" stroke="{ACCENT}" stroke-width="1.4"/>')
    if caption:
        canvas.axis_caption(caption)
    return canvas.render(title=title, subtitle=subtitle)


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    title: str,
    subtitle: str = "",
    unit: str = "",
    caption: str = "",
    rotate_labels: bool = False,
    diverging: bool = False,
) -> str:
    """Draw one vertical bar per category, colouring by sign when diverging."""

    usable = [value for value in values if value is not None]
    if not usable:
        return empty_chart(title, "本轮没有可绘制的分组结果")
    canvas = Canvas()
    canvas.set_value_domain(usable, include_zero=True)
    canvas.value_axis(unit=unit)
    baseline = canvas.y(max(canvas.low, min(canvas.high, 0.0)))
    for index, value in enumerate(values):
        if value is None:
            continue
        start, width = canvas.band(index, len(values))
        bar_width = max(2.0, width * 0.66)
        offset = start + (width - bar_width) / 2
        position = canvas.y(value)
        top = min(position, baseline)
        height = max(1.0, abs(baseline - position))
        color = (NEGATIVE if value < 0 else POSITIVE) if diverging else ACCENT
        canvas.parts.append(
            f'<rect x="{offset:.2f}" y="{top:.2f}" width="{bar_width:.2f}" height="{height:.2f}" '
            f'rx="2" fill="{color}"/>'
        )
    canvas.zero_line()
    canvas.category_labels(labels, rotate=rotate_labels)
    if caption:
        canvas.axis_caption(caption)
    return canvas.render(title=title, subtitle=subtitle)


def histogram_chart(
    edges: Sequence[float],
    counts: Sequence[int],
    *,
    title: str,
    subtitle: str = "",
    unit: str = "",
) -> str:
    """Draw a distribution with a real numeric x axis and a zero marker."""

    if not counts:
        return empty_chart(title, "本轮没有可绘制的分布")
    canvas = Canvas()
    canvas.set_value_domain([0, max(counts)])
    canvas.low = 0.0
    canvas.value_axis(unit="时段数")
    low, high = float(edges[0]), float(edges[-1])
    span = high - low or 1.0
    for index, count in enumerate(counts):
        start = canvas.x((float(edges[index]) - low) / span)
        end = canvas.x((float(edges[index + 1]) - low) / span)
        top = canvas.y(count)
        canvas.parts.append(
            f'<rect x="{start:.2f}" y="{top:.2f}" width="{max(0.6, end - start - 0.6):.2f}" '
            f'height="{max(0.4, canvas.y(0) - top):.2f}" fill="{ACCENT}" opacity="0.85"/>'
        )
    if low < 0 < high:
        zero = canvas.x((0 - low) / span)
        canvas.parts.append(
            f'<line x1="{zero:.2f}" y1="{canvas.top:.1f}" x2="{zero:.2f}" '
            f'y2="{canvas.height - canvas.bottom:.1f}" stroke="{NEGATIVE}" stroke-width="1" '
            f'stroke-dasharray="4 3"/>'
        )
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = low + span * ratio
        canvas.parts.append(
            f'<text x="{canvas.x(ratio):.1f}" y="{canvas.height - canvas.bottom + 16:.1f}" '
            f'text-anchor="middle" font-size="10" fill="{MUTED}">{escape(format_number(value))}</text>'
        )
    canvas.axis_caption(f"价格区间{f'（{unit}）' if unit else ''}；虚线为零价")
    return canvas.render(title=title, subtitle=subtitle)


def stem_chart(
    lags: Sequence[int],
    values: Sequence[float | None],
    *,
    title: str,
    subtitle: str = "",
    band: float | None = None,
    caption: str = "",
) -> str:
    """Draw correlation-by-lag stems with an optional significance band."""

    usable = [value for value in values if value is not None]
    if not usable:
        return empty_chart(title, "本轮没有可绘制的滞后结果")
    canvas = Canvas(height=260)
    domain = [*usable, 0.0]
    if band is not None:
        domain.extend([band, -band])
    canvas.set_value_domain(domain, include_zero=True)
    canvas.value_axis(unit="相关系数")
    if band is not None:
        top = canvas.y(band)
        bottom = canvas.y(-band)
        canvas.parts.append(
            f'<rect x="{canvas.left:.1f}" y="{top:.1f}" width="{canvas.plot_width:.1f}" '
            f'height="{max(1.0, bottom - top):.1f}" fill="{ACCENT_SOFT}" opacity="0.7"/>'
        )
    baseline = canvas.y(0)
    count = len(values)
    for index, value in enumerate(values):
        if value is None:
            continue
        start, width = canvas.band(index, count)
        center = start + width / 2
        position = canvas.y(value)
        significant = band is None or abs(value) > band
        color = ACCENT if significant else AXIS
        canvas.parts.append(
            f'<line x1="{center:.2f}" y1="{baseline:.2f}" x2="{center:.2f}" y2="{position:.2f}" '
            f'stroke="{color}" stroke-width="{max(1.0, min(3.0, width * 0.45)):.2f}"/>'
        )
    canvas.zero_line()
    canvas.category_labels([str(lag) for lag in lags])
    canvas.axis_caption(caption or "滞后（个采样间隔）")
    return canvas.render(title=title, subtitle=subtitle)


def multi_line_chart(
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    *,
    title: str,
    subtitle: str = "",
    unit: str = "",
    caption: str = "",
) -> str:
    """Overlay a small number of named series that share one value axis."""

    values = [value for _name, points in series for _position, value in points]
    if not values:
        return empty_chart(title, "本轮没有可绘制的曲线")
    canvas = Canvas(height=320, bottom=64)
    canvas.set_value_domain(values, include_zero=True)
    canvas.value_axis(unit=unit)
    canvas.zero_line()
    for index, (name, points) in enumerate(series):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        path = " ".join(
            f"{'M' if position_index == 0 else 'L'}{canvas.x(position):.2f} {canvas.y(value):.2f}"
            for position_index, (position, value) in enumerate(points)
        )
        canvas.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        legend_x = canvas.left + (index % 3) * (canvas.plot_width / 3)
        legend_y = canvas.height - 26 + (index // 3) * 15
        canvas.parts.append(
            f'<rect x="{legend_x:.1f}" y="{legend_y - 7:.1f}" width="10" height="3" rx="1.5" fill="{color}"/>'
            f'<text x="{legend_x + 15:.1f}" y="{legend_y:.1f}" font-size="10.5" '
            f'fill="{MUTED}">{escape(name)}</text>'
        )
    if caption:
        canvas.parts.append(
            f'<text x="{canvas.left:.1f}" y="{canvas.height - canvas.bottom + 16:.1f}" font-size="10" '
            f'fill="{AXIS}">{escape(caption)}</text>'
        )
    return canvas.render(title=title, subtitle=subtitle)


def ranked_bar_chart(
    rows: Sequence[tuple[str, float | None]],
    *,
    title: str,
    subtitle: str = "",
    caption: str = "",
    symmetric: bool = True,
    unit: str = "",
) -> str:
    """Draw one horizontal bar per named item, ranked by magnitude."""

    usable = [(name, value) for name, value in rows if value is not None]
    if not usable:
        return empty_chart(title, "本轮没有可排序的结果")
    usable = sorted(usable, key=lambda row: abs(row[1]), reverse=True)[:12]
    row_height = 26
    height = 46 + row_height * len(usable)
    width = 760
    left = 168
    right = 72
    plot_width = width - left - right
    limit = max(abs(value) for _name, value in usable) or 1.0
    if symmetric:
        low, high = -limit, limit
    else:
        low, high = 0.0, limit
    span = high - low

    def position(value: float) -> float:
        return left + plot_width * ((value - low) / span)

    parts = [
        f'<text x="18" y="18" font-size="13" font-weight="600" fill="{INK}">{escape(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{width - 18}" y="18" text-anchor="end" font-size="10.5" '
            f'fill="{MUTED}">{escape(subtitle)}</text>'
        )
    zero = position(0.0)
    parts.append(
        f'<line x1="{zero:.1f}" y1="32" x2="{zero:.1f}" y2="{height - 20}" stroke="{GRID}" stroke-width="1"/>'
    )
    for index, (name, value) in enumerate(usable):
        center = 44 + row_height * index
        end = position(value)
        start = min(zero, end)
        bar_width = max(1.5, abs(end - zero))
        color = NEGATIVE if value < 0 else POSITIVE
        parts.append(
            f'<text x="{left - 12}" y="{center + 4}" text-anchor="end" font-size="11" '
            f'fill="{INK}">{escape(elide(name, 22))}</text>'
        )
        parts.append(
            f'<rect x="{start:.1f}" y="{center - 8:.1f}" width="{bar_width:.1f}" height="16" rx="3" '
            f'fill="{color}" opacity="0.9"/>'
        )
        label = format_number(value, unit=unit)
        label_width = len(label) * LABEL_CHARACTER_WIDTH
        outside_x = end + (LABEL_GAP if value >= 0 else -LABEL_GAP)
        outside_fits = (
            outside_x + label_width <= width - 6 if value >= 0 else outside_x - label_width >= left - 6
        )
        if outside_fits:
            anchor, label_x, fill = ("start" if value >= 0 else "end"), outside_x, MUTED
        else:
            # The bar reaches the axis margin; keeping the label outside would print it
            # on top of the category name. Put it inside the bar instead.
            anchor = "end" if value >= 0 else "start"
            label_x = end - LABEL_GAP if value >= 0 else end + LABEL_GAP
            fill = PANEL
        parts.append(
            f'<text x="{label_x:.1f}" y="{center + 4:.1f}" text-anchor="{anchor}" font-size="10.5" '
            f'fill="{fill}">{escape(label)}</text>'
        )
    if caption:
        parts.append(f'<text x="18" y="{height - 6}" font-size="10" fill="{AXIS}">{escape(caption)}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}" style="width:100%;height:auto;font-family:{FONT}">'
        f'<rect width="{width}" height="{height}" fill="{PANEL}"/>{"".join(parts)}</svg>'
    )


def _heat_color(value: float) -> str:
    """Map a correlation in [-1, 1] onto a teal-to-crimson diverging ramp."""

    ratio = max(-1.0, min(1.0, value))
    if ratio >= 0:
        weight = ratio
        red = int(255 + (15 - 255) * weight)
        green = int(255 + (118 - 255) * weight)
        blue = int(255 + (110 - 255) * weight)
    else:
        weight = -ratio
        red = int(255 + (190 - 255) * weight)
        green = int(255 + (18 - 255) * weight)
        blue = int(255 + (60 - 255) * weight)
    return f"#{red:02X}{green:02X}{blue:02X}"


def heatmap_chart(
    names: Sequence[str],
    matrix: Sequence[Sequence[float | None]],
    *,
    title: str,
    subtitle: str = "",
    caption: str = "",
) -> str:
    """Draw a labelled correlation matrix with in-cell values."""

    count = len(names)
    if count == 0:
        return empty_chart(title, "本轮没有可绘制的相关矩阵")
    cell = max(28, min(58, int(520 / count)))
    left = 150
    top = 62
    width = left + cell * count + 24
    height = top + cell * count + 42
    parts = [
        f'<text x="18" y="20" font-size="13" font-weight="600" fill="{INK}">{escape(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{width - 18}" y="20" text-anchor="end" font-size="10.5" '
            f'fill="{MUTED}">{escape(subtitle)}</text>'
        )
    for index, name in enumerate(names):
        column_x = left + cell * index + cell / 2
        parts.append(
            f'<text transform="translate({column_x:.1f} {top - 8:.1f}) rotate(-42)" text-anchor="start" '
            f'font-size="10" fill="{MUTED}">{escape(elide(name, 18))}</text>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{top + cell * index + cell / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="{INK}">{escape(elide(name, 20))}</text>'
        )
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            x = left + cell * column_index
            y = top + cell * row_index
            if value is None:
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="3" fill="#F4F5F7"/>'
                )
                continue
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="3" '
                f'fill="{_heat_color(float(value))}"/>'
            )
            text_color = "#FFFFFF" if abs(float(value)) > 0.62 else INK
            parts.append(
                f'<text x="{x + (cell - 2) / 2:.1f}" y="{y + (cell - 2) / 2 + 3.5:.1f}" text-anchor="middle" '
                f'font-size="{9 if cell < 40 else 10}" fill="{text_color}">{float(value):.2f}</text>'
            )
    parts.append(
        f'<text x="18" y="{height - 10}" font-size="10" fill="{AXIS}">'
        f'{escape(caption or "同期 Pearson 相关；蓝绿为正、红为负")}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}" style="width:100%;height:auto;font-family:{FONT}">'
        f'<rect width="{width}" height="{height}" fill="{PANEL}"/>{"".join(parts)}</svg>'
    )


def stacked_share_chart(
    parts_in: Sequence[tuple[str, float]],
    *,
    title: str,
    subtitle: str = "",
    caption: str = "",
) -> str:
    """Draw one horizontal 100% bar showing how variance splits across components."""

    usable = [(name, float(value)) for name, value in parts_in if value is not None and float(value) > 0]
    if not usable:
        return empty_chart(title, "本轮没有可绘制的成分占比")
    total = sum(value for _name, value in usable) or 1.0
    width = 760
    height = 150
    left = 18
    bar_width = width - 36
    parts = [
        f'<text x="{left}" y="20" font-size="13" font-weight="600" fill="{INK}">{escape(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{width - 18}" y="20" text-anchor="end" font-size="10.5" '
            f'fill="{MUTED}">{escape(subtitle)}</text>'
        )
    offset = float(left)
    for index, (name, value) in enumerate(usable):
        segment = bar_width * value / total
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        parts.append(
            f'<rect x="{offset:.1f}" y="38" width="{max(1.0, segment - 1):.1f}" height="34" rx="3" fill="{color}"/>'
        )
        if segment > 46:
            parts.append(
                f'<text x="{offset + segment / 2:.1f}" y="59" text-anchor="middle" font-size="11" '
                f'fill="#FFFFFF">{value / total * 100:.0f}%</text>'
            )
        legend_x = left + (index % 3) * (bar_width / 3)
        legend_y = 100 + (index // 3) * 18
        parts.append(
            f'<rect x="{legend_x:.1f}" y="{legend_y - 8:.1f}" width="10" height="10" rx="2" fill="{color}"/>'
            f'<text x="{legend_x + 16:.1f}" y="{legend_y:.1f}" font-size="10.5" fill="{MUTED}">'
            f'{escape(name)} · {value / total * 100:.1f}%</text>'
        )
        offset += segment
    if caption:
        parts.append(f'<text x="{left}" y="{height - 8}" font-size="10" fill="{AXIS}">{escape(caption)}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}" style="width:100%;height:auto;font-family:{FONT}">'
        f'<rect width="{width}" height="{height}" fill="{PANEL}"/>{"".join(parts)}</svg>'
    )


def range_chart(
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    subtitle: str = "",
    caption: str = "",
) -> str:
    """Draw min/mean/max ranges so drifting relationships are visible at a glance."""

    usable = [row for row in rows if row.get("mean") is not None]
    if not usable:
        return empty_chart(title, "本轮没有可绘制的稳定性区间")
    width = 760
    left = 168
    right = 60
    row_height = 30
    height = 52 + row_height * len(usable)
    plot_width = width - left - right
    limit = max(
        max(abs(float(row.get("max") or 0)), abs(float(row.get("min") or 0)), abs(float(row["mean"])))
        for row in usable
    ) or 1.0

    def position(value: float) -> float:
        return left + plot_width * ((value + limit) / (2 * limit))

    parts = [f'<text x="18" y="20" font-size="13" font-weight="600" fill="{INK}">{escape(title)}</text>']
    if subtitle:
        parts.append(
            f'<text x="{width - 18}" y="20" text-anchor="end" font-size="10.5" '
            f'fill="{MUTED}">{escape(subtitle)}</text>'
        )
    zero = position(0.0)
    parts.append(
        f'<line x1="{zero:.1f}" y1="34" x2="{zero:.1f}" y2="{height - 24}" stroke="{GRID}" stroke-width="1"/>'
    )
    for index, row in enumerate(usable):
        center = 48 + row_height * index
        low = position(float(row.get("min") if row.get("min") is not None else row["mean"]))
        high = position(float(row.get("max") if row.get("max") is not None else row["mean"]))
        mean = position(float(row["mean"]))
        crosses_zero = (float(row.get("min") or 0) < 0 < float(row.get("max") or 0))
        color = NEGATIVE if crosses_zero else ACCENT
        parts.append(
            f'<text x="{left - 12}" y="{center + 4}" text-anchor="end" font-size="11" '
            f'fill="{INK}">{escape(elide(str(row.get("name", "")), 22))}</text>'
        )
        parts.append(
            f'<rect x="{min(low, high):.1f}" y="{center - 5:.1f}" width="{max(2.0, abs(high - low)):.1f}" '
            f'height="10" rx="5" fill="{color}" opacity="0.22"/>'
        )
        parts.append(f'<circle cx="{mean:.1f}" cy="{center:.1f}" r="4.5" fill="{color}"/>')
        parts.append(
            f'<text x="{width - right + 8}" y="{center + 4}" font-size="10.5" fill="{MUTED}">'
            f'{format_number(row["mean"])}</text>'
        )
    parts.append(
        f'<text x="18" y="{height - 8}" font-size="10" fill="{AXIS}">'
        f'{escape(caption or "圆点为滚动窗口均值，色带为最小值到最大值；红色表示区间跨越零")}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}" style="width:100%;height:auto;font-family:{FONT}">'
        f'<rect width="{width}" height="{height}" fill="{PANEL}"/>{"".join(parts)}</svg>'
    )
