"""
PDF parsing utilities - Watermark module
Purpose: Leave watermark text out of the layout model input

A large watermark printed across a page distorts the layout prediction: its
lines overlap the page content and their boxes cover much of the page.

find_watermark_lines() picks such lines from the "dict" output of a TextPage.
page_without_lines() makes the TextPage and the page image again without
them. The document is not changed.

Text that is left out here gets no layout box. So the rule is strict, and
text that cannot be told apart from other text stays.
"""

import math
import weakref

import numpy as np
import pymupdf
from pymupdf import mupdf


# Module-level default, overridden per call by page.get_layout(remove_watermarks=...).
REMOVE_WATERMARKS = True

# A line is diagonal if it is more than 15 degrees away from both axes.
MIN_DIAGONAL = 0.2588  # sin(15 degrees)
# Large text is this much larger than the body text and has at least this size.
LARGE_RATIO = 1.5
LARGE_SIZE = 12
# Luminance range of light text (grey, pastel). Text that is almost white is
# not looked at.
LIGHT_MIN = 0.6
LIGHT_MAX = 0.99
# Text is spread over the page if this many lines or more cover this share of it.
SPREAD_LINES = 3
SPREAD_SHARE = 0.25
# Two lines overlap if they share this much of the area of the smaller one.
OVERLAP_SHARE = 0.1
# A watermark lies over at least this many lines of other text.
OVERLAP_LINES = 3
# The axis-parallel text that stays must be at least this share of the text
# that goes. Otherwise the text in question is the page content.
KEEP_SHARE = 0.25


def iter_lines(blocks):
    """Yield the text lines of a "dict" block list, including nested blocks."""
    for block in blocks:
        if block["type"] == 2:
            yield from iter_lines(block["blocks"])
        elif block["type"] == 0:
            yield from block["lines"]


# ── Geometry of convex polygons, given as lists of (x, y) ─────────────
def _area(polygon):
    """Signed area."""
    area = 0.0
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[i - 1]
        area += x2 * y1 - x1 * y2
    return area / 2


def _intersection(subject, clipper):
    """Intersection of two convex polygons with positive area (Sutherland-Hodgman)."""
    for i, (bx, by) in enumerate(clipper):
        ax, ay = clipper[i - 1]
        if not subject:
            break
        points, subject = subject, []
        px, py = points[-1]
        p_side = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        for qx, qy in points:
            q_side = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
            if (p_side >= 0) != (q_side >= 0):  # edge p-q crosses the border
                t = p_side / (p_side - q_side)
                subject.append((px + t * (qx - px), py + t * (qy - py)))
            if q_side >= 0:
                subject.append((qx, qy))
            px, py, p_side = qx, qy, q_side
    return subject


def _hull(points):
    """Convex hull (monotone chain)."""
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and (
                (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])
            ) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return half(points) + half(points[::-1])


def _line_polygon(line, spans):
    """Corners of the rectangle a line covers, which is rotated like the text."""
    quad = pymupdf.recover_line_quad(line, spans=spans)
    polygon = [tuple(quad.ul), tuple(quad.ur), tuple(quad.lr), tuple(quad.ll)]
    return polygon if _area(polygon) >= 0 else polygon[::-1]


# ── Detection ─────────────────────────────────────────────────────────
class _Line:
    """What the rule needs to know about a text line."""

    def __init__(self, line):
        self.line = line
        self.spans = spans = [s for s in line["spans"] if s["text"].strip()]
        if not spans:
            return
        cos, sin = line["dir"]
        self.diagonal = min(abs(cos), abs(sin)) > MIN_DIAGONAL
        self.size = max(s["size"] for s in spans)
        self.text = "".join("".join(s["text"].split()) for s in line["spans"]).lower()
        self.chars = sum(len(s["text"].strip()) for s in spans)
        alpha = [s.get("alpha", 255) for s in spans]
        self.translucent = 0 < min(alpha) and max(alpha) < 255
        light = []
        for s in spans:
            c = s["color"]
            light.append((0.299 * (c >> 16 & 255) + 0.587 * (c >> 8 & 255) + 0.114 * (c & 255)) / 255)
        self.light = LIGHT_MIN <= min(light) and max(light) < LIGHT_MAX
        # Lines of the same style are judged together.
        self.style = (
            round(math.degrees(math.atan2(-sin, cos)) / 2),
            round(self.size * 2),
            min(alpha),
            max(alpha),
            round(min(light), 2),
            round(max(light), 2),
        )
        # Distance of the baseline from the origin: the same for the parts
        # of a line that MuPDF has split.
        x, y = spans[0]["origin"]
        self.baseline = y * cos - x * sin
        self._polygon = None

    @property
    def polygon(self):
        if self._polygon is None:
            self._polygon = _line_polygon(self.line, self.spans)
        return self._polygon


def _body_size(lines):
    """Character-weighted median font size of the axis-parallel lines."""
    sizes = sorted((s["size"], len(s["text"].strip()))
                   for l in lines if not l.diagonal for s in l.spans)
    half = sum(n for _, n in sizes) / 2
    seen = 0
    for size, n in sizes:
        seen += n
        if seen >= half:
            return size
    return 0


def _count(lines):
    """Number of lines of one style, where the parts of a split line count as one."""
    if not lines:
        return 0
    baselines = sorted(l.baseline for l in lines)
    gap = lines[0].size / 4
    return 1 + sum(b - a > gap for a, b in zip(baselines, baselines[1:]))


def _covers_page(lines, page_area):
    """Do the lines of one style together span a good part of the page?"""
    if _count(lines) < SPREAD_LINES:
        return False
    hull = _hull([p for l in lines for p in l.polygon])
    return abs(_area(hull)) >= SPREAD_SHARE * page_area


def _over_other_text(lines, all_lines, bboxes):
    """Return those of the lines that lie over text of another style.

    bboxes: the bboxes of all_lines as an array. Nothing is returned if the
    lines lie over fewer than OVERLAP_LINES lines.
    """
    found = []
    below = set()
    for a in lines:
        polygon = a.polygon
        area = _area(polygon)
        if area <= 0:
            continue
        x0 = min(p[0] for p in polygon)
        y0 = min(p[1] for p in polygon)
        x1 = max(p[0] for p in polygon)
        y1 = max(p[1] for p in polygon)
        near = (bboxes[:, 0] < x1) & (bboxes[:, 2] > x0) & (bboxes[:, 1] < y1) & (bboxes[:, 3] > y0)
        over = False
        for i in np.flatnonzero(near):
            b = all_lines[i]
            if b.style == a.style or b.text == a.text:
                continue
            if over and (i in below or len(below) >= OVERLAP_LINES):
                continue  # nothing new to learn
            smaller = min(area, _area(b.polygon))
            if smaller > 0 and _area(_intersection(polygon, b.polygon)) >= OVERLAP_SHARE * smaller:
                over = True
                below.add(i)
        if over:
            found.append(a)
    return found if len(below) >= OVERLAP_LINES else []


def find_watermark_lines(blocks, page_rect):
    """Return the text lines of a "dict" block list that are watermark text.

    MuPDF splits a watermark into several lines wherever it crosses a ruling
    line or other text. So the lines of a page are grouped by style (direction,
    font size, opacity, colour). A group is looked at if it is translucent, or
    diagonal and light, and if it lies over text of another style (three
    lines or more). It is watermark text if it is
    - spread over the page and lies over other text at three places or more:
      all of its lines are returned (a watermark repeated all over the page)
    - large: the lines that lie over other text are returned, together with
      the lines on the same baseline (the other parts of a split line). Other
      text of the same style stays.

    "Large" refers to the body font size, the character-weighted median span
    size of the axis-parallel lines. Axis-parallel text in question counts as
    well: it is not large if it makes up half of the page.

    Nothing is returned if little axis-parallel text would be left: then the
    text in question is the page content.
    """
    lines = [l for l in map(_Line, iter_lines(blocks)) if l.spans]
    # Most pages have neither translucent nor light diagonal text and end here.
    if not any(l.translucent or l.light and l.diagonal for l in lines):
        return []
    page_area = abs(pymupdf.Rect(page_rect).get_area())
    body_chars = sum(l.chars for l in lines if not l.diagonal)
    if not (page_area and body_chars):
        return []

    large_size = max(LARGE_RATIO * _body_size(lines), LARGE_SIZE)

    groups = {}
    for l in lines:
        groups.setdefault(l.style, []).append(l)

    bboxes = None
    found = []
    for group in groups.values():
        # A style is translucent for all of its lines or for none.
        if not (group[0].translucent or all(l.light and l.diagonal for l in group)):
            continue
        # A group that holds most of the text is not looked at, see below.
        chars = sum(l.chars for l in group)
        if body_chars - sum(l.chars for l in group if not l.diagonal) < KEEP_SHARE * chars:
            continue
        large = min(l.size for l in group) >= large_size
        spread = _covers_page(group, page_area)
        if not (large or spread):
            continue
        if bboxes is None:
            bboxes = np.array([l.line["bbox"] for l in lines])
        over = _over_other_text(group, lines, bboxes)
        if spread and _count(over) >= SPREAD_LINES:
            found.extend(group)
        elif large:
            # with the other parts of lines that MuPDF has split
            found.extend(l for l in group
                         if any(abs(l.baseline - o.baseline) <= l.size / 4 for o in over))

    removed = sum(l.chars for l in found)
    removed_axis_parallel = sum(l.chars for l in found if not l.diagonal)
    if body_chars - removed_axis_parallel < KEEP_SHARE * removed:
        return []
    return [l.line for l in found]


# ── Removal ───────────────────────────────────────────────────────────
_forwarding = {}


def _forwarding_callbacks():
    """Return the callbacks with which a MuPDF device passes calls on.

    These are the C functions of MuPDF's pass-through device. They hand a
    call to the device in the 'passthrough' member, whatever device they
    belong to. The text drawing callbacks are left out: they are handled in
    _SkipLines.
    """
    if not _forwarding:
        target = mupdf.FzDevice2()  # does nothing
        # The size only has to be at least that of a fz_device.
        forwarder = mupdf.fz_new_passthrough_device_of_size(target, 4096)
        for name in dir(mupdf.FzDevice2):
            if name.startswith("use_virtual_"):
                name = name[12:]
                if name not in ("fill_text", "stroke_text", "close_device", "drop_device"):
                    _forwarding[name] = getattr(forwarder.m_internal, name)
        mupdf.fz_close_device(forwarder)
    return _forwarding


class _Mark:
    """Where the text of a line to leave out is, and how it looks."""

    def __init__(self, line):
        spans = [s for s in line["spans"] if s["text"].strip()]
        cos, sin = self.cos, self.sin = line["dir"]
        self.min_size = min(s["size"] for s in spans) - 0.5
        self.max_size = max(s["size"] for s in spans) + 0.5
        self.min_alpha = min(s.get("alpha", 255) for s in spans) - 2
        self.max_alpha = max(s.get("alpha", 255) for s in spans) + 2
        # The glyph origins of the line are on its baseline, between the
        # ends of its bbox.
        x, y = spans[0]["origin"]
        self.baseline = y * cos - x * sin
        self.slack = self.max_size / 4
        x0, y0, x1, y1 = line["bbox"]
        along = [x * cos + y * sin for x in (x0, x1) for y in (y0, y1)]
        self.start = min(along) - self.slack
        self.end = max(along) + self.slack

    def contains(self, x, y):
        return (abs(y * self.cos - x * self.sin - self.baseline) <= self.slack
                and self.start <= x * self.cos + y * self.sin <= self.end)


class _SkipLines(mupdf.FzDevice2):
    """Device that passes a page on to another device without the text of given lines.

    Only the text drawing calls come here. All other calls are forwarded by
    MuPDF itself, see _forwarding_callbacks().

    A text object is left out if each of its spans looks like one of the
    lines (direction, font size, opacity) and has most of its glyphs in them.
    A text object that also contains other text is passed on as it is, and
    so is text in vertical writing mode.

    The decisions are kept in 'skipped', one per text drawing call. With
    'replay' a device repeats the decisions of an earlier run of the page.
    """

    def __init__(self, target, marks, replay=None):
        super().__init__()
        self.use_virtual_fill_text()
        self.use_virtual_stroke_text()
        for name, callback in _forwarding_callbacks().items():
            setattr(self.m_internal, name, callback)
        self.m_internal.passthrough = target.m_internal
        self._target = target  # keeps the target device alive
        self._marks = marks
        self._replay = replay
        self.skipped = []

    def _skip(self, text, ctm, alpha):
        if self._replay is None:
            skip = self._is_marked(text, ctm, alpha)
        else:
            skip = self._replay[len(self.skipped)]
        self.skipped.append(skip)
        return skip

    def _is_marked(self, text, ctm, alpha):
        """Does this text object consist of text of the lines to leave out?"""
        alpha *= 255
        marks = [m for m in self._marks if m.min_alpha <= alpha <= m.max_alpha]
        if not marks:
            return False
        found = False
        span = text.head
        while span:
            trm = span.trm
            a = trm.a * ctm.a + trm.b * ctm.c
            b = trm.a * ctm.b + trm.b * ctm.d
            c = trm.c * ctm.a + trm.d * ctm.c
            d = trm.c * ctm.b + trm.d * ctm.d
            size = math.sqrt(abs(a * d - b * c))
            length = math.hypot(a, b)
            alike = [
                m for m in marks
                if m.min_size <= size <= m.max_size and a * m.cos + b * m.sin >= 0.999 * length
            ]
            inside = outside = 0
            items = mupdf.FzTextSpan(span)
            for i in range(span.len):
                item = items.items(i)
                if item.ucs == 32:
                    continue
                x = item.x * ctm.a + item.y * ctm.c + ctm.e
                y = item.x * ctm.b + item.y * ctm.d + ctm.f
                if any(m.contains(x, y) for m in alike):
                    inside += 1
                else:
                    outside += 1
            if outside and outside >= inside:
                return False
            found = found or inside > 0
            span = span.next
        return found

    def fill_text(self, ctx, text, ctm, colorspace, color, alpha, color_params):
        if self._skip(text, ctm, alpha):
            return
        # The wrapper of fz_fill_text() wants the colour as a sequence.
        n = mupdf.ll_fz_colorspace_n(colorspace) if colorspace else 0
        color = tuple(mupdf.floats_getitem(color, i) for i in range(n)) or (0,)
        mupdf.ll_fz_fill_text(self._target.m_internal, text, ctm, colorspace, color, alpha, color_params)

    def stroke_text(self, ctx, text, stroke, ctm, colorspace, color, alpha, color_params):
        if self._skip(text, ctm, alpha):
            return
        mupdf.ll_fz_stroke_text(self._target.m_internal, text, stroke, ctm, colorspace, color, alpha, color_params)


def _run(fzpage, target, marks, replay=None):
    """Run the page into the target device and return what was left out."""
    device = _SkipLines(target, marks, replay)
    try:
        mupdf.fz_run_page(fzpage, device, mupdf.FzMatrix(), mupdf.FzCookie())
    finally:
        mupdf.fz_close_device(device)
        mupdf.fz_close_device(target)
    return device.skipped


def page_without_lines(page, flags, lines):
    """Return TextPage and Pixmap of a page without the text of the given lines.

    The lines are from the "dict" output of page.get_textpage(flags=flags).
    The results are made like those of page.get_textpage(flags=flags) and
    page.get_pixmap(), only that the text objects of these lines are not
    drawn. Text of a line that shares its text object with other text stays.
    """
    marks = [_Mark(line) for line in lines]
    fzpage = pymupdf._as_fz_page(page)

    # Text is extracted from the unrotated page, as in Page.get_textpage().
    rotation = page.rotation
    if rotation:
        page.set_rotation(0)
    try:
        stext_page = mupdf.FzStextPage(mupdf.fz_bound_page(fzpage))
        skipped = _run(fzpage, mupdf.fz_new_stext_device(stext_page, mupdf.FzStextOptions(flags)), marks)
    finally:
        if rotation:
            page.set_rotation(rotation)
    textpage = pymupdf.TextPage(stext_page)
    textpage.parent = weakref.proxy(page)

    # The image is made from a display list, as in Page.get_pixmap(). The
    # page may be rotated now, so the same text drawing calls are left out
    # by their position in the run.
    display_list = mupdf.FzDisplayList(mupdf.fz_bound_page(fzpage))
    if _run(fzpage, mupdf.fz_new_list_device(display_list), marks, replay=skipped) != skipped:
        raise RuntimeError("text drawing calls of the page differ between two runs")
    pixmap = pymupdf.DisplayList(display_list).get_pixmap()
    return textpage, pixmap
