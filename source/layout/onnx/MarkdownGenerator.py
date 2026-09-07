"""
MarkdownGenerator.py

Converts BoxRFDGNN.predict(page, return_raw=True) output to Markdown text.

Usage
-----
    from source.layout.onnx.MarkdownGenerator import MarkdownGenerator
    gen = MarkdownGenerator(model)
    md  = gen.generate(page)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Class-to-Markdown mapping constants
# ---------------------------------------------------------------------------

_SKIP_CLASSES    = {"page-header", "page-footer"}
_HEADING_CLASSES = {"title", "section-header"}
_LIST_CLASSES    = {"list-item"}
_ITALIC_CLASSES  = {"caption", "footnote"}


# ---------------------------------------------------------------------------
# MarkdownGenerator
# ---------------------------------------------------------------------------

class MarkdownGenerator:
    """
    Converts BoxRFDGNN layout detection results into Markdown text.

    Parameters
    ----------
    model : BoxRFDGNN instance used to run inference.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, page: Any, groups: list | None = None,
                 join: bool = True, skip_header_footer: bool = True):
        from ..pymupdf_util import create_input_data_from_page

        if groups is None:
            groups = self.model.predict(page, return_raw=True)['groups']
        if not groups:
            return "" if join else []

        data_dict = create_input_data_from_page(page, options={
            "input_type": self.model.input_type,
            "feature_set_name": self.model.feature_set_name,
            "feature_extractor": self.model.feature_extractor,
        })
        all_texts: list[str] = data_dict.get("text", [])
        all_bboxes: list = data_dict.get("bboxes", [])

        md_blocks: list[str] = []
        for group_idx, group in enumerate(groups):
            if skip_header_footer and group["class_name"] in _SKIP_CLASSES:
                continue
            block = self._render_group(group, group_idx, all_texts, all_bboxes)
            md_blocks.append(block)

        if join:
            return "\n\n".join(b for b in md_blocks if b)
        else:
            return md_blocks

    # ------------------------------------------------------------------
    # Group rendering
    # ------------------------------------------------------------------

    def _render_group(self, group: dict, group_idx: int, all_texts: list[str], all_bboxes: list) -> str:
        """Convert a single layout group to a Markdown string."""
        cls  = group.get("class_name", "text")
        text = self._group_text(group, all_texts, all_bboxes).strip()

        if cls == "table":
            cells    = group.get("table_cells", [])
            table_md = self._render_table(cells)
            return table_md if table_md else text

        if cls in _HEADING_CLASSES:
            return f"## {text}" if text else ""

        if cls in _LIST_CLASSES:
            return f"- {text}" if text else ""

        if cls in _ITALIC_CLASSES:
            return f"*{text}*" if text else ""

        if cls == "formula":
            return f"${text}$" if text else ""

        if cls == "picture":
            x1, y1, x2, y2 = group["group_bbox"]
            return f"[Figure-{group_idx}]({int(x1)},{int(y1)},{int(x2)},{int(y2)})"

        # text, paragraph, and any unknown classes
        return text

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_table(cells: list[Any]) -> str:
        """
        Renders a table into Markdown format.
        Modified to support data replication for merged cells (row_span/col_span).
        """
        if not cells:
            return ""

        # Determine the overall grid size
        n_rows = max(c.row + (c.row_span if hasattr(c, 'row_span') else 1) for c in cells)
        n_cols = max(c.col + (c.col_span if hasattr(c, 'col_span') else 1) for c in cells)

        # Initialize an empty grid
        grid: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]

        for cell in cells:
            r_start, c_start = cell.row, cell.col
            # Default to 1 if span attributes are missing
            r_span = getattr(cell, 'row_span', 1)
            c_span = getattr(cell, 'col_span', 1)

            text = cell.text.strip()
            if not text:
                continue

            # Replicate data across all cells covered by the span
            # This ensures that even if the markdown renderer doesn't support merging,
            # the information is preserved in all logical coordinates.
            for r in range(r_start, r_start + r_span):
                for co in range(c_start, c_start + c_span):
                    if 0 <= r < n_rows and 0 <= co < n_cols:
                        prev = grid[r][co]
                        # If multiple text blocks fall into the same replicated cell, append them
                        grid[r][co] = (prev + " " + text).strip() if prev else text

        def _esc(s: str) -> str:
            # Escape pipe characters and remove newlines for Markdown table compatibility
            return s.replace("|", "\\|").replace("\n", " ")

        # 1. Generate Header Row
        header = "| " + " | ".join(_esc(grid[0][c]) for c in range(n_cols)) + " |"

        # 2. Generate Separator Row
        sep = "| " + " | ".join("---" for _ in range(n_cols)) + " |"

        # 3. Generate Data Rows (starting from row 1)
        rows = []
        for r in range(1, n_rows):
            row_str = "| " + " | ".join(_esc(grid[r][c]) for c in range(n_cols)) + " |"
            rows.append(row_str)

        return "\n".join([header, sep] + rows)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_text(group: dict, all_texts: list[str], all_bboxes: list) -> str:
        """Join texts sorted by visual reading order within the group."""
        margin = 5
        indicies = group.get("indicies", [])
        valid = [
            (idx, all_bboxes[idx])
            for idx in indicies
            if 0 <= idx < len(all_texts) and 0 <= idx < len(all_bboxes)
               and all_texts[idx].strip()
        ]
        valid.sort(key=lambda x: (x[1][1] // margin, x[1][0]))
        return " ".join(all_texts[idx].strip() for idx, _ in valid)

