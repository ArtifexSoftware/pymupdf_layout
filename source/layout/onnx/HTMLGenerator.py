"""
HTMLGenerator.py

Converts BoxRFDGNN layout detection results into a complete HTML document.

Single-page usage
-----------------
    from source.layout.onnx.HTMLGenerator import HTMLGenerator
    gen  = HTMLGenerator(model)
    html = gen.generate(page)

Whole-PDF usage (multi-process, mirrors MultiProcessWrapper.to_markdown)
------------------------------------------------------------------------
    from source.layout.onnx.HTMLGenerator import HTMLGenerator
    gen  = HTMLGenerator(model, n_workers=4)
    html = gen.generate("/path/to/document.pdf")
"""

from __future__ import annotations

import html as html_module
import multiprocessing as mp
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Callable, Optional, Union


# ---------------------------------------------------------------------------
# Class-to-rendering mapping constants
# ---------------------------------------------------------------------------

_SKIP_CLASSES    = {"page-header", "page-footer"}
_HEADING_CLASSES = {"title", "section-header"}
_LIST_CLASSES    = {"list-item"}
_ITALIC_CLASSES  = {"caption", "footnote"}


# ---------------------------------------------------------------------------
# Worker-side globals
# ---------------------------------------------------------------------------

_worker_model = None


def _worker_init(
    config_path     : str,
    model_path      : str,
    imf_model_path  : str,
    feature_set_name: str,
    input_type      : Optional[tuple],
    use_gpu         : bool,
    use_sort        : bool,
) -> None:
    """Initialize BoxRFDGNN once per worker process."""
    global _worker_model
    from .BoxRFDGNN import BoxRFDGNN
    _worker_model = BoxRFDGNN(
        config_path            = config_path,
        model_path             = model_path,
        imf_model_path         = imf_model_path,
        feature_set_name       = feature_set_name,
        input_type             = input_type,
        enable_inference_cache = False,
        use_gpu                = use_gpu,
        use_sort               = use_sort,
    )


def _worker_generate_page(args: tuple) -> tuple[int, str]:
    """
    Worker task: open PDF, process one page, return (page_no, html_body_fragment).

    Parameters
    ----------
    args : (pdf_path, page_no, skip_header_footer)
    """
    pdf_path, page_no, skip_header_footer = args
    import fitz
    doc  = fitz.open(pdf_path)
    page = doc[page_no]
    fragment = _worker_model.html_generator.generate_page_fragment(
        page, skip_header_footer=skip_header_footer
    )
    doc.close()
    return page_no, fragment


# ---------------------------------------------------------------------------
# HTMLGenerator
# ---------------------------------------------------------------------------

class HTMLGenerator:
    """
    Converts BoxRFDGNN layout detection results into a complete HTML document.

    Parameters
    ----------
    model      : BoxRFDGNN instance.
    n_workers  : Number of worker processes for whole-PDF processing.
                 Values < 2 run in the main process (no Pool created).
    use_gpu    : Passed to worker BoxRFDGNN instances.
    use_sort   : Passed to worker BoxRFDGNN instances.
    title      : Value used in the HTML <title> tag.
    """

    def __init__(
        self,
        model    : Any,
        n_workers: int = 1,
        use_gpu  : bool = False,
        use_sort : bool = False,
        title    : str = "Document",
    ) -> None:
        self.model     = model
        self.title     = title
        self._n_workers = n_workers
        self._worker_args = (
            model.model_config_path,
            model.model_path,
            model.imf_model_path,
            model.feature_set_name,
            model.input_type,
            use_gpu,
            use_sort,
        )
        self._pool: Optional[Pool] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        page_or_path      : Union[str, Path, Any],
        skip_header_footer: bool = True,
        progress_callback : Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Generate a complete HTML document from a single page or an entire PDF.

        Parameters
        ----------
        page_or_path       : PyMuPDF page object  -- processes that single page.
                             str or Path           -- treated as a PDF file path;
                                                      all pages are processed.
        skip_header_footer : If True, page-header and page-footer blocks are excluded.
        progress_callback  : Called as callback(completed_count, total_count).
                             Only used when processing a PDF file.

        Returns
        -------
        Complete HTML document string.
        """
        if not isinstance(page_or_path, (str, Path)):
            fragment = self.generate_page_fragment(
                page_or_path, skip_header_footer=skip_header_footer
            )
            return self._wrap_document([fragment])

        fragments = self._generate_pdf(
            str(page_or_path),
            skip_header_footer=skip_header_footer,
            progress_callback=progress_callback,
        )
        return self._wrap_document(fragments)

    def generate_page_fragment(
        self,
        page              : Any,
        skip_header_footer: bool = True,
    ) -> str:
        """
        Generate an HTML fragment (no <html>/<body> wrapper) for a single page.

        Returns
        -------
        str -- HTML fragment representing one page, wrapped in <div class="page">.
        """
        from ..pymupdf_util import create_input_data_from_page

        groups = self.model.predict(page, return_raw=True)
        if not groups:
            return '<div class="page"></div>'

        data_dict = create_input_data_from_page(page, options={
            "input_type"      : self.model.input_type,
            "feature_set_name": self.model.feature_set_name,
            "feature_extractor": self.model.feature_extractor,
        })
        all_texts : list[str] = data_dict.get("text", [])
        all_bboxes: list      = data_dict.get("bboxes", [])

        blocks: list[str] = []
        for group_idx, group in enumerate(groups):
            if skip_header_footer and group["class_name"] in _SKIP_CLASSES:
                continue
            block = self._render_group(group, group_idx, all_texts, all_bboxes)
            if block:
                blocks.append(block)

        inner = "\n".join(blocks)
        return f'<div class="page">\n{inner}\n</div>'

    # ------------------------------------------------------------------
    # PDF processing
    # ------------------------------------------------------------------

    def _generate_pdf(
        self,
        pdf_path          : str,
        skip_header_footer: bool,
        progress_callback : Optional[Callable[[int, int], None]],
    ) -> list[str]:
        import fitz
        doc     = fitz.open(pdf_path)
        n_pages = doc.page_count
        doc.close()

        args = [(pdf_path, i, skip_header_footer) for i in range(n_pages)]
        results: list[tuple[int, str]] = []

        if self._n_workers >= 2:
            pool = self._ensure_pool()
            for i, res in enumerate(pool.imap_unordered(_worker_generate_page, args), 1):
                results.append(res)
                if progress_callback:
                    progress_callback(i, n_pages)
        else:
            _worker_init(*self._worker_args)
            for i, arg in enumerate(args, 1):
                res = _worker_generate_page(arg)
                results.append(res)
                if progress_callback:
                    progress_callback(i, n_pages)

        results.sort(key=lambda x: x[0])
        return [fragment for _, fragment in results]

    def _ensure_pool(self) -> Pool:
        if self._pool is None:
            self._pool = Pool(
                processes   = self._n_workers,
                initializer = _worker_init,
                initargs    = self._worker_args,
            )
        return self._pool

    # ------------------------------------------------------------------
    # Document wrapper
    # ------------------------------------------------------------------

    def _wrap_document(self, page_fragments: list[str]) -> str:
        esc_title = html_module.escape(self.title)
        body      = "\n\n".join(page_fragments)
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"UTF-8\">\n"
            f"  <title>{esc_title}</title>\n"
            "  <style>\n"
            "    body { font-family: sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem; }\n"
            "    .page { margin-bottom: 3rem; border-bottom: 1px solid #ccc; padding-bottom: 2rem; }\n"
            "    h2 { margin-top: 1.5rem; }\n"
            "    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }\n"
            "    th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }\n"
            "    th { background: #f0f0f0; }\n"
            "    ul { margin: 0.5rem 0; padding-left: 1.5rem; }\n"
            "    .formula { font-style: italic; }\n"
            "    .figure { color: #666; font-size: 0.9em; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            f"{body}\n"
            "</body>\n"
            "</html>"
        )

    # ------------------------------------------------------------------
    # Group rendering
    # ------------------------------------------------------------------

    def _render_group(
        self,
        group     : dict,
        group_idx : int,
        all_texts : list[str],
        all_bboxes: list,
    ) -> str:
        cls  = group.get("class_name", "text")
        text = self._group_text(group, all_texts, all_bboxes).strip()

        if cls == "table":
            cells = group.get("table_cells", [])
            return self._render_table(cells) if cells else (
                f"<p>{html_module.escape(text)}</p>" if text else ""
            )

        if cls in _HEADING_CLASSES:
            return f"<h2>{html_module.escape(text)}</h2>" if text else ""

        if cls in _LIST_CLASSES:
            return f"<ul><li>{html_module.escape(text)}</li></ul>" if text else ""

        if cls in _ITALIC_CLASSES:
            return f"<p><em>{html_module.escape(text)}</em></p>" if text else ""

        if cls == "formula":
            return f'<p class="formula">{html_module.escape(text)}</p>' if text else ""

        if cls == "picture":
            x1, y1, x2, y2 = group["group_bbox"]
            label = f"Figure-{group_idx}"
            coords = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"
            return f'<p class="figure">[<span title="{coords}">{label}</span>]</p>'

        # text, paragraph, and any unknown classes
        return f"<p>{html_module.escape(text)}</p>" if text else ""

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_table(cells: list[Any]) -> str:
        """
        Renders table cells into an HTML table.
        Row 0 is treated as the header row (th), remaining rows as data rows (td).
        Merged cells are replicated across all spanned positions.
        """
        if not cells:
            return ""

        n_rows = max(c.row + (c.row_span if hasattr(c, "row_span") else 1) for c in cells)
        n_cols = max(c.col + (c.col_span if hasattr(c, "col_span") else 1) for c in cells)

        grid: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]

        for cell in cells:
            r_start = cell.row
            c_start = cell.col
            r_span  = getattr(cell, "row_span", 1)
            c_span  = getattr(cell, "col_span", 1)
            text    = cell.text.strip()
            if not text:
                continue
            for r in range(r_start, r_start + r_span):
                for c in range(c_start, c_start + c_span):
                    if 0 <= r < n_rows and 0 <= c < n_cols:
                        prev = grid[r][c]
                        grid[r][c] = (prev + " " + text).strip() if prev else text

        lines = ["<table>"]

        # Header row
        lines.append("  <tr>")
        for c in range(n_cols):
            lines.append(f"    <th>{html_module.escape(grid[0][c])}</th>")
        lines.append("  </tr>")

        # Data rows
        for r in range(1, n_rows):
            lines.append("  <tr>")
            for c in range(n_cols):
                lines.append(f"    <td>{html_module.escape(grid[r][c])}</td>")
            lines.append("  </tr>")

        lines.append("</table>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_text(group: dict, all_texts: list[str], all_bboxes: list) -> str:
        """Join texts sorted by visual reading order within the group."""
        margin   = 5
        indicies = group.get("indicies", [])
        valid = [
            (idx, all_bboxes[idx])
            for idx in indicies
            if 0 <= idx < len(all_texts) and 0 <= idx < len(all_bboxes)
               and all_texts[idx].strip()
        ]
        valid.sort(key=lambda x: (x[1][1] // margin, x[1][0]))
        return " ".join(all_texts[idx].strip() for idx, _ in valid)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
