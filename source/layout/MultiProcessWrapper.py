from __future__ import annotations

import multiprocessing as mp
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional, Union, Callable

# ---------------------------------------------------------------------------
# Worker-side globals (one per worker process)
# ---------------------------------------------------------------------------

_worker_model = None


def _worker_init(
    config_path: str,
    model_path: str,
    imf_model_path: str,
    feature_set_name: str,
    input_type: Optional[tuple],
    use_gpu: bool,
    use_sort: bool,
    table_grid_model_ver: str,
) -> None:
    """Initialize BoxRFDGNN once per worker process."""
    global _worker_model
    from .onnx.BoxRFDGNN import BoxRFDGNN
    _worker_model = BoxRFDGNN(
        config_path            = config_path,
        model_path             = model_path,
        imf_model_path         = imf_model_path,
        feature_set_name       = feature_set_name,
        input_type             = input_type,
        enable_inference_cache = False,
        use_gpu                = use_gpu,
        use_sort               = use_sort,
        table_grid_model_ver   = table_grid_model_ver,
    )


def _worker_to_markdown(args: tuple) -> tuple[int, str]:
    """
    Worker task: open PDF, process one page, return (page_no, markdown).

    Parameters
    ----------
    args : (pdf_path, page_no, join, skip_header_footer)
    """
    pdf_path, page_no, join, skip_header_footer = args
    import fitz
    doc  = fitz.open(pdf_path)
    page = doc[page_no]
    md   = _worker_model.to_markdown(page, join=join, skip_header_footer=skip_header_footer)
    doc.close()
    return page_no, md


def _worker_to_markdown_html_table(args: tuple) -> tuple[int, str]:
    """
    Worker task: open PDF, process one page, return (page_no, markdown with HTML tables).

    Parameters
    ----------
    args : (pdf_path, page_no, join, skip_header_footer)
    """
    pdf_path, page_no, join, skip_header_footer = args
    import fitz
    doc  = fitz.open(pdf_path)
    page = doc[page_no]
    md   = _worker_model.to_markdown_html_table(page, join=join, skip_header_footer=skip_header_footer)
    doc.close()
    return page_no, md


def _worker_to_result(args: tuple) -> tuple[int, list, str]:
    """
    Worker task: open PDF, process one page, return (page_no, layout, markdown).

    predict() is called first so its result is cached internally,
    allowing to_markdown() to reuse it without redundant computation.

    Parameters
    ----------
    args : (pdf_path, page_no, join, skip_header_footer)

    Returns
    -------
    (page_no, layout, markdown)
        layout   : list of [x1, y1, x2, y2, class_name] per detected box
        markdown : markdown text for the page
    """
    pdf_path, page_no, join, skip_header_footer = args
    import fitz
    doc    = fitz.open(pdf_path)
    page   = doc[page_no]
    layout = _worker_model.predict(page)
    md     = _worker_model.to_markdown(page, join=join, skip_header_footer=skip_header_footer)
    doc.close()
    return page_no, layout, md


# ---------------------------------------------------------------------------
# MultiProcessWrapper
# ---------------------------------------------------------------------------

class MultiProcessWrapper:
    """
    Wraps BoxRFDGNN for both single-page and whole-PDF parallel processing.
    """

    def __init__(
        self,
        config_path          : Optional[str] = None,
        model_path           : Optional[str] = None,
        imf_model_path       : Optional[str] = None,
        feature_set_name     : str = 'imf+rf',
        input_type           : Optional[tuple] = None,
        n_workers            : int = 1,
        use_gpu              : bool = False,
        use_sort             : bool = False,
        table_grid_model_ver : str = 'V4',
    ) -> None:
        from .onnx.BoxRFDGNN import BoxRFDGNN

        # Main-process model for single-page delegation
        self._model = BoxRFDGNN(
            config_path            = config_path,
            model_path             = model_path,
            imf_model_path         = imf_model_path,
            feature_set_name       = feature_set_name,
            input_type             = input_type,
            enable_inference_cache = True,
            use_gpu                = use_gpu,
            use_sort               = use_sort,
            table_grid_model_ver   = table_grid_model_ver,
        )

        self._n_workers = n_workers
        self._worker_args = (
            # self._model.model_config_path,
            self._model.model_config_path,
            self._model.model_path,
            self._model.imf_model_path,
            self._model.feature_set_name,
            self._model.input_type,
            use_gpu,
            use_sort,
            table_grid_model_ver,
        )

        self._pool: Optional[Pool] = None

    def is_image_page(self, page) -> bool:
        return self._model.is_image_page(page)

    def is_page_ocr_needed(self, page, data_dict=None,
                           iou_threshold: float = 0.2) -> "bool | None":
        return self._model.is_page_ocr_needed(page, data_dict, iou_threshold)

    def predict(self, page, data_dict=None, **kwargs):
        return self._model.predict(page, data_dict=data_dict, **kwargs)

    def to_markdown(
        self,
        page_or_path: Union[str, Path, object],
        join: bool = True,
        skip_header_footer: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Union[list[str], str]:
        """
        Convert a page or an entire PDF to Markdown with progress tracking.

        Parameters
        ----------
        page_or_path : PyMuPDF page object or PDF file path.
        join         : If True, returns a single string joined by separators.
        skip_header_footer : If True, page-header and page-footer blocks are excluded.
        progress_callback : Called as callback(completed_count, total_count).
        """
        if not isinstance(page_or_path, (str, Path)):
            return self._model.to_markdown(page_or_path, join=join, skip_header_footer=skip_header_footer)

        return self._to_markdown_pdf(
            str(page_or_path),
            join=join,
            skip_header_footer=skip_header_footer,
            progress_callback=progress_callback,
        )

    def to_markdown_html_table(
        self,
        page_or_path      : Union[str, Path, object],
        join              : bool = True,
        skip_header_footer: bool = True,
        progress_callback : Optional[Callable[[int, int], None]] = None,
    ) -> Union[list[str], str]:
        """
        Convert a page or an entire PDF to Markdown with HTML tables.

        Parameters
        ----------
        page_or_path       : PyMuPDF page object or PDF file path.
        join               : If True, returns a single string joined by separators.
        skip_header_footer : If True, page-header and page-footer blocks are excluded.
        progress_callback  : Called as callback(completed_count, total_count).
        """
        if not isinstance(page_or_path, (str, Path)):
            return self._model.to_markdown_html_table(
                page_or_path, join=join, skip_header_footer=skip_header_footer
            )

        return self._to_markdown_html_table_pdf(
            str(page_or_path),
            join=join,
            skip_header_footer=skip_header_footer,
            progress_callback=progress_callback,
        )

    def to_html(
        self,
        pdf_path          : Union[str, Path],
        skip_header_footer: bool = True,
        progress_callback : Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Convert an entire PDF to a complete HTML document.

        Parameters
        ----------
        pdf_path           : Path to the PDF file.
        skip_header_footer : If True, page-header and page-footer blocks are excluded.
        progress_callback  : Called as callback(completed_count, total_count).
        """
        return self._model.html_generator.generate(
            str(pdf_path),
            skip_header_footer=skip_header_footer,
            progress_callback=progress_callback,
        )

    def get_result(
        self,
        pdf_path: Union[str, Path],
        join: bool = True,
        skip_header_footer: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """
        Process an entire PDF and return layout and markdown for all pages.

        predict() is called before to_markdown() on each page so that the
        internal inference cache is populated and reused, avoiding duplicate
        computation.

        Parameters
        ----------
        pdf_path : Path to the PDF file.
        join : If True, result['markdown'] is list[str] where each str is the
               joined markdown of all elements on that page.
               If False, result['markdown'] is list[list[str]] where each inner
               list contains one markdown string per layout element.
        skip_header_footer : If True, page-header and page-footer blocks are excluded.
        progress_callback : Called as callback(completed_count, total_count).

        Returns
        -------
        dict with keys:
            'layout'   : list[list]        -- per-page list of
                         [x1, y1, x2, y2, class_name] boxes
            'markdown' : list[str]         -- when join=True
                       : list[list[str]]   -- when join=False
        """
        import fitz
        pdf_path = str(pdf_path)
        doc      = fitz.open(pdf_path)
        n_pages  = doc.page_count
        doc.close()

        results: list[tuple[int, list, str]] = []

        if self._n_workers >= 2:
            args = [(pdf_path, i, join, skip_header_footer) for i in range(n_pages)]
            pool = self._ensure_pool()
            for i, res in enumerate(
                pool.imap_unordered(_worker_to_result, args), 1
            ):
                results.append(res)
                if progress_callback:
                    progress_callback(i, n_pages)
        else:
            # Use the already-initialised main-process model directly.
            # Calling _worker_init here would create a redundant ONNX session
            # on every PDF, causing RSS to grow unboundedly across documents.
            import fitz as _fitz
            doc = _fitz.open(pdf_path)
            for page_no in range(n_pages):
                page   = doc[page_no]
                layout = self._model.predict(page)
                md     = self._model.to_markdown(page, join=join, skip_header_footer=skip_header_footer)
                results.append((page_no, layout, md))
                if progress_callback:
                    progress_callback(page_no + 1, n_pages)
            doc.close()

        results.sort(key=lambda x: x[0])

        return {
            'layout'  : [layout for _, layout, _ in results],
            'markdown': [md     for _, _,      md in results],
        }

    def _ensure_pool(self) -> Pool:
        if self._pool is None:
            self._pool = Pool(
                processes   = self._n_workers,
                initializer = _worker_init,
                initargs    = self._worker_args,
            )
        return self._pool

    def _to_markdown_pdf(
        self,
        pdf_path: str,
        join: bool,
        skip_header_footer: bool,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Union[list[str], str]:
        import fitz
        doc     = fitz.open(pdf_path)
        n_pages = doc.page_count
        doc.close()

        results: list[tuple[int, str]] = []

        if self._n_workers >= 2:
            args = [(pdf_path, i, join, skip_header_footer) for i in range(n_pages)]
            pool = self._ensure_pool()
            for i, res in enumerate(
                pool.imap_unordered(_worker_to_markdown, args), 1
            ):
                results.append(res)
                if progress_callback:
                    progress_callback(i, n_pages)
        else:
            # Use the already-initialised main-process model directly.
            # Calling _worker_init here would create a redundant ONNX session
            # on every PDF, causing RSS to grow unboundedly across documents.
            doc = fitz.open(pdf_path)
            for page_no in range(n_pages):
                page = doc[page_no]
                md   = self._model.to_markdown(page, join=join, skip_header_footer=skip_header_footer)
                results.append((page_no, md))
                if progress_callback:
                    progress_callback(page_no + 1, n_pages)
            doc.close()

        results.sort(key=lambda x: x[0])
        pages_md = [md for _, md in results]

        if join:
            return "\n\n---\n\n".join(pages_md)
        return pages_md

    def _to_markdown_html_table_pdf(
        self,
        pdf_path          : str,
        join              : bool,
        skip_header_footer: bool,
        progress_callback : Optional[Callable[[int, int], None]] = None,
    ) -> Union[list[str], str]:
        import fitz
        doc     = fitz.open(pdf_path)
        n_pages = doc.page_count
        doc.close()

        results: list[tuple[int, str]] = []

        if self._n_workers >= 2:
            args = [(pdf_path, i, join, skip_header_footer) for i in range(n_pages)]
            pool = self._ensure_pool()
            for i, res in enumerate(
                pool.imap_unordered(_worker_to_markdown_html_table, args), 1
            ):
                results.append(res)
                if progress_callback:
                    progress_callback(i, n_pages)
        else:
            # Use the already-initialised main-process model directly.
            # Calling _worker_init here would create a redundant ONNX session
            # on every PDF, causing RSS to grow unboundedly across documents.
            doc = fitz.open(pdf_path)
            for page_no in range(n_pages):
                page = doc[page_no]
                md   = self._model.to_markdown_html_table(page, join=join, skip_header_footer=skip_header_footer)
                results.append((page_no, md))
                if progress_callback:
                    progress_callback(page_no + 1, n_pages)
            doc.close()

        results.sort(key=lambda x: x[0])
        pages_md = [md for _, md in results]

        if join:
            return "\n\n---\n\n".join(pages_md)
        return pages_md

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
