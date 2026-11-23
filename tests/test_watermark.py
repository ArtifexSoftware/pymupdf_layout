import numpy as np

import pymupdf
import pymupdf.layout
from pymupdf.layout import pymupdf_util
from pymupdf.layout import pymupdf_util_watermark


GREY = (0.5, 0.5, 0.5)


def _make_page(watermark=None, extra=None, rotation=0):
    '''
    Return a one-page document with a heading, a fully ruled 14x4 table and
    a footer.

    watermark: text over the page content:
        'diagonal'      two large translucent lines at 55 degrees
        'tiled'         a small translucent word, repeated all over the page
        'horizontal'    two large translucent horizontal lines
        'outlined'      as 'diagonal', drawn as outlines
        'grey'          as 'diagonal', opaque light grey
    extra: other text:
        'vertical_header'       90 degree headers inside the table
        'slanted_header'        45 degree body-size headers above the table
        'translucent_heading'   a translucent word next to the heading
        'grey_title'            a large light grey word below the table
        'twin'                  a line that looks like the 'horizontal'
                                watermark, on free space at the bottom
        'crossing'              an opaque line of the size of the 'horizontal'
                                watermark, at the place of its second line
    '''
    document = pymupdf.open()
    page = document.new_page(width=612, height=912 if extra == 'twin' else 792)
    page.insert_text((72, 60), 'Section 4.2 Torque specifications', fontsize=14, fontname='hebo')
    page.insert_text((72, 76), 'Tighten all fasteners to the values listed below.', fontsize=9)
    table = pymupdf.Rect(89, 87, 561, 699)
    rows, cols = 14, 4
    rh = table.height / rows
    cw = table.width / cols
    shape = page.new_shape()
    for r in range(rows + 1):
        shape.draw_line((table.x0, table.y0 + r * rh), (table.x1, table.y0 + r * rh))
    for c in range(cols + 1):
        shape.draw_line((table.x0 + c * cw, table.y0), (table.x0 + c * cw, table.y1))
    shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()
    heads = ['Item', 'Part number', 'Torque (Nm)', 'Remarks']
    for r in range(rows):
        for c in range(cols):
            cell = pymupdf.Rect(table.x0 + c * cw, table.y0 + r * rh,
                                table.x0 + (c + 1) * cw, table.y0 + (r + 1) * rh)
            if r == 0:
                if extra == 'vertical_header':
                    page.insert_textbox(cell + (4, 2, -4, -2), heads[c], fontsize=8,
                                        fontname='hebo', rotate=90)
                else:
                    page.insert_text((cell.x0 + 4, cell.y0 + 16), heads[c], fontsize=9, fontname='hebo')
            else:
                values = [str(r), f'A{1000 + 37 * r}-{r % 7}', f'{12 + 3.5 * r:.1f}',
                          'replace seal' if r % 3 == 0 else 'check']
                page.insert_text((cell.x0 + 4, cell.y0 + 16), values[c], fontsize=9)
            if extra == 'slanted_header' and r == 0:
                p = pymupdf.Point(cell.x0 + 20, table.y0 - 2)
                page.insert_text(p, heads[c], fontsize=9, morph=(p, pymupdf.Matrix(45)))
    page.insert_text((72, 740), 'Service manual  rev. 7  page 506', fontsize=8)
    if extra == 'translucent_heading':
        page.insert_text((400, 60), 'Rev. 7', fontsize=20, color=(0, 0, 0.6), fill_opacity=0.6)
    elif extra == 'grey_title':
        page.insert_text((150, 770), 'APPENDIX', fontsize=40, color=(0.8, 0.8, 0.8))
    elif extra == 'twin':
        page.insert_text((80, 880), 'APPENDIX', fontsize=70, color=GREY, fill_opacity=0.25)
    elif extra == 'crossing':
        page.insert_text((150, 560), 'Part B', fontsize=70)

    if watermark in ('diagonal', 'outlined', 'grey'):
        kwargs = dict(fontsize=110, color=GREY, fill_opacity=0.25)
        if watermark == 'outlined':
            kwargs = dict(fontsize=110, color=GREY, stroke_opacity=0.25, render_mode=1)
        elif watermark == 'grey':
            kwargs = dict(fontsize=110, color=(0.85, 0.85, 0.85))
        for p in (pymupdf.Point(150, 420), pymupdf.Point(330, 620)):
            page.insert_text(p, 'CONFIDENTIAL', morph=(p, pymupdf.Matrix(55)), **kwargs)
    elif watermark == 'tiled':
        for y in range(80, 780, 90):
            for x in range(30, 600, 130):
                p = pymupdf.Point(x, y)
                page.insert_text(p, 'DRAFT', fontsize=20, color=(0.6, 0.6, 0.6),
                                 fill_opacity=0.3, morph=(p, pymupdf.Matrix(30)))
    elif watermark == 'horizontal':
        for y in (330, 560):
            page.insert_text((80, y), 'CONFIDENTIAL', fontsize=70, color=GREY, fill_opacity=0.25)
    else:
        assert watermark is None, f'{watermark=}'
    if rotation:
        page.set_rotation(rotation)
    return pymupdf.open('pdf', document.tobytes())


def _input(page, **options):
    return pymupdf_util.create_input_data_from_page(page, options=dict(feature_set_name='', **options))


def _text(data_dict):
    return data_dict['stext_page'].extractDICT()


def _same_input(data_dict, expected):
    return (data_dict['bboxes'] == expected['bboxes']
            and _text(data_dict) == _text(expected)
            and np.array_equal(data_dict['image'], expected['image']))


def test_watermark_removed_from_input():
    clean = _input(_make_page()[0])
    for watermark in 'diagonal', 'tiled', 'horizontal', 'outlined', 'grey':
        document = _make_page(watermark)
        page = document[0]
        contents = page.read_contents()
        # The watermark is in all inputs unless removal is switched off ...
        kept = _input(page, remove_watermarks=False)
        assert len(kept['bboxes']) > len(clean['bboxes']), f'{watermark=}'
        assert not np.array_equal(kept['image'], clean['image']), f'{watermark=}'
        # ... and with removal the inputs are exactly those of the clean page.
        assert _same_input(_input(page), clean), f'{watermark=}'
        assert _same_input(_input(page, remove_watermarks=True), clean), f'{watermark=}'
        # The page itself is unchanged.
        assert page.read_contents() == contents
        assert _same_input(_input(page, remove_watermarks=False), kept)

    # Module-level switch, used when no keyword is given.
    page = _make_page('diagonal')[0]
    pymupdf_util_watermark.REMOVE_WATERMARKS = False
    try:
        assert len(_input(page)['bboxes']) > len(clean['bboxes'])
    finally:
        pymupdf_util_watermark.REMOVE_WATERMARKS = True


def test_watermark_rotated_page():
    # Text is extracted from the unrotated page and the image shows the
    # rotated one. A user unit scales the page.
    for rotation, user_unit in (90, None), (180, None), (270, None), (90, 2):
        documents = [_make_page(rotation=rotation), _make_page('diagonal', rotation=rotation)]
        if user_unit:
            for document in documents:
                document.xref_set_key(document[0].xref, 'UserUnit', str(user_unit))
            documents = [pymupdf.open('pdf', document.tobytes()) for document in documents]
        clean = _input(documents[0][0])
        page = documents[1][0]
        assert not _same_input(_input(page, remove_watermarks=False), clean)
        assert _same_input(_input(page), clean), f'{rotation=} {user_unit=}'
        assert page.rotation == rotation


def test_watermark_other_text_stays():
    # Text that is no watermark stays: rotated headers, translucent or light
    # text that does not lie over other text.
    for extra in 'vertical_header', 'slanted_header', 'translucent_heading', 'grey_title':
        page = _make_page(extra=extra)[0]
        assert _same_input(_input(page), _input(page, remove_watermarks=False)), f'{extra=}'

    # So does text that looks like the watermark but lies elsewhere, and
    # text of the same size across the watermark.
    for extra in 'twin', 'crossing':
        expected = _input(_make_page(extra=extra)[0])
        assert _same_input(_input(_make_page('horizontal', extra=extra)[0]), expected), f'{extra=}'

    # A page with little other text: the large translucent text is the content.
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((60, 300), 'ANNUAL REPORT', fontsize=60, fill_opacity=0.5)
    page.insert_text((70, 290), '2026', fontsize=12)
    page = pymupdf.open('pdf', document.tobytes())[0]
    assert _same_input(_input(page), _input(page, remove_watermarks=False))


def test_watermark_removal_fails():
    # The prediction goes on with the page as it is if the removal fails,
    # also on a rotated page.
    page = _make_page('diagonal', rotation=90)[0]
    kept = _input(page, remove_watermarks=False)

    def fail(text, ctm, alpha):
        raise RuntimeError('test')

    original = pymupdf_util_watermark._SkipLines._is_marked
    pymupdf_util_watermark._SkipLines._is_marked = staticmethod(fail)
    try:
        assert _same_input(_input(page), kept)
    finally:
        pymupdf_util_watermark._SkipLines._is_marked = original
    assert page.rotation == 90
    assert not _same_input(_input(page), kept)


def test_watermark_layout():
    # Two large rotated watermark lines over the page swallow heading and
    # text into one page-sized table when they are model input. Without
    # them the layout is that of the clean page.
    clean_document = _make_page()
    marked_document = _make_page('diagonal')
    clean = clean_document[0]
    marked = marked_document[0]
    clean.get_layout()
    marked.get_layout()
    print(f'{clean.layout_information=}')
    print(f'{marked.layout_information=}')
    assert marked.layout_information == clean.layout_information
