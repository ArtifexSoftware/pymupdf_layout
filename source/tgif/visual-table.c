// Copyright (C) 2026 Artifex Software, Inc.
//
// This file is part of MuPDF.
//
// MuPDF is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// MuPDF is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
// details.
//
// You should have received a copy of the GNU Affero General Public License
// along with MuPDF. If not, see <https://www.gnu.org/licenses/agpl-3.0.en.html>
//
// Alternative licensing terms are available from the licensor.
// For commercial licensing, see <https://www.artifex.com/> or contact
// Artifex Software, Inc., 39 Mesa Street, Suite 108A, San Francisco,
// CA 94129, USA, for further information.

#include "mupdf/fitz.h"

#include "tgif_grid.h"
#include "tgif_image.h"
#include "tgif_model.h"
#include "tgif_postprocess.h"
#include "tgif_runtime.h"
#include "visual-table.h"

//#define DEBUG_VISUAL_GRID_FINDER

static void *fz_malloc_no_throw_wrap(void *opaque, size_t size)
{
    return fz_malloc_no_throw((fz_context *) opaque, size);
}

static void *fz_calloc_no_throw_wrap(void *opaque, size_t n, size_t size)
{
    return fz_calloc_no_throw((fz_context *) opaque, n, size);
}

static void fz_free_wrap(void *opaque, void *ptr)
{
    fz_free((fz_context *) opaque, ptr);
}

int
fz_visual_table_grid_finder(fz_context *ctx, fz_page *page, fz_rect bounds, fz_stext_grid_positions **xposp, fz_stext_grid_positions **yposp)
{
	fz_device *dev = NULL;
	fz_pixmap *pix = NULL;
	fz_matrix ctm;
	tgif_context tctx = { ctx, fz_malloc_no_throw_wrap, fz_calloc_no_throw_wrap, fz_free_wrap };
	tgif_image *tim;
	fz_path *path = NULL;
	tgif_runtime_output rt;
	tgif_grid *grid = NULL;
	int i;
	const int MODEL_W = 300;
	const int MODEL_H = 360;
	fz_stext_grid_positions *xs, *ys;
	int ret = 0;

	fz_var(dev);
	fz_var(page);
	fz_var(path);
	fz_var(grid);
	fz_var(ret);

	*xposp = NULL;
	*yposp = NULL;

	fz_try(ctx)
	{
		/* We want to render the given bounds into MODEL_W x MODEL_H */
		ctm.a = MODEL_W / (bounds.x1 - bounds.x0);
		ctm.b = 0;
		ctm.c = 0;
		ctm.d = MODEL_H / (bounds.y1 - bounds.y0);
		ctm.e = -bounds.x0 * ctm.a;
		ctm.f = -bounds.y0 * ctm.d;

		pix = fz_new_pixmap(ctx, fz_device_rgb(ctx), MODEL_W, MODEL_H, NULL, 0);
		fz_clear_pixmap(ctx, pix);
		dev = fz_new_draw_device(ctx, ctm, pix);
		fz_run_page_contents(ctx, page, dev, fz_identity, NULL);
		fz_close_device(ctx, dev);
		fz_drop_device(ctx, dev);
		dev = NULL;

		/* Convert pix to tgif_image. */
		tim = tgif_new_image_borrowed_data(&tctx, pix->w, pix->h, pix->n, pix->stride, pix->samples);
		if (tim == NULL)
			fz_throw(ctx, FZ_ERROR_SYSTEM, "Image conversion failed");

		/* Run the model */
		if (tgif_run_model(&tctx, &table_grid_model, tim, &rt) != 0) {
			fz_throw(ctx, FZ_ERROR_LIBRARY, "failed to run model");
		}

		grid = tgif_get_grid(&tctx, &rt);
		if (grid == NULL)
			break;

#ifdef DEBUG_VISUAL_GRID_FINDER
		{
			float blue[3] = { 0, 0, 1 };
			float red[3] = { 1, 0, 0 };
			/* Draw the results */
			dev = fz_new_draw_device(ctx, fz_identity, pix);
			path = fz_new_path(ctx);

			for (i = 0; i < grid->h_len; i++)
			{
				fz_moveto(ctx, path, 0.5f, grid->h[i]);
				fz_lineto(ctx, path, grid->v[0], grid->h[i]);
				fz_moveto(ctx, path, grid->v[grid->v_len-1], grid->h[i]);
				fz_lineto(ctx, path, MODEL_W-0.5f, grid->h[i]);
			}
			for (i = 0; i < grid->v_len; i++)
			{
				fz_moveto(ctx, path, grid->v[i], 0.5f);
				fz_lineto(ctx, path, grid->v[i], grid->h[0]);
				fz_moveto(ctx, path, grid->v[i], grid->h[grid->h_len-1]);
				fz_lineto(ctx, path, grid->v[i], MODEL_H-0.5f);
			}
			fz_rectto(ctx, path, 0.5f, 0.5f, MODEL_W-0.5f, MODEL_H-0.5f);
			fz_stroke_path(ctx, dev, path, &fz_default_stroke_state, fz_identity, fz_device_rgb(ctx), blue, 0.5, fz_default_color_params);
			fz_drop_path(ctx, path);
			path = NULL;
			path = fz_new_path(ctx);
			for (i = 0; i < grid->h_len; i++)
			{
				fz_moveto(ctx, path, grid->v[0], grid->h[i]);
				fz_lineto(ctx, path, grid->v[grid->v_len-1], grid->h[i]);
			}
			for (i = 0; i < grid->v_len; i++)
			{
				fz_moveto(ctx, path, grid->v[i], grid->h[0]);
				fz_lineto(ctx, path, grid->v[i], grid->h[grid->h_len-1]);
			}
			fz_stroke_path(ctx, dev, path, &fz_default_stroke_state, fz_identity, fz_device_rgb(ctx), red, 0.5, fz_default_color_params);
			fz_close_device(ctx, dev);
			fz_drop_device(ctx, dev);

			fz_save_pixmap_as_png(ctx, pix, "out.png");
		}
#endif
		/* Scale the grid back to full page coords */
		for (i = 0; i < grid->h_len; i++)
			grid->h[i] = (grid->h[i] - ctm.f) / ctm.d;
		for (i = 0; i < grid->v_len; i++)
			grid->v[i] = (grid->v[i] - ctm.e) / ctm.a;

		xs = *xposp = fz_malloc_flexible(ctx, fz_stext_grid_positions, list, grid->v_len+2);
		ys = *yposp = fz_malloc_flexible(ctx, fz_stext_grid_positions, list, grid->h_len+2);

		xs->len = grid->v_len+2;
		ys->len = grid->h_len+2;

		ys->list[0].min = bounds.y0;
		ys->list[0].pos = bounds.y0;
		ys->list[0].max = bounds.y0;
		for (i = 0; i < grid->h_len; i++)
		{
			ys->list[i+1].min = grid->h[i];
			ys->list[i+1].pos = grid->h[i];
			ys->list[i+1].max = grid->h[i];
		}
		ys->list[grid->h_len+1].min = bounds.y1;
		ys->list[grid->h_len+1].pos = bounds.y1;
		ys->list[grid->h_len+1].max = bounds.y1;
		xs->list[0].min = bounds.x0;
		xs->list[0].pos = bounds.x0;
		xs->list[0].max = bounds.x0;
		for (i = 0; i < grid->v_len; i++)
		{
			xs->list[i+1].min = grid->v[i];
			xs->list[i+1].pos = grid->v[i];
			xs->list[i+1].max = grid->v[i];
		}
		xs->list[grid->v_len+1].min = bounds.x1;
		xs->list[grid->v_len+1].pos = bounds.x1;
		xs->list[grid->v_len+1].max = bounds.x1;

		ret = 1;
	}
	fz_always(ctx)
	{
		tgif_runtime_output_destroy(&tctx, &rt);
		tgif_image_destroy(&tctx, tim);
		tgif_grid_destroy(&tctx, grid);
		fz_drop_path(ctx, path);
		fz_drop_pixmap(ctx, pix);
	}
	fz_catch(ctx)
	{
		fz_free(ctx, *xposp);
		fz_free(ctx, *yposp);
		*xposp = NULL;
		*yposp = NULL;
		fz_rethrow(ctx);
	}

	return ret;
}

fz_stext_block *
fz_find_visual_table_within_bounds(fz_context *ctx, fz_stext_page *stext, fz_page *page, fz_rect bounds)
{
	fz_stext_block *ret;
	fz_stext_grid_positions *xpos = NULL;
	fz_stext_grid_positions *ypos = NULL;

	if (!fz_visual_table_grid_finder(ctx, page, bounds, &xpos, &ypos))
		return NULL;

	fz_try(ctx)
		ret = fz_find_table_within_grid(ctx, stext, xpos, ypos, 999999, NULL);
	fz_always(ctx)
	{
		fz_free(ctx, xpos);
		fz_free(ctx, ypos);
	}
	fz_catch(ctx)
		fz_rethrow(ctx);

	return ret;
}
