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

#define DEBUG_VISUAL_GRID_FINDER

enum { MODEL_W = 300, MODEL_H = 360 };

static void
make_heatmap(fz_pixmap *pix, float *h, float *v)
{
	const uint8_t *data;
	int x, y;
	ptrdiff_t stride = pix->stride;

	/* Calculate vertical sums. */
	data = pix->samples;
	for (y = MODEL_H; y > 0; y--)
	{
		uint32_t sum = 0;
		for (x = MODEL_W; x > 0; x--)
		{
			sum += 255-*data++;
		}
		*h++ = sum / (float)(255 * MODEL_W);
	}
	h -= MODEL_H;

	/* Calculate horizontal sums. */
	data = pix->samples;
	for (x = MODEL_W; x > 0; x--)
	{
		uint32_t sum = 0;
		for (y = MODEL_H; y > 0; y--)
		{
			sum += 255-*data;
			data += stride;
		}
		*v++ = sum / (float)(255 * MODEL_H);
		data += 1 - stride * MODEL_H;
	}
	v -= MODEL_W;
}

static void
blur1d(float *h, int n)
{
	int i;
	float prev;

	prev = h[0];
	for (i = n-1; i > 0; i--)
	{
		float f = *h;
		*h = (prev + 2*f + h[1]) / 4.0f;
		h++;
		prev = f;
	}
	*h = ((*h * 3) + prev) / 4.0f;
}

static void
scale1d(float *h, int n)
{
	int i;
	float min, max, avg;

	avg = 0;
	min = max = *h++;
	for (i = n-1; i > 0; i--)
	{
		float f = *h++;
		avg += f;
		if (f < min)
			min = f;
		else if (f > max)
			max = f;
	}
	h -= n;
	avg /= n;

	if (min == 0 && max == 1)
		return;
	if (min == max)
	{
		memset(h, 0, sizeof(float) * n);
		return;
	}

	for (i = n; i > 0; i--)
	{
		float f = *h;
		if (f < avg)
		{
			f = (f - min) * 0.5f / (avg-min);
		}
		else
		{
			f = (f - avg) * 0.5f / (max-avg) + 0.5f;
		}
		//*h = (*h - min) / (max-min);
		*h++ = f;
	}
}

static int
extract_heatmap(const float *h, int n, float threshold, fz_stext_grid_divider *div)
{
	int count = 0;
	int i;
	int start, in_div;

	in_div = 0;
	if (h[0] >= threshold)
	{
		/* Add an entry at the edge */
		count++;
		if (div)
		{
			div->min = 0;
			div->pos = 0;
			div->max = 0;
			div++;
		}
	}

	start = 0;
	for (i = 0; i < n; i++)
	{
		if (h[i] < threshold)
		{
			if (!in_div)
			{
				/* Starting a division */
				start = i;
				in_div = 1;
			}
		}
		else if (in_div)
		{
			/* Stopping a division */
			in_div = 0;
			count++;
			if (div)
			{
				div->min = start;
				div->max = i-1;
				div->pos = (div->min + div->max) / 2;
				div++;
			}
		}
	}

	/* If we're in a division, we want to end it. If we're not we
	 * want to make one at the edge! */
	count++;
	if (div)
	{
		div->min = n;
		div->pos = n;
		div->max = n;
		div++;
	}

	return count;
}

int
fz_visual_table_grid_finder2(fz_context *ctx, fz_page *page, fz_rect bounds, fz_stext_grid_positions **xposp, fz_stext_grid_positions **yposp)
{
	fz_device *dev = NULL;
	fz_pixmap *pix = NULL;
	fz_pixmap *pix2 = NULL;
	fz_matrix ctm;
	fz_path *path = NULL;
	fz_stext_grid_positions *xs;
	fz_stext_grid_positions *ys;
	int i, xlen, ylen;
	int ret = 0;

	fz_var(dev);
	fz_var(page);
	fz_var(path);
	fz_var(pix);
	fz_var(ret);

	*xposp = NULL;
	*yposp = NULL;

	fz_try(ctx)
	{
		float h[MODEL_H], v[MODEL_W];

		/* We want to render the given bounds into MODEL_W x MODEL_H */
		ctm.a = MODEL_W / (bounds.x1 - bounds.x0);
		ctm.b = 0;
		ctm.c = 0;
		ctm.d = MODEL_H / (bounds.y1 - bounds.y0);
		ctm.e = -bounds.x0 * ctm.a;
		ctm.f = -bounds.y0 * ctm.d;

		pix = fz_new_pixmap(ctx, fz_device_gray(ctx), MODEL_W, MODEL_H, NULL, 0);
		fz_clear_pixmap(ctx, pix);
		dev = fz_new_draw_device(ctx, ctm, pix);
		fz_run_page_contents(ctx, page, dev, fz_identity, NULL);
		fz_close_device(ctx, dev);
		fz_drop_device(ctx, dev);
		dev = NULL;

		/* Make heatmap */
		make_heatmap(pix, h, v);
		blur1d(h, MODEL_H);
		blur1d(h, MODEL_H);
		blur1d(h, MODEL_H);
		blur1d(v, MODEL_W);
		blur1d(v, MODEL_W);
		blur1d(v, MODEL_W);
		blur1d(v, MODEL_W);
		blur1d(v, MODEL_W);
		scale1d(h, MODEL_H);
		scale1d(v, MODEL_W);

		/* get_grid */
		ylen = extract_heatmap(h, MODEL_H, 0.5f, NULL);
		xlen = extract_heatmap(v, MODEL_W, 0.5f, NULL);
		xs = *xposp = fz_malloc_flexible(ctx, fz_stext_grid_positions, list, xlen+2);
		ys = *yposp = fz_malloc_flexible(ctx, fz_stext_grid_positions, list, ylen+2);
		xs->len = xlen;
		ys->len = ylen;
		extract_heatmap(h, MODEL_H, 0.5f, ys->list);
		extract_heatmap(v, MODEL_W, 0.5f, xs->list);

#ifdef DEBUG_VISUAL_GRID_FINDER
		{
			int j;
			float red[3] = { 1, 0, 0 };
			pix2 = fz_new_pixmap(ctx, fz_device_rgb(ctx), MODEL_W*2, MODEL_H*2, NULL, 0);
			fz_clear_pixmap(ctx, pix2);
			dev = fz_new_draw_device(ctx, ctm, pix2);
			fz_run_page_contents(ctx, page, dev, fz_identity, NULL);
			fz_close_device(ctx, dev);
			fz_drop_device(ctx, dev);
			dev = NULL;
			/* Draw the results */
			dev = fz_new_draw_device(ctx, fz_identity, pix2);
			path = fz_new_path(ctx);

			for (i = 0; i < ys->len; i++)
			{
				fz_moveto(ctx, path, xs->list[0].pos, ys->list[i].pos);
				fz_lineto(ctx, path, xs->list[xs->len-1].pos, ys->list[i].pos);
			}
			for (i = 0; i < xs->len; i++)
			{
				fz_moveto(ctx, path, xs->list[i].pos, ys->list[0].pos);
				fz_lineto(ctx, path, xs->list[i].pos, ys->list[ys->len-1].pos);
			}
			fz_stroke_path(ctx, dev, path, &fz_default_stroke_state, fz_identity, fz_device_rgb(ctx), red, 0.5, fz_default_color_params);
			fz_drop_path(ctx, path);
			path = NULL;
			fz_close_device(ctx, dev);
			fz_drop_device(ctx, dev);

			/* show H and V heatmaps */
			for (i = 0; i < MODEL_W; i++)
			{
				uint8_t c = (int)(v[i]*255+0.5f);
				for (j = 0; j < MODEL_H; j++)
				{
					pix2->samples[i*3 + (MODEL_H+j)*pix2->stride    ] = c;
					pix2->samples[i*3 + (MODEL_H+j)*pix2->stride + 1] = 0;
					pix2->samples[i*3 + (MODEL_H+j)*pix2->stride + 2] = 0;
				}
			}
			for (i = 0; i < MODEL_H; i++)
			{
				uint8_t c = (int)(h[i]*255+0.5f);
				for (j = 0; j < MODEL_W; j++)
				{
					pix2->samples[i*pix2->stride + (MODEL_W+j)*3    ] = c;
					pix2->samples[i*pix2->stride + (MODEL_W+j)*3 + 1] = 0;
					pix2->samples[i*pix2->stride + (MODEL_W+j)*3 + 2] = 0;
				}
			}


			fz_save_pixmap_as_png(ctx, pix2, "out.png");
		}
#endif
		/* Scale the grid back to full page coords */
		for (i = 0; i < xs->len; i++)
		{
			xs->list[i].min = (xs->list[i].min - ctm.f) / ctm.d;
			xs->list[i].pos = (xs->list[i].pos - ctm.f) / ctm.d;
			xs->list[i].max = (xs->list[i].max - ctm.f) / ctm.d;
		}
		for (i = 0; i < ys->len; i++)
		{
			ys->list[i].min = (ys->list[i].min - ctm.e) / ctm.a;
			ys->list[i].pos = (ys->list[i].pos - ctm.e) / ctm.a;
			ys->list[i].max = (ys->list[i].max - ctm.e) / ctm.a;
		}

		ret = 1;
	}
	fz_always(ctx)
	{
		fz_drop_path(ctx, path);
		fz_drop_page(ctx, page);
		fz_drop_pixmap(ctx, pix);
		fz_drop_pixmap(ctx, pix2);
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
fz_find_visual_table_within_bounds2(fz_context *ctx, fz_stext_page *stext, fz_page *page, fz_rect bounds)
{
	fz_stext_block *ret;
	fz_stext_grid_positions *xpos, *ypos;

	if (!fz_visual_table_grid_finder2(ctx, page, bounds, &xpos, &ypos))
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
