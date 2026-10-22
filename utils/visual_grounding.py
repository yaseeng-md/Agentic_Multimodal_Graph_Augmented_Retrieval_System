"""
AMG Multimodal RAG — Visual Grounding and Bounding Box Extraction

Derives post-retrieval patch-level visual grounding from ColPali multi-vector
late interaction. Maps query token activations against the 32x32 visual patch
grid of retrieved PDF pages, extracts focal bounding boxes, and generates
visual highlight overlays to guide multimodal LLM generation (Qwen-VL).

Includes debug visualization combining the ColPali patch heatmap and page image.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ===========================================================================
# Debug Visualization Configuration
# (Configure directly in this file; do NOT add to .env)
# ===========================================================================
VISUALIZE: bool = True  # Set to True to generate combined heatmap + page debug images
VISUALIZATION_DIR: Path = Path("data/visualizations")
VISUALIZATION_ALPHA: float = 0.45  # Heatmap opacity when blended over page

# Default highlight styling
DEFAULT_BORDER_COLOR = (230, 50, 50, 220)       # Translucent bold red outline
DEFAULT_FILL_COLOR = (255, 230, 0, 45)           # Translucent amber/yellow focus wash
DEFAULT_CORNER_COLOR = (200, 0, 0, 255)          # High-contrast solid red reticle corners
DEFAULT_BORDER_WIDTH = 4


# ---------------------------------------------------------------------------
# Colormap LUT (Pure NumPy implementation; avoids matplotlib/cv2 dependencies)
# ---------------------------------------------------------------------------
def _generate_jet_lut() -> np.ndarray:
    """
    Returns a 256x3 RGB lookup table for the Jet colormap.
    Transitions: Blue (low) -> Cyan -> Green -> Yellow -> Red (peak).
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut


_JET_LUT: np.ndarray | None = None


def get_jet_lut() -> np.ndarray:
    """Get or build process-cached 256-entry Jet RGB lookup table."""
    global _JET_LUT
    if _JET_LUT is None:
        _JET_LUT = _generate_jet_lut()
    return _JET_LUT


# ---------------------------------------------------------------------------
# Combined Heatmap & Page Visualization
# ---------------------------------------------------------------------------
def visualize_heatmap_and_page(
    page_image: Image.Image | str | Path,
    heatmap: np.ndarray,
    bbox_norm: Sequence[float],
    query_text: str = "",
    stats: dict[str, float] | None = None,
    output_path: str | Path | None = None,
    alpha: float = VISUALIZATION_ALPHA,
) -> str:
    """
    Generates a combined side-by-side debug visualization of:
      1. Original Page with visual focus highlight and reticle brackets.
      2. 32x32 ColPali patch similarity heatmap smoothly blended over the page.
      3. Diagnostic header with query string, coordinates, and activation metrics.
      4. Colorbar gradient key for heatmap intensity.

    Parameters
    ----------
    page_image:
        Path to page image or existing PIL Image.
    heatmap:
        (32, 32) similarity matrix or activation map.
    bbox_norm:
        [ymin, xmin, ymax, xmax] normalized bounding box coordinates.
    query_text:
        Natural language query string for debug annotation.
    stats:
        Dictionary of activation statistics (min, max, mean, contrast, confidence).
    output_path:
        Optional target file path. If None, saves to VISUALIZATION_DIR.
    alpha:
        Heatmap overlay opacity in [0.0, 1.0].

    Returns
    -------
    str
        Path string to the saved combined debug image.
    """
    # 1. Resolve page image
    if isinstance(page_image, (str, Path)):
        img_path = Path(page_image).expanduser().resolve()
        if not img_path.is_file():
            raise FileNotFoundError(f"Page image not found: {img_path}")
        stem = img_path.stem
        orig_img = Image.open(img_path).convert("RGB")
    elif isinstance(page_image, Image.Image):
        stem = "page"
        orig_img = page_image.convert("RGB")
    else:
        raise TypeError(f"Unsupported page_image type: {type(page_image)}")

    width, height = orig_img.size

    # 2. Normalize 32x32 heatmap to [0.0, 1.0]
    h_arr = np.asarray(heatmap, dtype=np.float32)
    if h_arr.ndim != 2:
        h_arr = h_arr.reshape((32, 32))

    h_min = float(h_arr.min())
    h_max = float(h_arr.max())
    if h_max > h_min:
        norm_map = (h_arr - h_min) / (h_max - h_min)
    else:
        norm_map = np.zeros_like(h_arr)

    # 3. Upsample 32x32 heatmap to full page resolution with bilinear interpolation
    hmap_uint8 = (np.clip(norm_map, 0.0, 1.0) * 255).astype(np.uint8)
    hmap_pil = Image.fromarray(hmap_uint8).resize((width, height), resample=Image.Resampling.BILINEAR)

    # 4. Colorize using Jet colormap LUT
    lut = get_jet_lut()
    colored_rgb = lut[np.array(hmap_pil)]
    colored_pil = Image.fromarray(colored_rgb).convert("RGBA")

    # 5. Blend heatmap over page
    blended = Image.blend(orig_img.convert("RGBA"), colored_pil, alpha=alpha).convert("RGB")

    # 6. Draw bounding box on blended heatmap
    ymin, xmin, ymax, xmax = [float(v) for v in bbox_norm]
    x1 = int(round(xmin * width))
    y1 = int(round(ymin * height))
    x2 = int(round(xmax * width))
    y2 = int(round(ymax * height))

    draw_blended = ImageDraw.Draw(blended)
    draw_blended.rectangle([x1, y1, x2, y2], outline=(255, 30, 30), width=5)

    # Reticle corner accents on heatmap panel
    corner_len = max(8, min(int(width * 0.035), int(height * 0.035), 24))
    draw_blended.line([(x1, y1), (x1 + corner_len, y1)], fill=(255, 255, 0), width=5)
    draw_blended.line([(x1, y1), (x1, y1 + corner_len)], fill=(255, 255, 0), width=5)
    draw_blended.line([(x2, y1), (x2 - corner_len, y1)], fill=(255, 255, 0), width=5)
    draw_blended.line([(x2, y1), (x2, y1 + corner_len)], fill=(255, 255, 0), width=5)
    draw_blended.line([(x1, y2), (x1 + corner_len, y2)], fill=(255, 255, 0), width=5)
    draw_blended.line([(x1, y2), (x1, y2 - corner_len)], fill=(255, 255, 0), width=5)
    draw_blended.line([(x2, y2), (x2 - corner_len, y2)], fill=(255, 255, 0), width=5)
    draw_blended.line([(x2, y2), (x2 - corner_len, y2)], fill=(255, 255, 0), width=5)

    # 7. Left panel: Page with visual highlight overlay
    left_panel = apply_visual_highlight(orig_img, bbox_norm)

    # 8. Create combined side-by-side canvas
    header_h = 95
    label_h = 24
    margin = 20
    total_w = width * 2 + margin * 3
    total_h = height + header_h + label_h + margin * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(24, 25, 28))
    draw = ImageDraw.Draw(canvas)

    # Header banner text
    title = "AMG Visual Grounding Debug: ColPali Patch Heatmap (32x32) vs Full Page"
    draw.text((margin, 12), title, fill=(255, 255, 255))

    q_display = f"Query: \"{query_text}\"" if query_text else "Query: [Not supplied]"
    if len(q_display) > 110:
        q_display = q_display[:107] + "..."
    draw.text((margin, 36), q_display, fill=(200, 220, 255))

    stat_str = (
        f"BBox: {[round(v, 4) for v in bbox_norm]} | "
        f"1000-scale: {[int(round(v * 1000)) for v in bbox_norm]}"
    )
    if stats:
        stat_str += (
            f" | Contrast: {stats.get('contrast', 0.0):.4f} | "
            f"Conf: {stats.get('confidence', 0.0):.4f} | "
            f"Peak: {stats.get('max', 0.0):.4f}"
        )
    draw.text((margin, 58), stat_str, fill=(180, 240, 180))

    # Colorbar strip
    cbar_w, cbar_h = 220, 10
    cbar_x = total_w - margin - cbar_w
    cbar_y = 56
    cbar_strip = np.tile(lut[np.linspace(0, 255, cbar_w).astype(np.uint8)], (cbar_h, 1, 1))
    cbar_img = Image.fromarray(cbar_strip)
    canvas.paste(cbar_img, (cbar_x, cbar_y))
    draw.text((cbar_x - 30, cbar_y - 2), "Min", fill=(160, 160, 160))
    draw.text((cbar_x + cbar_w + 6, cbar_y - 2), "Peak", fill=(255, 120, 120))
    draw.text((cbar_x + 50, cbar_y - 18), "ColPali Similarity Scale", fill=(180, 180, 180))

    # Panel sub-headers
    panel_y = header_h
    draw.text((margin, panel_y), "Panel 1: Original Page with Focus Highlight (Fed to VLM)", fill=(230, 230, 230))
    draw.text((margin * 2 + width, panel_y), "Panel 2: 32x32 ColPali Similarity Heatmap Blended Over Page", fill=(255, 205, 110))

    # Paste image panels
    img_y = header_h + label_h
    canvas.paste(left_panel, (margin, img_y))
    canvas.paste(blended, (margin * 2 + width, img_y))

    # 9. Save image
    if output_path is not None:
        target = Path(output_path).expanduser().resolve()
    else:
        out_dir = Path(VISUALIZATION_DIR).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic hash of bbox coordinates to prevent naming clashes
        coords_str = "_".join(f"{v:.4f}" for v in bbox_norm)
        bbox_hash = hashlib.sha256((query_text + coords_str).encode("utf-8")).hexdigest()[:8]
        target = out_dir / f"{stem}_debug_heatmap_{bbox_hash}.png"

    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(target), format="PNG")
    logger.info("Saved combined heatmap + page visualization: %s", target.name)
    return str(target)


# ---------------------------------------------------------------------------
# Bounding Box Extraction
# ---------------------------------------------------------------------------
def extract_bounding_box(
    query_embeddings: np.ndarray | Sequence[Sequence[float]],
    page_embeddings: np.ndarray | Sequence[Sequence[float]],
    query_tokens: Sequence[str] | None = None,
    percentile_threshold: float = 85.0,
    margin_ratio: float = 0.05,
    min_box_area_ratio: float = 0.02,
    max_box_area_ratio: float = 0.85,
    confidence_contrast_thresh: float = 0.12,
    page_image: Image.Image | str | Path | None = None,
    query_text: str = "",
    visualize: bool | None = None,
) -> dict[str, Any]:
    """
    Derives normalized focal bounding box [ymin, xmin, ymax, xmax] in [0, 1]
    from ColPali query and page multi-vectors.

    Parameters
    ----------
    query_embeddings:
        Query token vectors of shape (L_q, 128).
    page_embeddings:
        Page multi-vectors of shape (N_p, 128). The first 1024 vectors
        correspond to the 32x32 spatial image patch grid.
    query_tokens:
        Optional sequence of token strings for query tokens. Used to filter out
        special prefix/pad tokens (<bos>, <eos>, <pad>) if present.
    percentile_threshold:
        Percentile cutoff for identifying high-activation patches (default: 85.0).
    margin_ratio:
        Safety padding margin added to derived bounding coordinates (default: 0.05 = 5%).
    min_box_area_ratio:
        Minimum box area fraction to guarantee readable context (default: 0.02).
    max_box_area_ratio:
        Maximum box area fraction. If exceeded, query is full-page context (default: 0.85).
    confidence_contrast_thresh:
        Minimum difference between peak and baseline activation required to treat
        the region as distinct rather than diffuse (default: 0.12).
    page_image:
        Optional path to rendered page PNG or PIL Image. If supplied and VISUALIZE
        is enabled, saves combined heatmap + page debug image.
    query_text:
        Optional user query string for debug annotation.
    visualize:
        Optional boolean override for VISUALIZE parameter in this module.

    Returns
    -------
    dict
        Contains:
        - "bbox_norm": [ymin, xmin, ymax, xmax] normalized to [0.0, 1.0]
        - "bbox_1000": [ymin, xmin, ymax, xmax] scaled to [0, 1000] for Qwen-VL
        - "confidence": float in [0.0, 1.0]
        - "is_focal": bool (True if a specific region was grounded; False if full page)
        - "heatmap_stats": {"min": float, "max": float, "mean": float, "contrast": float}
        - "visualization_path": Optional[str] (path to combined debug image if generated)
    """
    should_vis = VISUALIZE if visualize is None else visualize

    q_arr = np.asarray(query_embeddings, dtype=np.float32)
    p_arr = np.asarray(page_embeddings, dtype=np.float32)

    if q_arr.ndim != 2 or p_arr.ndim != 2:
        logger.warning(
            "Invalid embedding dimensions for bbox extraction: q=%s, p=%s",
            q_arr.shape,
            p_arr.shape,
        )
        return _fallback_full_page(
            confidence=0.0,
            page_image=page_image,
            query_text=query_text,
            visualize=should_vis,
        )

    if p_arr.shape[0] < 1024:
        logger.warning(
            "Page embedding token count %d is less than 1024 patch grid.",
            p_arr.shape[0],
        )
        return _fallback_full_page(
            confidence=0.0,
            page_image=page_image,
            query_text=query_text,
            visualize=should_vis,
        )

    # ColPali maps the first 1024 tokens to a 32x32 row-major spatial grid
    patches = p_arr[:1024]  # [1024, 128]

    # Optional: filter out non-content tokens if token strings are provided
    if query_tokens is not None and len(query_tokens) == q_arr.shape[0]:
        content_indices = [
            i
            for i, tok in enumerate(query_tokens)
            if tok not in ("<bos>", "<eos>") and not tok.startswith("<pad>")
        ]
        if content_indices:
            q_arr = q_arr[content_indices]

    if q_arr.shape[0] == 0:
        return _fallback_full_page(
            confidence=0.0,
            page_image=page_image,
            query_text=query_text,
            visualize=should_vis,
        )

    # 1. Similarity matrix: [L_q, 1024]
    sim = np.matmul(q_arr, patches.T)

    # 2. Maximum activation per patch across all query tokens
    max_act = np.max(sim, axis=0)  # [1024]
    heatmap = max_act.reshape((32, 32))

    h_min = float(heatmap.min())
    h_max = float(heatmap.max())
    h_mean = float(heatmap.mean())
    contrast = h_max - h_min

    stats = {
        "min": round(h_min, 4),
        "max": round(h_max, 4),
        "mean": round(h_mean, 4),
        "contrast": round(contrast, 4),
    }

    # If contrast is too weak, activation is diffuse across the page
    if contrast < confidence_contrast_thresh:
        return _fallback_full_page(
            confidence=round(max(0.0, float(h_max)), 4),
            stats=stats,
            page_image=page_image,
            query_text=query_text,
            heatmap=heatmap,
            visualize=should_vis,
        )

    # 3. Normalization to [0, 1]
    norm_map = (heatmap - h_min) / contrast

    # 4. Thresholding to extract top activation patches
    # Ensure threshold is strictly above baseline activation so sparse activations don't select all zeros
    pct_thresh = float(np.percentile(norm_map, percentile_threshold))
    thresh_val = max(pct_thresh, 0.35)
    binary_mask = norm_map >= thresh_val

    # 5. Morphological closing to connect adjacent lines/words into cohesive blocks
    struct = np.ones((3, 3), dtype=bool)
    closed_mask = scipy.ndimage.binary_closing(binary_mask, structure=struct)

    # 6. Connected components analysis
    labeled, num_features = scipy.ndimage.label(closed_mask)
    if num_features == 0:
        return _fallback_full_page(
            confidence=round(float(h_max), 4),
            stats=stats,
            page_image=page_image,
            query_text=query_text,
            heatmap=norm_map,
            visualize=should_vis,
        )

    # Measure activation mass per component
    component_mass = [
        float(np.sum(norm_map[labeled == comp_id]))
        for comp_id in range(1, num_features + 1)
    ]
    max_mass = max(component_mass)

    # Keep all components that have significant mass (>= 25% of top component)
    significant_comp_ids = [
        comp_id
        for comp_id, mass in enumerate(component_mass, start=1)
        if mass >= 0.25 * max_mass
    ]

    selected_mask = np.isin(labeled, significant_comp_ids)
    rows, cols = np.where(selected_mask)

    if len(rows) == 0:
        return _fallback_full_page(
            confidence=round(float(h_max), 4),
            stats=stats,
            page_image=page_image,
            query_text=query_text,
            heatmap=norm_map,
            visualize=should_vis,
        )

    r_min, r_max = int(rows.min()), int(rows.max()) + 1
    c_min, c_max = int(cols.min()), int(cols.max()) + 1

    # Convert to normalized coordinates [0.0, 1.0]
    ymin = r_min / 32.0
    ymax = r_max / 32.0
    xmin = c_min / 32.0
    xmax = c_max / 32.0

    # Apply safety margin padding
    box_h = ymax - ymin
    box_w = xmax - xmin

    pad_y = max(margin_ratio, box_h * 0.08)
    pad_x = max(margin_ratio, box_w * 0.08)

    ymin = max(0.0, ymin - pad_y)
    ymax = min(1.0, ymax + pad_y)
    xmin = max(0.0, xmin - pad_x)
    xmax = min(1.0, xmax + pad_x)

    area = (ymax - ymin) * (xmax - xmin)

    # If the box covers almost the entire page, treat as full-page focus
    if area >= max_box_area_ratio:
        return _fallback_full_page(
            confidence=round(float(norm_map[selected_mask].max()), 4),
            stats=stats,
            page_image=page_image,
            query_text=query_text,
            heatmap=norm_map,
            visualize=should_vis,
        )

    # Enforce minimum box size to preserve legibility for the vision model
    if area < min_box_area_ratio:
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        min_half = np.sqrt(min_box_area_ratio) / 2.0
        ymin = max(0.0, cy - min_half)
        ymax = min(1.0, cy + min_half)
        xmin = max(0.0, cx - min_half)
        xmax = min(1.0, cx + min_half)

    confidence = float(np.max(norm_map[selected_mask]))

    bbox_norm = [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]
    bbox_1000 = [
        int(round(ymin * 1000)),
        int(round(xmin * 1000)),
        int(round(ymax * 1000)),
        int(round(xmax * 1000)),
    ]

    # Generate combined debug visualization if requested and page image is provided
    vis_path: str | None = None
    if should_vis and page_image is not None:
        try:
            vis_path = visualize_heatmap_and_page(
                page_image=page_image,
                heatmap=norm_map,
                bbox_norm=bbox_norm,
                query_text=query_text,
                stats={**stats, "confidence": round(confidence, 4)},
            )
        except Exception as exc:
            logger.warning("Failed to generate debug heatmap visualization: %s", exc)
            vis_path = None

    response: dict[str, Any] = {
        "bbox_norm": bbox_norm,
        "bbox_1000": bbox_1000,
        "confidence": round(confidence, 4),
        "is_focal": True,
        "heatmap_stats": stats,
    }
    if vis_path is not None:
        response["visualization_path"] = vis_path

    return response


def _fallback_full_page(
    confidence: float = 0.0,
    stats: dict[str, float] | None = None,
    page_image: Image.Image | str | Path | None = None,
    query_text: str = "",
    heatmap: np.ndarray | None = None,
    visualize: bool = False,
) -> dict[str, Any]:
    """Graceful fallback representing the ungrounded full page."""
    resolved_stats = stats or {"min": 0.0, "max": 0.0, "mean": 0.0, "contrast": 0.0}
    vis_path: str | None = None

    if visualize and page_image is not None:
        try:
            hmap = heatmap if heatmap is not None else np.zeros((32, 32), dtype=np.float32)
            vis_path = visualize_heatmap_and_page(
                page_image=page_image,
                heatmap=hmap,
                bbox_norm=[0.0, 0.0, 1.0, 1.0],
                query_text=query_text,
                stats={**resolved_stats, "confidence": round(confidence, 4)},
            )
        except Exception as exc:
            logger.warning("Failed to generate debug heatmap visualization on fallback: %s", exc)
            vis_path = None

    result: dict[str, Any] = {
        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
        "bbox_1000": [0, 0, 1000, 1000],
        "confidence": round(confidence, 4),
        "is_focal": False,
        "heatmap_stats": resolved_stats,
    }
    if vis_path is not None:
        result["visualization_path"] = vis_path

    return result


def apply_visual_highlight(
    image: Image.Image,
    bbox_norm: Sequence[float],
    border_color: tuple[int, int, int, int] = DEFAULT_BORDER_COLOR,
    fill_color: tuple[int, int, int, int] = DEFAULT_FILL_COLOR,
    corner_color: tuple[int, int, int, int] = DEFAULT_CORNER_COLOR,
    border_width: int = DEFAULT_BORDER_WIDTH,
    draw_corners: bool = True,
) -> Image.Image:
    """
    Overlays a translucent focus rectangle and reticle corner accents
    on the original page image. Preserves full-page layout context while
    directing visual attention for VLM question answering.

    Parameters
    ----------
    image:
        Original PIL Image of the rendered page.
    bbox_norm:
        [ymin, xmin, ymax, xmax] in [0.0, 1.0].
    border_color:
        RGBA border outline color (default: translucent red).
    fill_color:
        RGBA inner highlight color (default: subtle amber tint).
    corner_color:
        RGBA reticle corner brackets color.
    border_width:
        Border width in pixels (default: 4).
    draw_corners:
        Whether to render corner brackets for enhanced grounding (default: True).

    Returns
    -------
    Image.Image
        Composited RGB Image.
    """
    if len(bbox_norm) != 4:
        raise ValueError(f"Expected 4 bounding coordinates; got {bbox_norm!r}")

    ymin, xmin, ymax, xmax = [float(v) for v in bbox_norm]

    # If full page or inverted, return unchanged image
    if (ymin <= 0.0 and xmin <= 0.0 and ymax >= 1.0 and xmax >= 1.0) or (ymin >= ymax or xmin >= xmax):
        return image.convert("RGB")

    width, height = image.size

    x1 = int(round(xmin * width))
    y1 = int(round(ymin * height))
    x2 = int(round(xmax * width))
    y2 = int(round(ymax * height))

    # Clamp coordinates to image boundaries
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 1. Translucent focus wash and border outline
    draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=border_color, width=border_width)

    # 2. Reticle corner brackets for sharp multimodal visual anchors
    if draw_corners:
        corner_len = max(8, min(int(width * 0.035), int(height * 0.035), 24))
        c_width = border_width + 2

        # Top-left
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=corner_color, width=c_width)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=corner_color, width=c_width)

        # Top-right
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=corner_color, width=c_width)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=corner_color, width=c_width)

        # Bottom-left
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=corner_color, width=c_width)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=corner_color, width=c_width)

        # Bottom-right
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=corner_color, width=c_width)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=corner_color, width=c_width)

    composite = Image.alpha_composite(base, overlay)
    return composite.convert("RGB")


def get_or_create_highlighted_page(
    page_image_path: str | Path,
    bbox_norm: Sequence[float],
    output_dir: str | Path = "data/highlighted_pages",
) -> str:
    """
    Renders and caches a highlighted page image with deterministic file naming.

    Parameters
    ----------
    page_image_path:
        Original rendered page image path.
    bbox_norm:
        [ymin, xmin, ymax, xmax] normalized coordinates.
    output_dir:
        Directory where highlighted page PNGs are cached.

    Returns
    -------
    str
        Path string to the highlighted page image (either newly created or cached).
    """
    source_path = Path(page_image_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Original page image not found: {source_path}")

    # If full page, return original path without generating duplicate files
    if (
        bbox_norm[0] <= 0.0
        and bbox_norm[1] <= 0.0
        and bbox_norm[2] >= 1.0
        and bbox_norm[3] >= 1.0
    ):
        return str(source_path)

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic hash of bbox coordinates to reuse existing overlays
    coords_str = "_".join(f"{v:.4f}" for v in bbox_norm)
    bbox_hash = hashlib.sha256(coords_str.encode("utf-8")).hexdigest()[:12]

    highlighted_filename = f"{source_path.stem}_bbox_{bbox_hash}.png"
    target_path = out_dir / highlighted_filename

    if target_path.is_file():
        return str(target_path)

    with Image.open(source_path) as img:
        highlighted = apply_visual_highlight(img, bbox_norm)
        highlighted.save(str(target_path), format="PNG")

    logger.info("Generated highlighted page: %s", target_path.name)
    return str(target_path)
