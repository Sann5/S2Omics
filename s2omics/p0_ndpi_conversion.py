#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
import tifffile

from . import step_paths


# Disable DecompressionBombError. WSI images are massive and will trigger Pillow limits.
Image.MAX_IMAGE_PIXELS = None


def _percent_to_pixels(length, percent):
    """Convert percentage to pixel count."""
    return int(round(length * percent / 100.0))


def convert_ndpi_to_tiff(
    ndpi_file_path,
    save_folder,
    output_tiff="he-raw.tiff",
    output_txt="pixel-size-raw.txt",
    target_level=0,
    crop_top_percent=0,
    crop_bottom_percent=0,
    crop_left_percent=0,
    crop_right_percent=0,
):
    """
    Converts an NDPI file to a TIFF image and extracts effective pixel size in micrometers.

    Outputs are written to ``<save_folder>/p0_ndpi_conversion/``.

    Optionally crops the image by percentages from each side.

    Args:
        ndpi_file_path: Path to input NDPI file
        save_folder: Per-sample S2Omics output root (e.g. ``<sample>/S2Omics_output``)
        output_tiff: Output TIFF filename
        output_txt: Output pixel size text filename
        target_level: Pyramid level to extract from
        crop_top_percent: Percentage to crop from top (0-100)
        crop_bottom_percent: Percentage to crop from bottom (0-100)
        crop_left_percent: Percentage to crop from left (0-100)
        crop_right_percent: Percentage to crop from right (0-100)
    """
    if not os.path.exists(ndpi_file_path):
        raise FileNotFoundError(f"Could not find input NDPI: {ndpi_file_path}")

    out_dir = step_paths.step_dir(save_folder, step_paths.P0_NDPI_CONVERSION)

    output_tiff = os.path.join(out_dir, output_tiff)
    output_txt = os.path.join(out_dir, output_txt)

    # Validate percentage ranges
    for name, val in (
        ("top", crop_top_percent),
        ("bottom", crop_bottom_percent),
        ("left", crop_left_percent),
        ("right", crop_right_percent),
    ):
        if not (0.0 <= float(val) <= 100.0):
            raise ValueError(f"Crop percent '{name}' must be between 0 and 100; got {val}")

    has_cropping = any(
        float(x) != 0.0 for x in (crop_top_percent, crop_bottom_percent, crop_left_percent, crop_right_percent)
    )

    # If outputs already exist and no cropping is requested, skip.
    if os.path.exists(output_tiff) and os.path.exists(output_txt) and not has_cropping:
        print(f"Skipping NDPI conversion; outputs already exist: '{output_tiff}', '{output_txt}'.")
        return output_tiff

    print(f"Opening {ndpi_file_path}...")
    slide = openslide.OpenSlide(ndpi_file_path)

    try:
        try:
            base_mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            downsample_factor = slide.level_downsamples[target_level]
            effective_mpp = base_mpp_x * downsample_factor
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(f"{effective_mpp:.6f}\n")
            print(f"Success: Wrote pixel size {effective_mpp:.6f} um to '{output_txt}'.")
        except KeyError:
            print("Warning: MPP physical size data not found in the NDPI metadata.")
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write("Unknown\n")

        width, height = slide.level_dimensions[target_level]

        if has_cropping:
            top_px = _percent_to_pixels(height, crop_top_percent)
            bottom_px = _percent_to_pixels(height, crop_bottom_percent)
            left_px = _percent_to_pixels(width, crop_left_percent)
            right_px = _percent_to_pixels(width, crop_right_percent)

            x0 = left_px
            y0 = top_px
            x1 = width - right_px
            y1 = height - bottom_px

            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    "Crop removes the entire image. "
                    f"Computed bounds: x=[{x0}, {x1}), y=[{y0}, {y1}) for size {width}x{height}."
                )

            crop_w = x1 - x0
            crop_h = y1 - y0

            print(
                f"Extracting cropped region: level {target_level} with dimensions "
                f"{crop_w}x{crop_h} pixels (from {width}x{height})..."
            )
            print(
                f"Crop bounds: top={top_px}, bottom={bottom_px}, "
                f"left={left_px}, right={right_px} pixels"
            )

            # read_region expects level-0 coordinates for the location argument.
            ds = slide.level_downsamples[target_level]
            loc_x0 = int(round(x0 * ds))
            loc_y0 = int(round(y0 * ds))
            img = slide.read_region((loc_x0, loc_y0), target_level, (crop_w, crop_h)).convert("RGB")
        else:
            print(
                f"Extracting level {target_level} with dimensions {width}x{height} pixels..."
            )
            # full-level extraction: location (0,0) at level 0 maps to top-left of the image
            img = slide.read_region((0, 0), target_level, (width, height)).convert("RGB")

        print(f"Saving image to '{output_tiff}' (this may take a moment for large files)...")
        tifffile.imwrite(output_tiff, np.asarray(img), photometric="rgb")
        print("Done!")
    except Exception as exc:
        raise RuntimeError(f"NDPI image extraction failed for '{ndpi_file_path}': {exc}") from exc
    finally:
        slide.close()


def convert_ndpi_to_image(*args, **kwargs):
    """Backward-compatible alias used in notebooks and docs."""
    return convert_ndpi_to_tiff(*args, **kwargs)


def _has_he_raw(p0_dir):
    base = Path(p0_dir) / "he-raw"
    for suffix in (".png", ".ome.tif", ".tiff", ".tif", ".svs"):
        if base.with_suffix(suffix).exists():
            return True
    return False


def convert_ndpi_with_fallback(
    ndpi_path,
    save_folder,
    target_level,
    crop_top_percent=0,
    crop_bottom_percent=0,
    crop_left_percent=0,
    crop_right_percent=0,
):
    """Run convert_ndpi_to_tiff with retries at deeper pyramid levels.

    Crop parameters are forwarded so fallback attempts produce the intended cropped output.
    """
    p0_dir = step_paths.step_dir(save_folder, step_paths.P0_NDPI_CONVERSION)
    for level in range(target_level, target_level + 5):
        try:
            convert_ndpi_to_tiff(
                ndpi_file_path=ndpi_path,
                save_folder=save_folder,
                output_tiff="he-raw.tiff",
                output_txt="pixel-size-raw.txt",
                target_level=level,
                crop_top_percent=crop_top_percent,
                crop_bottom_percent=crop_bottom_percent,
                crop_left_percent=crop_left_percent,
                crop_right_percent=crop_right_percent,
            )
            if _has_he_raw(p0_dir) and (Path(p0_dir) / "pixel-size-raw.txt").exists():
                if level != target_level:
                    print(
                        f"[WARN] NDPI extraction retried at target level {level} "
                        f"(requested {target_level})."
                    )
                return
            raise RuntimeError("Conversion finished but expected output files were not created.")
        except Exception as exc:
            if level == target_level + 4:
                raise RuntimeError(
                    f"NDPI conversion failed after retries from level {target_level} to {level}: {exc}"
                ) from exc
            print(
                f"[WARN] NDPI conversion failed at level {level} for '{ndpi_path}'. "
                f"Retrying deeper level. Reason: {exc}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Convert NDPI to TIFF and extract pixel size metadata. Optionally crop the image."
    )
    parser.add_argument("input_ndpi", help="Path to input .ndpi file")
    parser.add_argument(
        "-s", "--save-folder",
        help=(
            "Per-sample S2Omics output root. Outputs are written to "
            "<save-folder>/p0_ndpi_conversion/. Defaults to '<NDPI dir>/S2Omics_output'."
        ),
        default=None,
    )
    parser.add_argument(
        "--output-tiff",
        help="Output TIFF filename",
        default="he-raw.tiff",
    )
    parser.add_argument(
        "--output-txt",
        help="Output pixel size filename",
        default="pixel-size-raw.txt",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Pyramid level to extract from",
    )
    parser.add_argument(
        "--crop-top",
        type=float,
        default=0.0,
        help="Percentage to crop from top (0-100)",
    )
    parser.add_argument(
        "--crop-bottom",
        type=float,
        default=0.0,
        help="Percentage to crop from bottom (0-100)",
    )
    parser.add_argument(
        "--crop-left",
        type=float,
        default=0.0,
        help="Percentage to crop from left (0-100)",
    )
    parser.add_argument(
        "--crop-right",
        type=float,
        default=0.0,
        help="Percentage to crop from right (0-100)",
    )

    args = parser.parse_args()

    save_folder = args.save_folder
    if save_folder is None:
        save_folder = os.path.join(os.path.dirname(os.path.abspath(args.input_ndpi)), "S2Omics_output")

    convert_ndpi_to_tiff(
        ndpi_file_path=args.input_ndpi,
        save_folder=save_folder,
        output_tiff=args.output_tiff,
        output_txt=args.output_txt,
        target_level=args.level,
        crop_top_percent=args.crop_top,
        crop_bottom_percent=args.crop_bottom,
        crop_left_percent=args.crop_left,
        crop_right_percent=args.crop_right,
    )


if __name__ == "__main__":
    main()
