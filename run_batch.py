import argparse
import gc
import glob
import os
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt

from s2omics.p1_histology_preprocess import histology_preprocess
from s2omics.p2_superpixel_quality_control import superpixel_quality_control
from s2omics.p3_feature_extraction import histology_feature_extraction
from s2omics.p0_ndpi_conversion import convert_ndpi_with_fallback
from s2omics.multiple_sections.p4_get_histology_segmentation import (
    get_joint_histology_segmentation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline for NDPI -> TIFF + S2Omics steps 1-4. "
            "Designed for cluster usage across many images."
        )
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob pattern for NDPI files, e.g. '/data/**/*.ndpi'.",
    )
    parser.add_argument(
        "--input-list",
        type=str,
        default=None,
        help="Text file with one NDPI path per line.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help=(
            "Output base directory. Each image will use a subfolder named after "
            "the NDPI filename stem."
        ),
    )
    parser.add_argument(
        "--target-level",
        type=int,
        default=0,
        help="NDPI pyramid level for conversion (0 = highest resolution).",
    )
    parser.add_argument(
        "--foundation-model",
        type=str,
        default="uni",
        choices=["uni", "virchow", "gigapath"],
        help="Foundation model used in feature extraction.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./checkpoints/uni/",
        help="Checkpoint folder containing pytorch_model.bin for selected model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for step 3, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--down-samp-step",
        type=int,
        default=10,
        help="Down-sampling step used in steps 3 and 4.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers in step 3.",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "fcm", "agglo", "bisect", "birch", "louvain", "leiden"],
        help="Clustering method for step 4 histology segmentation.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=20,
        help="Initial number of clusters for step 4.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution for leiden/louvain in step 4.",
    )
    parser.add_argument(
        "--if-evaluate",
        action="store_true",
        help="Compute clustering metrics in step 4.",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Show matplotlib previews (usually disabled for clusters).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort whole batch when one sample fails.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of images processed after filtering/splitting.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="0-based task index for array job splitting.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total array tasks for splitting images across cluster jobs.",
    )
    parser.add_argument(
        "--sample-names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional sample folder names under --work-dir when using --start-step >= 4. "
            "If omitted, valid sample folders are auto-discovered."
        ),
    )
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=80,
        help="Number of PCA components for joint segmentation in step 4.",
    )
    parser.add_argument(
        "--global-pca-model-path",
        type=str,
        default=None,
        help=(
            "Optional path to a pre-fitted PCA model for step 4. "
            "If provided, PCA fitting is skipped and the model is loaded from this path."
        ),
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="First pipeline step to execute (1-4).",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Last pipeline step to execute (1-4).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Superpixel tile size in pixels (default 16).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.5,
        help="Physical size of one pixel in µm (default 0.5).",
    )
    return parser.parse_args()


def read_input_list(input_list_path):
    paths = []
    with open(input_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    return paths


def collect_inputs(args):
    if not args.input_glob and not args.input_list:
        raise ValueError("Provide at least one of --input-glob or --input-list.")

    files = []
    if args.input_glob:
        files.extend(glob.glob(args.input_glob, recursive=True))
    if args.input_list:
        files.extend(read_input_list(args.input_list))

    # Keep ordering deterministic and unique.
    files = sorted({os.path.abspath(p) for p in files})

    if not files:
        raise ValueError("No NDPI files found from provided inputs.")

    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} input files do not exist. First missing: {missing[0]}"
        )

    return files


def has_required_step4_files(sample_dir, foundation_model, down_samp_step):
    pickle_dir = Path(sample_dir) / "S2Omics_output" / "pickle_files"
    required = [
        pickle_dir / "shapes.pickle",
        pickle_dir / "qc_preserve_indicator.pickle",
        pickle_dir / f"{foundation_model}_embeddings_downsamp_{down_samp_step}_part_0.pickle",
    ]
    return all(p.exists() for p in required)


def collect_existing_sample_dirs(args, require_step4_files=True):
    work_dir = Path(args.work_dir).resolve()
    if not work_dir.exists():
        raise FileNotFoundError(f"work-dir does not exist: {work_dir}")

    if args.sample_names:
        sample_dirs = [work_dir / name for name in args.sample_names]
        missing = [p for p in sample_dirs if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Sample folders not found: " + ", ".join(str(p.name) for p in missing)
            )
    else:
        sample_dirs = []
        for child in sorted(work_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "S2Omics_output").exists():
                continue
            sample_dirs.append(child)

    if not require_step4_files:
        resolved = [str(p.resolve()) for p in sample_dirs]
        if not resolved:
            raise ValueError("No sample folders found under --work-dir.")
        return resolved

    valid = [
        str(p.resolve())
        for p in sample_dirs
        if has_required_step4_files(p, args.foundation_model, args.down_samp_step)
    ]

    if not valid:
        discovered = len(sample_dirs)
        sample_msg = ""
        if discovered:
            first = sample_dirs[0]
            first_pickle = first / "S2Omics_output" / "pickle_files"
            expected = [
                first_pickle / "shapes.pickle",
                first_pickle / "qc_preserve_indicator.pickle",
                first_pickle / f"{args.foundation_model}_embeddings_downsamp_{args.down_samp_step}_part_0.pickle",
            ]
            missing_names = [p.name for p in expected if not p.exists()]
            sample_msg = (
                f" Example sample: {first.name}. "
                f"Missing there: {', '.join(missing_names) if missing_names else 'unknown mismatch'}."
            )
        raise ValueError(
            "No valid sample folders found for step4-only mode. "
            "Expected S2Omics_output/pickle_files with shapes.pickle, "
            "qc_preserve_indicator.pickle and embeddings part_0. "
            f"Discovered {discovered} sample folder(s) under --work-dir.{sample_msg}"
        )

    return valid


def split_for_task(files, task_id=None, num_tasks=None):
    if task_id is None and num_tasks is None:
        return files
    if task_id is None or num_tasks is None:
        raise ValueError("Use --task-id and --num-tasks together.")
    if num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0.")
    if task_id < 0 or task_id >= num_tasks:
        raise ValueError("--task-id must satisfy 0 <= task_id < num_tasks.")
    return [p for idx, p in enumerate(files) if idx % num_tasks == task_id]


def should_run_step(args, step):
    return args.start_step <= step <= args.end_step


def reset_step123_outputs(sample_out_dir, step, foundation_model, down_samp_step):
    sample_path = Path(sample_out_dir)
    s2_out = sample_path / "S2Omics_output"
    pickle_dir = s2_out / "pickle_files"

    if step == 1:
        for rel in ["he.tiff", "he-scaled.tiff", "he.jpg", "he-scaled.jpg"]:
            target = sample_path / rel
            if target.exists():
                target.unlink()
    elif step == 2:
        for rel in ["shapes.pickle", "qc_preserve_indicator.pickle"]:
            target = pickle_dir / rel
            if target.exists():
                target.unlink()
    elif step == 3:
        target = pickle_dir / "num_patches.pickle"
        if target.exists():
            target.unlink()
        for emb in pickle_dir.glob(
            f"{foundation_model}_embeddings_downsamp_{down_samp_step}_part_*.pickle"
        ):
            emb.unlink()


def reset_step4_outputs(sample_out_dir):
    sample_path = Path(sample_out_dir)
    s2_out = sample_path / "S2Omics_output"
    image_dir = s2_out / "image_files"
    pickle_dir = s2_out / "pickle_files"

    for rel in ["cluster_image.pickle", "clustering_metrics.pickle"]:
        target = pickle_dir / rel
        if target.exists():
            target.unlink()
    for ext in ["jpg", "jpeg", "tif", "tiff"]:
        for image in image_dir.glob(f"cluster_image_num_clusters_*.{ext}"):
            image.unlink()


def process_one_steps123(ndpi_path, args):
    """Run steps 1-3 for a single NDPI file (per-sample)."""
    sample_name = Path(ndpi_path).stem
    sample_out_dir = os.path.join(args.work_dir, sample_name)
    os.makedirs(sample_out_dir, exist_ok=True)

    print(f"[START steps 1-3] {sample_name}")

    prefix = sample_out_dir.rstrip("/") + "/"
    save_folder = os.path.join(sample_out_dir, "S2Omics_output")

    if should_run_step(args, 1):
        reset_step123_outputs(
            sample_out_dir, 1, args.foundation_model, args.down_samp_step,
        )
        convert_ndpi_with_fallback(
            ndpi_path=ndpi_path,
            sample_out_dir=sample_out_dir,
            target_level=args.target_level,
        )
        histology_preprocess(prefix, show_image=args.show_image)

    if should_run_step(args, 2):
        reset_step123_outputs(
            sample_out_dir, 2, args.foundation_model, args.down_samp_step,
        )
        superpixel_quality_control(prefix, save_folder, show_image=args.show_image)

    if should_run_step(args, 3):
        reset_step123_outputs(
            sample_out_dir, 3, args.foundation_model, args.down_samp_step,
        )
        histology_feature_extraction(
            prefix, save_folder,
            foundation_model=args.foundation_model,
            ckpt_path=args.ckpt_path,
            device=args.device,
            batch_size=args.batch_size,
            down_samp_step=args.down_samp_step,
            num_workers=args.num_workers,
        )

    print(f"[DONE steps 1-3]  {sample_name}")


def run_joint_step4(sample_dirs, args):
    """Run joint histology segmentation (step 4) across all samples at once."""
    prefix_list = []
    save_folder_list = []

    for sample_dir in sample_dirs:
        prefix_list.append(str(Path(sample_dir).resolve()) + "/")
        save_folder = os.path.join(sample_dir, "S2Omics_output")
        save_folder_list.append(save_folder)
        reset_step4_outputs(sample_dir)

    pca_model_path = args.global_pca_model_path or ""

    get_joint_histology_segmentation(
        prefix_list=prefix_list,
        save_folder_list=save_folder_list,
        foundation_model=args.foundation_model,
        down_samp_step=args.down_samp_step,
        clustering_method=args.clustering_method,
        n_clusters=args.n_clusters,
        resolution=args.resolution,
        if_evaluate=args.if_evaluate,
        pca_model_path=pca_model_path,
    )


def main():
    args = parse_args()

    if args.start_step > args.end_step:
        raise ValueError("--start-step must be <= --end-step.")

    # Determine if running step 4 only (no input NDPI files needed)
    step4_only = args.start_step >= 4

    if step4_only:
        files = collect_existing_sample_dirs(args, require_step4_files=True)
    else:
        files = collect_inputs(args)

    files = split_for_task(files, args.task_id, args.num_tasks)

    if args.max_images is not None:
        files = files[: args.max_images]

    if not files:
        print("No files assigned to this run after filtering/splitting.")
        return

    print(f"Total assigned files: {len(files)}")
    print(f"Output base directory: {os.path.abspath(args.work_dir)}")
    print(f"Foundation model: {args.foundation_model}")
    if not step4_only:
        print(f"Device: {args.device}")
    print(f"Run steps: {args.start_step} -> {args.end_step}")
    if step4_only:
        print("Mode: step 4 only (existing samples, joint clustering)")

    failed = []

    if step4_only:
        # --- Step 4 only mode: samples already have embeddings ---
        sample_dirs = files

        if should_run_step(args, 4):
            try:
                print(f"\n[JOINT STEP 4] Running joint segmentation on {len(sample_dirs)} samples...")
                run_joint_step4(sample_dirs, args)
            except Exception as exc:
                print(f"[FAIL] Joint step 4 failed: {exc}")
                traceback.print_exc()
                if args.stop_on_error:
                    sys.exit(1)
                failed.extend(sample_dirs)
            finally:
                plt.close("all")
                gc.collect()
    else:
        # --- Full pipeline: steps 1-3 per sample, then joint step 4 ---

        # Phase 1: Steps 1-3 per sample
        completed_sample_dirs = []
        for ndpi_path in files:
            try:
                if args.end_step >= 1 and args.start_step <= 3:
                    process_one_steps123(ndpi_path, args)
                sample_name = Path(ndpi_path).stem
                sample_out_dir = os.path.join(args.work_dir, sample_name)
                completed_sample_dirs.append(sample_out_dir)
            except Exception as exc:
                print(f"[FAIL]  {ndpi_path}")
                print(f"Reason: {exc}")
                traceback.print_exc()
                failed.append(ndpi_path)
                if args.stop_on_error:
                    break
            finally:
                plt.close("all")
                gc.collect()

        # Phase 2: Joint step 4 on all successfully completed samples
        if should_run_step(args, 4) and completed_sample_dirs:
            try:
                print(f"\n[JOINT STEP 4] Running joint segmentation on {len(completed_sample_dirs)} samples...")
                run_joint_step4(completed_sample_dirs, args)
            except Exception as exc:
                print(f"[FAIL] Joint step 4 failed: {exc}")
                traceback.print_exc()
                if args.stop_on_error:
                    sys.exit(1)
                failed.extend(completed_sample_dirs)
                completed_sample_dirs = []
            finally:
                plt.close("all")
                gc.collect()

    print("\nBatch finished.")
    print(f"Success: {len(files) - len(failed)}")
    print(f"Failed:  {len(failed)}")

    if failed:
        print("Failed files:")
        for p in failed:
            print(p)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()