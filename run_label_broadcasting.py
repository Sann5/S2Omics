import argparse

from s2omics.p1_histology_preprocess import histology_preprocess
from s2omics.p2_superpixel_quality_control import superpixel_quality_control
from s2omics.p3_feature_extraction import histology_feature_extraction
from s2omics.multiple_sections.p6_cell_label_broadcasting import label_broadcasting


def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--WSI_save_folder', type=str, required=True,
                        help='S2Omics output root for the whole-slide image (per-step subfolders inside).')
    parser.add_argument('--SO_save_folder', type=str, required=True,
                        help='S2Omics output root for the spatial-omics image.')
    parser.add_argument('--SO_annotation_csv', type=str, required=True,
                        help='Path to annotation_file.csv with super_pixel_x/y + annotation columns.')
    parser.add_argument('--WSI_cache_path', type=str, default='',
                        help='Optional override for WSI embedding cache directory (must end in /).')
    parser.add_argument('--SO_cache_path', type=str, default='',
                        help='Optional override for SO embedding cache directory (must end in /).')
    parser.add_argument('--need_preprocess', action='store_true')
    parser.add_argument('--need_feature_extraction', action='store_true')
    parser.add_argument('--foundation_model', type=str, default='uni')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/uni/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--density_thresh', type=float, default=100)
    parser.add_argument('--clean_background_flag', action='store_true')
    parser.add_argument('--min_size', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=16)
    return parser.parse_args()


def main():
    args = get_args()

    qc_kwargs = dict(
        density_thresh=args.density_thresh,
        clean_background_flag=args.clean_background_flag,
        min_size=args.min_size,
        patch_size=args.patch_size,
        show_image=False,
    )

    if args.need_preprocess:
        histology_preprocess(args.WSI_save_folder, show_image=False)
        histology_preprocess(args.SO_save_folder, show_image=False)
        superpixel_quality_control(args.WSI_save_folder, **qc_kwargs)
        superpixel_quality_control(args.SO_save_folder, **qc_kwargs)

    if args.need_feature_extraction:
        histology_feature_extraction(
            args.WSI_save_folder,
            foundation_model=args.foundation_model,
            ckpt_path=args.ckpt_path,
            device=args.device,
            down_samp_step=1,
        )
        if args.WSI_save_folder != args.SO_save_folder:
            histology_feature_extraction(
                args.SO_save_folder,
                foundation_model=args.foundation_model,
                ckpt_path=args.ckpt_path,
                device=args.device,
                down_samp_step=1,
            )

    label_broadcasting(
        WSI_save_folder=args.WSI_save_folder,
        SO_save_folder=args.SO_save_folder,
        SO_annotation_csv=args.SO_annotation_csv,
        WSI_cache_path=args.WSI_cache_path,
        SO_cache_path=args.SO_cache_path,
        foundation_model=args.foundation_model,
        device=args.device,
    )


if __name__ == '__main__':
    main()
