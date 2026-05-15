import argparse

from s2omics.p1_histology_preprocess import histology_preprocess
from s2omics.p2_superpixel_quality_control import superpixel_quality_control
from s2omics.p3_feature_extraction import histology_feature_extraction
from s2omics.multiple_sections.p4_get_histology_segmentation import get_joint_histology_segmentation
from s2omics.multiple_sections.p5_roi_selection_rectangle import roi_selection_for_multiple_sections
## if need to select circle-shaped ROI, please
# from s2omics.multiple_sections.p5_roi_selection_circle import roi_selection_for_multiple_sections


def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--save_folder_list', type=str, nargs='+', required=True,
                        help='Per-sample S2Omics output roots (one per section).')
    parser.add_argument('--foundation_model', type=str, default='uni')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/uni/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--down_samp_step', type=int, default=10)
    parser.add_argument('--density_thresh', type=float, default=100)
    parser.add_argument('--clean_background_flag', action='store_true')
    parser.add_argument('--masking_method', type=str, default='s2omics',
                        choices=['s2omics', 'victor'])
    parser.add_argument('--victor_mean_threshold', type=float, default=0.85)
    parser.add_argument('--victor_max_iterations', type=int, default=5)
    parser.add_argument('--victor_sigma', type=float, default=20)
    parser.add_argument('--victor_positive_contrast', action='store_true')
    parser.add_argument('--victor_superpixel_threshold', type=float, default=0.5)
    parser.add_argument('--min_size', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--clustering_method', type=str, default='kmeans')
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--roi_size', type=float, nargs='+', default=[6.5, 6.5])
    parser.add_argument('--num_roi', type=int, default=0)  # 0 means automatic
    parser.add_argument('--fusion_weights', type=float, nargs='+', default=[0.33, 0.33, 0.33])
    parser.add_argument('--emphasize_clusters', type=int, nargs='+', default=[])
    parser.add_argument('--discard_clusters', type=int, nargs='+', default=[])
    parser.add_argument('--prior_preference', type=int, default=2)
    return parser.parse_args()


def main():
    args = get_args()

    for save_folder in args.save_folder_list:
        histology_preprocess(save_folder, show_image=False)

    for save_folder in args.save_folder_list:
        superpixel_quality_control(
            save_folder,
            density_thresh=args.density_thresh,
            clean_background_flag=args.clean_background_flag,
            min_size=args.min_size,
            patch_size=args.patch_size,
            masking_method=args.masking_method,
            victor_mean_threshold=args.victor_mean_threshold,
            victor_max_iterations=args.victor_max_iterations,
            victor_sigma=args.victor_sigma,
            victor_positive_contrast=args.victor_positive_contrast,
            victor_superpixel_threshold=args.victor_superpixel_threshold,
            show_image=False,
        )

    for save_folder in args.save_folder_list:
        histology_feature_extraction(
            save_folder,
            foundation_model=args.foundation_model,
            ckpt_path=args.ckpt_path,
            device=args.device,
            down_samp_step=args.down_samp_step,
        )

    get_joint_histology_segmentation(
        args.save_folder_list,
        foundation_model=args.foundation_model,
        down_samp_step=args.down_samp_step,
        clustering_method=args.clustering_method,
        n_clusters=args.n_clusters,
    )

    roi_selection_for_multiple_sections(
        args.save_folder_list,
        down_samp_step=args.down_samp_step,
        roi_size=args.roi_size,
        num_roi=args.num_roi,
        fusion_weights=args.fusion_weights,
        emphasize_clusters=args.emphasize_clusters,
        discard_clusters=args.discard_clusters,
        prior_preference=args.prior_preference,
    )


if __name__ == '__main__':
    main()
