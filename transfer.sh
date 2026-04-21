SRC="/scratch/gsolun/S2Omics/outputs_1"
DST="/scratch/gsolun/S2Omics/outputs_step4_archive/global_1_$(date +%Y%m%d_%H%M%S)"
PREFIX=""


mkdir -p "$DST"


find "$SRC" -mindepth 1 -maxdepth 1 -type d -name "${PREFIX}*" | while read -r sample; do
  name="$(basename "$sample")"
  pdir="$sample/S2Omics_output/pickle_files"
  idir="$sample/S2Omics_output/image_files"


  [[ -d "$pdir" || -d "$idir" ]] || continue


  mkdir -p "$DST/$name/S2Omics_output/pickle_files"
  mkdir -p "$DST/$name/S2Omics_output/image_files"


  # Step 4 pickles
  for f in cluster_image.pickle clustering_metrics.pickle; do
    [[ -f "$pdir/$f" ]] && mv "$pdir/$f" "$DST/$name/S2Omics_output/pickle_files/"
  done


  # Step 4 images (all supported extensions)
  for ext in jpg jpeg tif tiff; do
    for f in "$idir"/cluster_image_num_clusters_*."$ext"; do
      [[ -e "$f" ]] && mv "$f" "$DST/$name/S2Omics_output/image_files/"
    done
  done
done


echo "Moved step 4 outputs to: $DST"
