import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from pathlib import Path
from harmonypy import run_harmony
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, BisectingKMeans
from skfuzzy.cluster import cmeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from .. import step_paths
from ..s1_utils import (
    load_pickle, save_figure_safely, save_pickle, setup_seed, stage_is_complete)


def _load_embedding_parts(cache_path, foundation_model, down_samp_step):
    he_embed_total = []
    i = 0
    while 1 > 0:
        embedding_path = cache_path + foundation_model + f'_embeddings_downsamp_{down_samp_step}_part_{i}.pickle'
        if os.path.exists(embedding_path):
            he_embed_part = load_pickle(embedding_path)
            he_embed_total.append(he_embed_part)
            i += 1
        else:
            break
    if len(he_embed_total) == 0:
        raise FileNotFoundError(
            f'No embedding parts found in {cache_path} for model={foundation_model}, down_samp_step={down_samp_step}'
        )
    return np.concatenate(he_embed_total)


def get_joint_histology_segmentation(save_folder_list,
                                    foundation_model='uni', cache_path_list='',
                                    down_samp_step=10, clustering_method='kmeans',
                                    n_clusters=20, resolution=1.0,
                                    if_evaluate=False, pca_encoder=None,
                                    pca_model_path=''):
    '''
    Joint histology segmentation across multiple sections using global PCA + Harmony + joint clustering.
    Parameters:
        save_folder_list: list of per-sample S2Omics output roots. Each entry
            must already have ``p2_qc/`` and ``p3_features/`` populated. The
            cluster image, completion marker, and segmentation JPG are written
            to ``<save_folder>/p4_segmentation/``.
        foundation_model: foundation model used for feature extraction (uni, virchow, gigapath)
        cache_path_list: optional path to a text file listing cache paths, or empty string to use default
        down_samp_step: the down-sampling step for feature extraction, default = 10
        clustering_method: 'kmeans', 'fcm', 'agglo', 'bisect', 'birch', 'louvain', or 'leiden'
        n_clusters: initial number of clusters when using kmeans/fcm/etc.
        resolution: resolution for leiden/louvain algorithm, default=1.0
        if_evaluate: compute clustering metrics, default=False
        pca_encoder: optional pre-fitted PCA encoder; if provided, PCA fitting is skipped
        pca_model_path: optional path to serialized PCA encoder (used when pca_encoder is None)
    '''

    # define color palette
    color_list = np.loadtxt(os.path.join(os.path.dirname(__file__), '../color_list.txt'), dtype='int').tolist()
    with open(os.path.join(os.path.dirname(__file__), '../color_list_16bit.txt'), "r", encoding="utf-8") as file:
        lines = file.readlines()
    color_list_16bit = []
    for line in lines:
        color_list_16bit.append(line.strip())
    cluster_color_mapping = np.arange(len(color_list))

    save_folder_list_raw = save_folder_list
    n_images = len(save_folder_list_raw)

    expected_metadata = {
        'save_folder_list': [os.path.abspath(folder) for folder in save_folder_list_raw],
        'foundation_model': foundation_model,
        'down_samp_step': down_samp_step,
        'clustering_method': clustering_method,
        'n_clusters': n_clusters,
        'resolution': resolution,
        'if_evaluate': if_evaluate,
        'pca_model_path': os.path.abspath(pca_model_path) if pca_model_path else '',
    }
    completion_marker_name = 'joint_histology_segmentation_complete.pickle'
    if all(
        stage_is_complete(
            str(Path(step_paths.step_dir(save_folder, step_paths.P4_SEGMENTATION, create=False).rstrip('/')) / completion_marker_name),
            expected_metadata,
            required_outputs=(
                str(Path(step_paths.step_dir(save_folder, step_paths.P4_SEGMENTATION, create=False).rstrip('/')) / 'cluster_image.pickle'),
            ),
        )
        for save_folder in save_folder_list_raw
    ):
        print(
            'Skipping joint histology segmentation; outputs already exist for the '
            'current sample set and configuration.'
        )
        return

    if len(cache_path_list) > 0:
        with open(cache_path_list, "r", encoding="utf-8") as file:
            lines = file.readlines()
        cache_path_list = [line.split()[0] for line in lines]
        assert len(save_folder_list_raw) == len(cache_path_list)

    dpi = 1200
    p2_dir_list = []
    p4_dir_list = []
    image_shape_list = []
    plt_figsize_list = []
    qc_mask_list = []
    down_samp_mask_list = []
    he_embed_qc_list = []
    pixels_counter = [0]

    for i in range(n_images):
        save_folder = save_folder_list_raw[i]
        p2_dir = step_paths.step_dir(save_folder, step_paths.P2_QC, create=False)
        p3_dir = step_paths.step_dir(save_folder, step_paths.P3_FEATURES, create=False)
        p4_dir = step_paths.step_dir(save_folder, step_paths.P4_SEGMENTATION)
        p2_dir_list.append(p2_dir)
        p4_dir_list.append(p4_dir)

        # load in previously obtained params
        shapes = load_pickle(p2_dir + 'shapes.pickle')
        image_shape = shapes['tiles']
        image_shape_list.append(image_shape)
        length = np.max(image_shape) // 100
        plt_figsize = (image_shape[1] // 100, image_shape[0] // 100)
        if dpi * length > np.power(2, 16):
            reduce_ratio = np.power(2, 16) / (dpi * length)
            plt_figsize = ((image_shape[1] * reduce_ratio) // 100, (image_shape[0] * reduce_ratio) // 100)
        plt_figsize_list.append(plt_figsize)
        qc_preserve_indicator = load_pickle(p2_dir + 'qc_preserve_indicator.pickle')
        qc_mask = np.reshape(qc_preserve_indicator, image_shape)
        qc_mask_list.append(qc_mask)

        # load in histology features
        print(f'Loading histology feature embeddings for image {i}...')
        if len(cache_path_list) > 0:
            cache_path = cache_path_list[i]
        else:
            cache_path = p3_dir
        he_embed_total = _load_embedding_parts(cache_path, foundation_model, down_samp_step)
        print('Sucessfully loaded and normalized all histology feature embeddings!')

        # create a mask for down-sampled superpixels in all superpixels
        down_samp_mask = np.full(image_shape, False)
        down_samp_shape = [(image_shape[0] - 1) // down_samp_step + 1, (image_shape[1] - 1) // down_samp_step + 1]
        for r in range(down_samp_shape[0]):
            for c in range(down_samp_shape[1]):
                down_samp_mask[r * down_samp_step, c * down_samp_step] = True
        down_samp_mask_list.append(down_samp_mask)

        # PCA+kmeans to cluster the superpixels into morphology clusters
        he_embed_qc = he_embed_total[qc_mask[down_samp_mask]]
        del he_embed_total
        he_embed_qc_list.append(he_embed_qc)
        pixels_counter.append(pixels_counter[-1] + len(he_embed_qc))
        del he_embed_qc

    setup_seed(42)
    he_embed_qc_concat = np.concatenate(he_embed_qc_list)
    len_data = []
    for i in range(n_images):
        len_data.append(len(he_embed_qc_list[i]))
    batch_label = ['slide 1'] * len_data[0]
    for i in range(1, n_images):
        batch_label += [f'slide {i+1}'] * len_data[i]
    del he_embed_qc_list

    if pca_encoder is None and len(pca_model_path) > 0 and os.path.exists(pca_model_path):
        pca_encoder = load_pickle(pca_model_path)
        print(f'Loaded external PCA model from: {pca_model_path}')
    if pca_encoder is None:
        pca_encoder = PCA(n_components=80)
        pca_encoder.fit(he_embed_qc_concat)
        print('Fitted new PCA model on concatenated embeddings.')
        if len(pca_model_path) > 0:
            save_pickle(pca_encoder, pca_model_path)
            print(f'Saved fitted PCA model to: {pca_model_path}')
    else:
        print('Using provided pre-fitted PCA model.')
    he_embed_qc_pca = pca_encoder.transform(he_embed_qc_concat)
    adata = sc.AnnData(he_embed_qc_pca)
    if not if_evaluate:
        del he_embed_qc_pca
    adata.obs['batch_label'] = batch_label
    he_embed_harmony = run_harmony(adata.X, adata.obs, 'batch_label', max_iter_harmony=10)
    # Access the corrected embeddings
    he_embed_harmony = he_embed_harmony.Z_corr.T
    del adata

    print(f'Start segmenting the histology image, clustering method: {clustering_method}')
    if clustering_method == 'kmeans':
        uni_cluster = KMeans(n_clusters=n_clusters).fit_predict(he_embed_harmony).astype('int')
    if clustering_method == 'fcm':
        train = he_embed_harmony.T
        center, u, _, _, _, _, _ = cmeans(train, m=2, c=n_clusters, error=0.005, maxiter=1000)
        for _ in u:
            uni_cluster = np.argmax(u, axis=0)
    if clustering_method == 'agglo':
        uni_cluster = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(he_embed_harmony).astype('int')
    if clustering_method == 'bisect':
        uni_cluster = BisectingKMeans(n_clusters=n_clusters).fit_predict(he_embed_harmony).astype('int')
    if clustering_method == 'birch':
        uni_cluster = Birch(n_clusters=n_clusters).fit_predict(he_embed_harmony).astype('int')
    if clustering_method == 'louvain':
        adata = sc.AnnData(he_embed_harmony)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
        sc.tl.louvain(adata, resolution=resolution)
        uni_cluster = adata.obs['louvain']
    if clustering_method == 'leiden':
        adata = sc.AnnData(he_embed_harmony)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
        sc.tl.leiden(adata, resolution=resolution)
        uni_cluster = adata.obs['leiden']

    # Renumber cluster IDs to consecutive 0..N-1 (globally, once)
    histology_clusters_new = np.unique(uni_cluster[uni_cluster > -1])
    actual_n_clusters = len(histology_clusters_new)
    uni_cluster_copy = uni_cluster.copy()
    for idx in range(actual_n_clusters):
        uni_cluster[uni_cluster_copy == histology_clusters_new[idx]] = idx
    n_clusters = actual_n_clusters

    for i in range(n_images):
        image_shape = image_shape_list[i]
        qc_mask = qc_mask_list[i]
        down_samp_shape = [(image_shape[0] - 1) // down_samp_step + 1, (image_shape[1] - 1) // down_samp_step + 1]
        down_samp_mask = down_samp_mask_list[i]
        curr_uni_cluster = uni_cluster[pixels_counter[i]:pixels_counter[i + 1]]
        p4_dir = p4_dir_list[i]
        plt_figsize = plt_figsize_list[i]

        cluster_image = -1 * np.ones(image_shape)
        cluster_image[qc_mask & down_samp_mask] = curr_uni_cluster
        cluster_image = cluster_image[down_samp_mask]
        cluster_image = np.reshape(cluster_image, [down_samp_shape[0], down_samp_shape[1]])
        save_pickle(cluster_image, p4_dir + 'cluster_image.pickle')

        # Optional clustering evaluation metrics
        if if_evaluate:
            curr_he_embed_pca = he_embed_qc_pca[pixels_counter[i]:pixels_counter[i + 1]]
            s1 = silhouette_score(curr_he_embed_pca, curr_uni_cluster, metric='euclidean')
            s2 = calinski_harabasz_score(curr_he_embed_pca, curr_uni_cluster)
            s3 = davies_bouldin_score(curr_he_embed_pca, curr_uni_cluster)
            print(f'''Image {i}: Segmented into {n_clusters} clusters.
            Silhouette score: {s1}
            C-H score: {s2}
            DBI: {s3}''')
            save_pickle([s1, s2, s3], p4_dir + 'clustering_metrics.pickle')

        fig = plt.figure(figsize=(plt_figsize[0], plt_figsize[1]))
        # output rgb cluster image
        cluster_image_rgb = 255 * np.ones([np.shape(cluster_image)[0], np.shape(cluster_image)[1], 3])
        for cluster in range(n_clusters):
            cluster_image_rgb[cluster_image == cluster] = color_list[cluster_color_mapping[cluster]]
        cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
        plt.imshow(cluster_image_rgb)
        plt.title('histology segmentation', fontsize=20)
        legend_x = legend_y = np.zeros(n_clusters)
        for j in range(n_clusters):
            plt.scatter(legend_x, legend_y, c=color_list_16bit[j])
        cluster_names = [f'cluster {j}' for j in range(1, n_clusters + 1)]
        plt.legend((cluster_names), fontsize=12)
        output_path, output_dpi = save_figure_safely(
            fig,
            p4_dir + f'cluster_image_num_clusters_{n_clusters}.jpg',
            format='jpg',
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0,
        )
        plt.close(fig)
        print(f'Segmentation image is stored at: {output_path} (dpi={output_dpi})')

        save_pickle(expected_metadata, p4_dir + completion_marker_name)
