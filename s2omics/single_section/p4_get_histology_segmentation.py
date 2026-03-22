import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, BisectingKMeans
from skfuzzy.cluster import cmeans
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from ..s1_utils import (
    load_pickle, save_figure_safely, save_pickle, setup_seed)


def _normalize_save_folder(save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = save_folder + '/'
    if not os.path.exists(save_folder + 'image_files'):
        os.makedirs(save_folder + 'image_files')
    if not os.path.exists(save_folder + 'pickle_files'):
        os.makedirs(save_folder + 'pickle_files')
    return save_folder


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


def fit_global_pca_for_samples(
    save_folder_list,
    foundation_model='uni',
    down_samp_step=10,
    n_components=80,
    pca_save_path='',
):
    '''
    Fit one global PCA model across multiple samples using QC-filtered embeddings.
    Parameters:
        save_folder_list: list of per-sample S2Omics output folders
        foundation_model: foundation model used during step 3
        down_samp_step: down-sampling step used during step 3
        n_components: number of PCA components
        pca_save_path: optional output path for serialized PCA model
    Returns:
        fitted PCA encoder
    '''
    if len(save_folder_list) == 0:
        raise ValueError('save_folder_list is empty.')

    he_embed_qc_all = []

    for save_folder in save_folder_list:
        save_folder = _normalize_save_folder(save_folder)
        pickle_folder = save_folder + 'pickle_files/'

        shapes = load_pickle(pickle_folder + 'shapes.pickle')
        image_shape = shapes['tiles']
        qc_preserve_indicator = load_pickle(pickle_folder + 'qc_preserve_indicator.pickle')
        qc_mask = np.reshape(qc_preserve_indicator, image_shape)

        cache_path = pickle_folder
        he_embed_total = _load_embedding_parts(cache_path, foundation_model, down_samp_step)

        down_samp_mask = np.full(image_shape, False)
        down_samp_shape = [(image_shape[0] - 1) // down_samp_step + 1, (image_shape[1] - 1) // down_samp_step + 1]
        for i in range(down_samp_shape[0]):
            for j in range(down_samp_shape[1]):
                down_samp_mask[i * down_samp_step, j * down_samp_step] = True

        he_embed_qc = he_embed_total[qc_mask[down_samp_mask]]
        he_embed_qc_all.append(he_embed_qc)

    he_embed_qc_all = np.concatenate(he_embed_qc_all)
    print(f'Fitting global PCA on {he_embed_qc_all.shape[0]} superpixels from {len(save_folder_list)} samples...')

    pca_encoder = PCA(n_components=n_components)
    pca_encoder.fit(he_embed_qc_all)
    explained_var = np.sum(pca_encoder.explained_variance_ratio_)
    print(f'Global PCA explained variance ratio sum: {explained_var:.4f}')

    if len(pca_save_path) > 0:
        save_pickle(pca_encoder, pca_save_path)
        print(f'Saved global PCA model to: {pca_save_path}')

    return pca_encoder

def get_histology_segmentation(prefix, save_folder,
                              foundation_model='uni', cache_path='',
                              down_samp_step=10, clustering_method='kmeans',
                              n_clusters=20, resolution=1.0,
                              if_evaluate=False, pca_encoder=None,
                              pca_model_path=''):
    '''
    extracting hierarchical features of superpixels using a modified version of UNI
    Parameters:
        prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
        save_folder: the name of save folder
        foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath
        cache_path: the path to exatracted feature embedding files
        down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate
        clustering_method: the clustering method used for H&E image segmentation, user can select among 
            'kmeans': k-means++, 'fcm': fuzzy c-means, 'louvain': Louvain algorithm, 'leiden': Leiden algorithm 
            default = 'kmeans'
        n_clusters: initial number of clusters for histology segmentation when using kmeans or fcm for clustering. 
            Please notice that this is not the final number of clusters when clustering method is fcm.
        resolution: resolution for leiden algorithm, default=1.0
        if_evaluate: if evaluate the clustering results by quantitative metrics, default=False
        pca_encoder: optional pre-fitted PCA encoder; if provided, local PCA fitting is skipped
        pca_model_path: optional path to serialized PCA encoder (used when pca_encoder is None)
    '''
    save_folder = _normalize_save_folder(save_folder)
    image_folder = save_folder + 'image_files/'
    pickle_folder = save_folder + 'pickle_files/'
    
    # load in previously obtained params
    shapes = load_pickle(pickle_folder+'shapes.pickle')
    image_shape = shapes['tiles']
    dpi = 1200
    length = np.max(image_shape)//100
    plt_figsize = (image_shape[1]//100,image_shape[0]//100)
    if dpi*length > np.power(2,16):
        reduce_ratio = np.power(2,16)/(dpi*length)
        plt_figsize = ((image_shape[1]*reduce_ratio)//100,(image_shape[0]*reduce_ratio)//100)
    qc_preserve_indicator =load_pickle(pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, image_shape)
    
    # load in histology features
    print('Loading histology feature embeddings...')
    if len(cache_path) > 0:
        cache_path = cache_path
    else:
        cache_path = pickle_folder
    he_embed_total = _load_embedding_parts(cache_path, foundation_model, down_samp_step)
    print('Sucessfully loaded and normalized all histology feature embeddings!')

    # define color palette
    color_list = np.loadtxt(os.path.join(os.path.dirname(__file__), '../color_list.txt'), dtype='int').tolist()
    with open(os.path.join(os.path.dirname(__file__), '../color_list_16bit.txt'), "r", encoding="utf-8") as file:
        lines = file.readlines()
    color_list_16bit = []
    for line in lines:
        color_list_16bit.append(line.strip())
    cluster_color_mapping = np.arange(len(color_list))
    colors = np.array(color_list_16bit)[cluster_color_mapping]

    setup_seed(42)
    # create a mask for down-sampled superpixels in all superpixels
    down_samp_mask = np.full(image_shape, False)
    down_samp_shape = [(image_shape[0]-1)//down_samp_step+1, (image_shape[1]-1)//down_samp_step+1]
    for i in range(down_samp_shape[0]):
        for j in range(down_samp_shape[1]):
            down_samp_mask[i*down_samp_step,j*down_samp_step] = True
    
    # PCA+kmeans to cluster the superpixels into morphology clusters
    he_embed_qc = he_embed_total[qc_mask[down_samp_mask]]
    del he_embed_total
    if pca_encoder is None and len(pca_model_path) > 0:
        pca_encoder = load_pickle(pca_model_path)
        print(f'Loaded external PCA model from: {pca_model_path}')

    if pca_encoder is None:
        pca_encoder = PCA(n_components=80)
        pca_encoder.fit(he_embed_qc)
        print('Using per-sample PCA (default behavior).')
    else:
        print('Using provided global PCA model.')

    he_embed_qc_pca = pca_encoder.transform(he_embed_qc)

    print(f'Start segmenting the histology image, clustering method: {clustering_method}')
    if clustering_method == 'kmeans':
        uni_cluster = KMeans(n_clusters=n_clusters).fit_predict(he_embed_qc_pca).astype('int')
    if clustering_method == 'fcm':
        train = he_embed_qc_pca.T
        center, u,_,_,_,_,_ = cmeans(train, m=2, c=n_clusters, error=0.005, maxiter=1000)
        for i in u:
            uni_cluster = np.argmax(u, axis=0)
    if clustering_method == 'agglo':
        uni_cluster = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(he_embed_qc_pca).astype('int')
    if clustering_method == 'bisect':
        uni_cluster = BisectingKMeans(n_clusters=n_clusters).fit_predict(he_embed_qc_pca).astype('int')
    if clustering_method == 'birch':
        uni_cluster = Birch(n_clusters=n_clusters).fit_predict(he_embed_qc_pca).astype('int')
    if clustering_method == 'louvain':
        adata = sc.AnnData(he_embed_qc_pca)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
        sc.tl.louvain(adata, resolution=resolution)
        uni_cluster = adata.obs['louvain']
    if clustering_method == 'leiden':
        adata = sc.AnnData(he_embed_qc_pca)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
        sc.tl.leiden(adata, resolution=resolution)
        uni_cluster = adata.obs['leiden']
    
    cluster_image = -1*np.ones(image_shape)
    cluster_image[qc_mask & down_samp_mask] = uni_cluster
    cluster_image = cluster_image[down_samp_mask]
    cluster_image = np.reshape(cluster_image, [down_samp_shape[0],down_samp_shape[1]])
    histology_clusters_new = np.unique(cluster_image[cluster_image>-1])
    n_clusters = len(histology_clusters_new)
    cluster_image_copy = cluster_image.copy()
    for i in range(n_clusters):
        cluster_image[cluster_image_copy==histology_clusters_new[i]] = i
    save_pickle(cluster_image, pickle_folder+'cluster_image.pickle')
    
    cluster_vector = cluster_image[cluster_image>-1]
    cluster_centroids = []
    for cluster in range(n_clusters):
        cluster_centroid = np.mean(he_embed_qc[cluster_vector==cluster], axis=0)
        #print(cluster, cluster_centroid)
        cluster_centroids.append(cluster_centroid)
    Z = linkage(cluster_centroids, 'average')
    save_pickle(Z, pickle_folder+'linkage_matrix.pickle')

    if if_evaluate:
        s1 = silhouette_score(he_embed_qc_pca, uni_cluster, metric='euclidean') 
        s2 = calinski_harabasz_score(he_embed_qc_pca, uni_cluster) 
        s3 = davies_bouldin_score(he_embed_qc_pca, uni_cluster)
        print(f'''Finish segmentation. Segmented the histology image into {n_clusters} clusters. 
        Sihouette score: {s1}
        C-H score: {s2}
        DBI: {s3}''')
        save_pickle([s1,s2,s3], pickle_folder+'clustering_metrics.pickle')

    fig = plt.figure(figsize=(plt_figsize[0],plt_figsize[1]))
    # output rgb cluster image
    cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],np.shape(cluster_image)[1],3])
    for cluster in range(n_clusters):
        cluster_image_rgb[cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
    cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
    plt.imshow(cluster_image_rgb)
    plt.title('histology segmentation', fontsize=20)
    legend_x = legend_y = np.zeros(n_clusters)
    for i in range(n_clusters):
        plt.scatter(legend_x, legend_y, c=color_list_16bit[i])
    cluster_names = [f'cluster {i}' for i in range(1, n_clusters+1)]
    plt.legend((cluster_names), fontsize=12)
    output_path, output_dpi = save_figure_safely(
        fig,
        image_folder+f'cluster_image_num_clusters_{n_clusters}.jpg',
        format='jpg',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.close(fig)
    print(f'Segmentation image is stored at: {output_path} (dpi={output_dpi})')
