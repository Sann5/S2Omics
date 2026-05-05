########################################################################################################
########################################### warning! ###################################################
### cell label broadcasting can only be run with feature extraction conducted with down_samp_step=1 ####
########################################################################################################

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from .. import step_paths
from ..s1_utils import *  # noqa: F401,F403
from ..s2_label_broadcasting import *  # noqa: F401,F403


def label_broadcasting(WSI_save_folder, SO_save_folder,
                       SO_annotation_csv,
                       WSI_cache_path='', SO_cache_path='',
                       foundation_model='uni', device='cuda:0'):
    ''' predict cell-level labels.

    Reads ``shapes.pickle`` and ``qc_preserve_indicator.pickle`` from each
    sample's ``p2_qc/`` folder, and embedding parts (downsamp_1) from each
    sample's ``p3_features/`` folder (or the explicit ``*_cache_path``).
    The whole-slide prediction JPG and the completion marker are written to
    ``WSI_save_folder/p6_label_broadcasting/``.

    Parameters:
        WSI_save_folder: per-sample S2Omics output root for the whole-slide image
        SO_save_folder: per-sample S2Omics output root for the spatial omics image
        SO_annotation_csv: path to ``annotation_file.csv`` with columns
            ``super_pixel_x``, ``super_pixel_y``, ``annotation``
        WSI_cache_path/SO_cache_path: optional override for the directory
            containing embedding parts (must end with '/'). Defaults to the
            sample's p3_features/.
        foundation_model: foundation model name used during feature extraction
        device: torch device, default 'cuda:0'
    '''
    p6_dir = step_paths.step_dir(WSI_save_folder, step_paths.P6_LABEL_BROADCASTING)
    completion_marker = p6_dir + 'label_broadcasting_complete.pickle'
    final_output = p6_dir + 'S2Omics_whole_slide_prediction.jpg'

    expected_metadata = {
        'WSI_save_folder': os.path.abspath(WSI_save_folder),
        'SO_save_folder': os.path.abspath(SO_save_folder),
        'SO_annotation_csv': os.path.abspath(SO_annotation_csv),
        'WSI_cache_path': os.path.abspath(WSI_cache_path) if WSI_cache_path else '',
        'SO_cache_path': os.path.abspath(SO_cache_path) if SO_cache_path else '',
        'foundation_model': foundation_model,
    }
    if stage_is_complete(completion_marker, expected_metadata, required_outputs=(final_output,)):
        print('Skipping label broadcasting; final prediction already exists.')
        return

    # ---- Spatial omics inputs ----
    SO_p2_dir = step_paths.step_dir(SO_save_folder, step_paths.P2_QC, create=False)
    SO_p3_dir = step_paths.step_dir(SO_save_folder, step_paths.P3_FEATURES, create=False)

    shapes = load_pickle(SO_p2_dir + 'shapes.pickle')
    SO_image_shape = shapes['tiles']
    qc_preserve_indicator = load_pickle(SO_p2_dir + 'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, SO_image_shape)
    annotation_file = pd.read_csv(SO_annotation_csv)
    unique_cell_type = np.unique(annotation_file['annotation'])
    label_vector = np.ones(len(annotation_file), dtype='int64')
    for ct in range(len(unique_cell_type)):
        ct_index = np.arange(len(annotation_file))[annotation_file['annotation'] == unique_cell_type[ct]]
        label_vector[ct_index] = ct
    annotation_file['label'] = label_vector
    cell_type_image = -1 * np.ones(SO_image_shape)
    for i in range(len(annotation_file)):
        x = annotation_file['super_pixel_x'][annotation_file.index[i]]
        y = annotation_file['super_pixel_y'][annotation_file.index[i]]
        label = annotation_file['label'][annotation_file.index[i]]
        cell_type_image[x, y] = label
    cell_type_image = np.array(cell_type_image, dtype='int64')
    cell_type_image_mask = np.full(shapes['tiles'], False)
    cell_type_image_mask[np.where(cell_type_image > -1)] = True

    print('Loading histology feature embeddings of the Spatial Omics data...')
    SO_cache = SO_cache_path if len(SO_cache_path) > 0 else SO_p3_dir
    SO_he_embed_total = []
    i = 0
    while True:
        path = SO_cache + foundation_model + f'_embeddings_downsamp_1_part_{i}.pickle'
        if os.path.exists(path):
            SO_he_embed_total.append(load_pickle(path))
            i += 1
        else:
            break
    if not SO_he_embed_total:
        raise FileNotFoundError(
            f'No SO embedding parts found at {SO_cache} for model={foundation_model}, downsamp=1'
        )
    SO_he_embed_total = np.concatenate(SO_he_embed_total)
    print('Sucessfully loaded all histology feature embeddings of the Spatial Omics data!')

    # ---- WSI inputs ----
    WSI_p2_dir = step_paths.step_dir(WSI_save_folder, step_paths.P2_QC, create=False)
    WSI_p3_dir = step_paths.step_dir(WSI_save_folder, step_paths.P3_FEATURES, create=False)

    shapes = load_pickle(WSI_p2_dir + 'shapes.pickle')
    WSI_image_shape = shapes['tiles']
    plt_figsize = (WSI_image_shape[1] // 100, WSI_image_shape[0] // 100)
    qc_preserve_indicator = load_pickle(WSI_p2_dir + 'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, WSI_image_shape)

    print('Loading histology feature embeddings of the whole-slide H&E data...')
    WSI_cache = WSI_cache_path if len(WSI_cache_path) > 0 else WSI_p3_dir
    WSI_he_embed_total = []
    i = 0
    while True:
        path = WSI_cache + foundation_model + f'_embeddings_downsamp_1_part_{i}.pickle'
        if os.path.exists(path):
            WSI_he_embed_total.append(load_pickle(path))
            i += 1
        else:
            break
    if not WSI_he_embed_total:
        raise FileNotFoundError(
            f'No WSI embedding parts found at {WSI_cache} for model={foundation_model}, downsamp=1'
        )
    WSI_he_embed_total = np.concatenate(WSI_he_embed_total)
    print('Sucessfully loaded all histology feature embeddings of the whole-slide H&E data!')

    # construct pytorch dataset, spatial omics (histo-feature, annotation) as training data
    setup_seed(42)
    num_cell_types = len(unique_cell_type)
    train_mask = cell_type_image_mask
    train_x = SO_he_embed_total[train_mask.flatten(), :]
    train_y = np.array(cell_type_image[train_mask], dtype='int64')
    del SO_he_embed_total
    total_x = WSI_he_embed_total[qc_mask.flatten(), :]
    total_y = -1 * np.ones(np.sum(qc_mask), dtype='int64')
    del WSI_he_embed_total
    TrainSet = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
    TotalSet = TensorDataset(torch.from_numpy(total_x).float(), torch.from_numpy(total_y))
    train_loader = DataLoader(TrainSet, shuffle=True, batch_size=512, num_workers=0, drop_last=False)
    total_loader = DataLoader(TotalSet, shuffle=False, batch_size=512, num_workers=0, drop_last=False)

    print('Start training the label transferring model...')
    model = S2Omics_Predictor(
        n_input=np.shape(train_x)[1],
        n_enc_1=1024,
        n_enc_2=1024,
        n_enc_3=1024,
        n_z=256,
        n_cls_1=64,
        n_cls_out=num_cell_types + 1).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = GCE_loss(0.6, num_cell_types + 1)

    epochs = 100
    test_interval = 20
    for epoch in range(epochs):
        for (i, data) in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            x_bar, z, cls_h1, cls_out = model(inputs)

            loss_recon = 0.5 * (F.mse_loss(x_bar, inputs) + nn.L1Loss()(x_bar, inputs))
            loss_cls = criterion(cls_out, labels)
            loss = loss_recon + loss_cls
            loss.backward()
            optimizer.step()

        if (epoch + 1) % test_interval == 0:
            train_cor, train_tot = 0, 0
            with torch.no_grad():
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    _, _, _, outputs = model(inputs)
                    pred = torch.argmax(outputs, axis=1)
                    train_cor += torch.sum(pred == labels)
                    train_tot += len(labels)
            print('Epoch [%d] loss: %.3f, train accuracy %.3f' % (epoch + 1, loss.item(), train_cor / train_tot))
    print('Finished Training')

    model.eval()
    total_pred = []
    with torch.no_grad():
        for i, data in enumerate(total_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, _, _, outputs = model(inputs)
            pred = torch.argmax(outputs, axis=1)
            total_pred.append(pred)
    total_pred = np.array(torch.concat(total_pred).cpu().numpy(), dtype='int')

    # visualize the prediction results
    color_list = np.loadtxt(os.path.join(os.path.dirname(__file__), '../color_list.txt'), dtype='int').tolist()
    with open(os.path.join(os.path.dirname(__file__), '../color_list_16bit.txt'), "r", encoding="utf-8") as file:
        lines = file.readlines()
    color_list_16bit = []
    for line in lines:
        color_list_16bit.append(line.strip())
    cluster_color_mapping = np.arange(len(color_list))

    pred_image = -1 * np.ones(WSI_image_shape)
    pred_image[qc_mask] = total_pred
    pred_image_rgb = 255 * np.ones([WSI_image_shape[0], WSI_image_shape[1], 3])
    for cluster in range(num_cell_types):
        pred_image_rgb[pred_image == cluster] = color_list[cluster_color_mapping[cluster]]
    pred_image_rgb = np.array(pred_image_rgb, dtype='int')
    plt.figure(figsize=plt_figsize)
    plt.imshow(pred_image_rgb)
    legend_x = legend_y = np.zeros(num_cell_types)
    for i in range(num_cell_types):
        plt.scatter(legend_x, legend_y, c=color_list_16bit[i])
    plt.legend((unique_cell_type), fontsize=12)
    plt.savefig(final_output, format='jpg', dpi=600, bbox_inches='tight', pad_inches=0)
    print('Predicted cell type distribution for the whole-slide H&E data is stored at: ' + final_output)

    save_pickle(expected_metadata, completion_marker)
