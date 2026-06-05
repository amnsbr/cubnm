import os
import numpy as np
import pandas as pd
import nibabel
import hcp_utils

PARCELLATIONS_DIR = os.path.join(os.path.dirname(__file__), 'parcellations')

MIDLINE_PARCELS = {
    'schaefer-100': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'schaefer-200': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'schaefer-400': ['Background+FreeSurfer_Defined_Medial_Wall'],
    'aparc': ['L_unknown', 'R_unknown', 'L_corpuscallosum', 'R_corpuscallosum'],
}

def load_parcellation_map_fsLR(parcellation_name, concatenate=False, load_indices=False):
    """
    Loads parcellation maps of L and R hemispheres in fsLR space, correctly relabels them,
    and separates or concatenates them.

    Parameters
    ----------
    parcellation_name: :obj:`str`
        Parcellation scheme. Supported values:

        - ``'schaefer-N'``: Schaefer parcellation with N parcels
        - ``'aparc'``: Desikan-Killiany aparc parcellation
    concatenate: :obj:`bool`
        If True, return a single concatenated array across hemispheres
    load_indices: :obj:`bool`
        If True, keep numeric parcel indices instead of label strings

    Returns
    -------
    :obj:`np.ndarray` or :obj:`dict` of :obj:`np.ndarray`
        Parcellation map(s) per hemisphere, or concatenated across hemispheres
    """
    if parcellation_name.startswith('schaefer'):
        n_parcels = int(parcellation_name.split('-')[1])
        parcellation_path = os.path.join(
            PARCELLATIONS_DIR, 
            f'Schaefer2018_{n_parcels}Parcels_7Networks_order.dlabel.nii')
        parcellation_map = nibabel.load(parcellation_path).get_fdata().squeeze()
        if not load_indices:
            labels = load_ordered_parcel_labels(parcellation_name).tolist()
            labels = ['Background+FreeSurfer_Defined_Medial_Wall'] + labels
            transdict = dict(enumerate(labels))
            parcellation_map = np.vectorize(transdict.get)(parcellation_map)
        if concatenate:
            return parcellation_map
        else:
            return {
                'L': parcellation_map[:32492],
                'R': parcellation_map[32492:]
            }
    elif parcellation_name == 'aparc':
        parcellation_map = {}
        for hem in ['L', 'R']:
            gii = nibabel.load(os.path.join(PARCELLATIONS_DIR, 
                f'Desikan.32k.{hem}.label.gii'))
            parcellation_map[hem] = gii.agg_data().squeeze()
            if not load_indices:
                transdict = gii.labeltable.get_labels_as_dict()
                # add L_ R_ to the labels
                transdict = dict(zip(transdict.keys(), [f'{hem}_{l}' for l in transdict.values()]))
                parcellation_map[hem] = np.vectorize(transdict.get)(parcellation_map[hem])
        if concatenate:
            np.concatenate([parcellation_map['L'], parcellation_map['R']])
        else:
            return parcellation_map
    else:
        raise NotImplementedError(f"{parcellation_name} not available in fs_LR")

def load_ordered_parcel_labels(parcellation_name):
    """
    Loads the labels corresponding to the parcels in volumetric space,
    except for parcel ID 0 (background / midline).
    Currently only works for Schaefer and aparc parcellations.

    Parameters
    ----------
    parcellation_name: :obj:`str`
        Parcellation name (e.g. ``'schaefer-100'``, ``'aparc'``)

    Returns
    -------
    :obj:`np.ndarray`
        Ordered parcel labels
    """
    if 'schaefer' in parcellation_name:
        # For schaefer load the names from color tables
        n_parcels = int(parcellation_name.replace('schaefer-', ''))
        lut_path = os.path.join(
            PARCELLATIONS_DIR, 
            f'Schaefer2018_{n_parcels}Parcels_7Networks_order.txt'
        )
        labels = pd.read_csv(lut_path, sep='\t', header=None)[1].values
    elif parcellation_name == 'aparc':
        # note that order of parcels in lut_aparc_mics is the same
        # as in abagen (the source of volumetric aparc)
        lut = pd.read_csv(os.path.join(
            PARCELLATIONS_DIR, 'lut_aparc_mics.csv'
        ))
        labels = lut.loc[~lut['label'].isin(['medial_wall', 'L_corpuscallosum', 'R_corpuscallosum']),'label'].values
    else:
        raise NotImplementedError
    return labels

def parcellate_surf(
        surface_data, 
        parcellation_name="schaefer-100", 
        method="mean", 
        midline="drop",
        space="fsLR",
        align_order=True,
        concat=False,
    ):
    """
    Parcellates surface data using a parcellation by aggregating vertices
    within each parcel.

    Parameters
    ----------
    surface_data: :obj:`dict` of :obj:`np.ndarray`
        n_vertices x n_features surface data of L and R hemispheres
    parcellation_name: :obj:`str`
        Parcellation name (e.g. ``'schaefer-100'``, ``'aparc'``)
    method: :obj:`str`, {'mean', 'median', 'sum'}
        Method of aggregating over vertices within a parcel
    midline: :obj:`str` or :obj:`None`, {None, 'drop', 'nan'}
        How to handle midline parcels:

        - ``None``: keep midline as is
        - ``'drop'``: drop midline parcels
        - ``'nan'``: set midline parcels to NaN
    space: :obj:`str`
        Surface space. Only ``'fsLR'`` is supported
    align_order: :obj:`bool`
        Align parcel order to other parcellated data (based on micapipe lut files)
    concat: :obj:`bool`
        If True, concatenate L and R hemispheres into a single DataFrame

    Returns
    -------
    :obj:`pd.DataFrame` or :obj:`dict` of :obj:`pd.DataFrame`
        n_parcels x n_features parcellated data per hemisphere, or concatenated
    """
    assert space == 'fsLR', "Only fsLR space is supported for now"
    # load parcellation map
    labeled_parcellation_maps = load_parcellation_map_fsLR(parcellation_name, concatenate=False)
    # load micapipe lut for correct ordering of parcels
    lut = pd.read_csv(
        os.path.join(
            PARCELLATIONS_DIR, 
            f'lut_{parcellation_name}_mics.csv')
            ).set_index('label')
    lut_cortical = lut.loc[(lut['mics']>=1000)]
    parcellated_data = {}
    for hem in ['L', 'R']:
        # parcellate
        parcellated_vertices = (
            pd.DataFrame(surface_data[hem], index=labeled_parcellation_maps[hem])
        )
        parcellated_vertices = (parcellated_vertices
            .reset_index(drop=False)
            .groupby('index')
        )
        # operate on groupby object if needed
        if method == 'median':
            parcellated_data[hem] = parcellated_vertices.median()
        elif method == 'mean':
            parcellated_data[hem] = parcellated_vertices.mean()
        elif method == 'sum':
            parcellated_data[hem] = parcellated_vertices.sum()
        # remove midline data
        if midline == 'drop':
            parcellated_data[hem] = parcellated_data[hem].drop(
                index=parcellated_data[hem].index.intersection(MIDLINE_PARCELS[parcellation_name]))
        elif midline == 'nan':
            parcellated_data[hem].loc[
                parcellated_data[hem].index.intersection(MIDLINE_PARCELS[parcellation_name])
            ] = np.NaN
        if align_order:
            # correctly order parcels
            # Warning: this removes non-cortical parcels that may be in the parcellation
            ordered_parcels = lut_cortical.index.intersection(parcellated_data[hem].index)
        else:
            ordered_parcels = parcellated_data[hem].index
        parcellated_data[hem] = parcellated_data[hem].loc[ordered_parcels]
    if concat:
        return pd.concat([parcellated_data['L'], parcellated_data['R']])
    else:
        return parcellated_data


def hcp_to_fs_LR(cifti_img_path):
    """
    Transforms an HCP fMRI CIFTI image to fs_LR surface space.

    Parameters
    ----------
    cifti_img_path: :obj:`str`
        Path to the ``.dtseries.nii`` file

    Returns
    -------
    :obj:`dict` of :obj:`np.ndarray`
        Keys ``'L'`` and ``'R'`` with fMRI activity on the 32492 fs_LR
        vertices per hemisphere. Shape: (n_vertices, n_timepoints)
    """
    cifti_data = nibabel.load(cifti_img_path).get_data()
    if cifti_data.ndim ==1:
        cifti_data = cifti_data.reshape(1,-1)
    lh_fs_LR = np.zeros([cifti_data.shape[0], 32492])* np.NaN
    lh_fs_LR[:, hcp_utils.vertex_info['grayl']] = cifti_data[:,hcp_utils.struct['cortex_left']]
    rh_fs_LR = np.zeros([cifti_data.shape[0], 32492])* np.NaN
    rh_fs_LR[:, hcp_utils.vertex_info['grayr']] = cifti_data[:,hcp_utils.struct['cortex_right']]
    # transpose to have vertices x timepoints
    fs_LR = {'L': lh_fs_LR.T, 'R': rh_fs_LR.T}
    return fs_LR