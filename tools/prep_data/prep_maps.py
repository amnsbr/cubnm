import os
import argparse
import numpy as np
import nibabel
import neuromaps.datasets
import neuromaps.transforms

import prep_data_utils

SUPPORTED_MAPS = ('myelinmap', 'fcgradient01', 'yeo7')
SUPPORTED_PARCS = ('aparc', 'schaefer-100', 'schaefer-200', 'schaefer-400')

YEO7_NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

CUBNM_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'cubnm', 'data')
)


def fetch_map_fsLR(map_desc):
    """
    Fetches a surface map from neuromaps in fsLR space.

    Parameters
    ----------
    map_desc: :obj:`str`
        Map description passed to neuromaps (e.g. ``'myelinmap'``)

    Returns
    -------
    :obj:`dict` of :obj:`np.ndarray`
        Keys ``'L'`` and ``'R'`` with map values on fsLR 32k vertices
    """
    annotations = neuromaps.datasets.available_annotations(desc=map_desc)
    fslr_annotations = [a for a in annotations if a[2] == 'fsLR']
    if len(fslr_annotations) == 0:
        raise ValueError(f'no fsLR annotation for {map_desc}')
    if len(fslr_annotations) > 1:
        raise NotImplementedError(
            f'multiple fsLR annotations for {map_desc}: {fslr_annotations}'
        )
    source = fslr_annotations[0][0]
    den = fslr_annotations[0][3]
    map_paths = neuromaps.datasets.fetch_annotation(desc=map_desc, source=source)
    if isinstance(map_paths, list) and len(map_paths) > 1:
        map_paths = sorted(map_paths)
        assert 'hemi-L' in map_paths[0]
    if den != '32k':
        maps_32k = neuromaps.transforms.fslr_to_fslr(map_paths, '32k')
        map_data = {
            'L': np.squeeze(maps_32k[0].agg_data()),
            'R': np.squeeze(maps_32k[1].agg_data()),
        }
    else:
        map_data = {
            'L': np.squeeze(nibabel.load(map_paths[0]).agg_data()),
            'R': np.squeeze(nibabel.load(map_paths[1]).agg_data()),
        }
    return map_data


def prep_yeo7(parc):
    """
    Builds and saves the Yeo-7 network index map from Schaefer parcel labels.

    Parameters
    ----------
    parc: :obj:`str`
        Schaefer parcellation name (e.g. ``'schaefer-100'``)

    Returns
    -------
    :obj:`str` or :obj:`None`
        Output path if saved, else ``None`` if output already exists
    """
    if not parc.startswith('schaefer-'):
        raise ValueError('yeo7 only supports Schaefer parcellations (name starts with "schaefer-")')

    out_dir = os.path.join(CUBNM_DATA_DIR, 'maps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'ctx_parc-{parc}_desc-yeo7.txt')
    if os.path.isfile(out_path):
        print(f'Skipping: output already exists ({out_path})')
        return None

    labels = prep_data_utils.load_ordered_parcel_labels(parc)
    networks = [label.split('_')[2] for label in labels]
    indices = [YEO7_NETWORKS.index(network) for network in networks]
    np.savetxt(out_path, indices, fmt='%d')
    print(f'Saved {out_path}')
    return out_path


def prep_map(map_desc, parc):
    """
    Fetches, parcellates, and saves a heterogeneity map for the package.

    Parameters
    ----------
    map_desc: :obj:`str`
        Map name (e.g. ``'myelinmap'``, ``'yeo7'``)
    parc: :obj:`str`
        Parcellation name (e.g. ``'schaefer-100'``)

    Returns
    -------
    :obj:`str` or :obj:`None`
        Output path if saved, else ``None`` if output already exists
    """
    if map_desc not in SUPPORTED_MAPS:
        raise ValueError(f'{map_desc} not in {SUPPORTED_MAPS}')
    if parc not in SUPPORTED_PARCS:
        raise ValueError(f'{parc} not in {SUPPORTED_PARCS}')

    if map_desc == 'yeo7':
        return prep_yeo7(parc)

    out_dir = os.path.join(CUBNM_DATA_DIR, 'maps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'ctx_parc-{parc}_desc-{map_desc}.txt')
    if os.path.isfile(out_path):
        print(f'Skipping: output already exists ({out_path})')
        return None

    map_data = fetch_map_fsLR(map_desc)
    parcellated = prep_data_utils.parcellate_surf(
        map_data, parc, space='fsLR', align_order=True, concat=True,
    )
    values = parcellated.iloc[:, 0].values
    np.savetxt(out_path, values)
    print(f'Saved {out_path}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare heterogeneity maps for cubnm.'
    )
    parser.add_argument(
        '--map', nargs='*', default=list(SUPPORTED_MAPS), choices=SUPPORTED_MAPS,
        help='Map(s) to prepare (default: all supported maps).',
    )
    parser.add_argument(
        '--parc', nargs='*', default=list(SUPPORTED_PARCS), choices=SUPPORTED_PARCS,
        help='Parcellation name(s) (default: all supported parcellations).',
    )
    args = parser.parse_args()
    for map_desc in args.map:
        for parc in args.parc:
            if map_desc == 'yeo7' and not parc.startswith('schaefer-'):
                print(f'Skipping yeo7 for {parc} (Schaefer parcellations only)')
                continue
            prep_map(map_desc, parc)
