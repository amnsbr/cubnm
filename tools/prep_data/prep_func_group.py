import os
import argparse
import numpy as np
from tqdm import tqdm
import dotenv

from cubnm.utils import calculate_fc, calculate_fcd

dotenv.load_dotenv()
HCP_OUTPUT_DIR = os.getenv('HCP_OUTPUT_DIR')
CUBNM_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'cubnm', 'data')
)

FCD_WINDOW_SIZE = 30
FCD_WINDOW_STEP = 5
TR = 0.72
FCD_WINDOW_SIZE_TR = int(np.round(FCD_WINDOW_SIZE / (TR*2))) * 2
FCD_WINDOW_STEP_TR = int(np.round(FCD_WINDOW_STEP / TR))
SCAN_DAYS = (1, 2)
PEs = ('LR', 'RL')
SES = 'REST'


def pool_fc_trils(in_fc_trils):
    """
    Pools FC lower-triangular vectors via Fisher's z-transform.

    Parameters
    ----------
    in_fc_trils: :obj:`list` of :obj:`np.ndarray`
        FC tril vectors to pool

    Returns
    -------
    :obj:`np.ndarray`
        Pooled FC tril vector
    """
    z_FCs = []
    for fc_tril in in_fc_trils:
        fc = np.asarray(fc_tril, dtype=float).copy()
        fc[np.isclose(fc, 1)] = 0
        z_FCs.append(np.arctanh(fc)[:, np.newaxis])
    z_FCs = np.concatenate(z_FCs, axis=1)
    pooled_z_FC = z_FCs.mean(axis=1)
    pooled_r_FC = np.tanh(pooled_z_FC)
    return pooled_r_FC


def pool_fcd_trils(in_fcd_trils, downsample=True):
    """
    Pools FCD lower-triangular vectors by concatenation (and
    sorted downsampling).

    Parameters
    ----------
    in_fcd_trils: :obj:`list` of :obj:`np.ndarray`
        FCD tril vectors to pool
    downsample: :obj:`bool`
        If True, downsample the sorted pooled tril by keeping every
        ``len(in_fcd_trils)``-th element

    Returns
    -------
    :obj:`np.ndarray`
        Pooled FCD tril vector
    """
    pooled_FCD_tril = np.concatenate(in_fcd_trils)
    if downsample:
        pooled_FCD_tril = np.sort(pooled_FCD_tril)[::len(in_fcd_trils)]
    return pooled_FCD_tril


def tril_to_fc(fc_tril, nodes, exc_inter=False):
    """
    Reconstructs a symmetric FC matrix from a lower-triangular vector.

    Parameters
    ----------
    fc_tril: :obj:`np.ndarray`
        FC lower-triangular vector
    nodes: :obj:`int`
        Number of parcels
    exc_inter: :obj:`bool`
        If True, fill intrahemispheric blocks only and set interhemispheric
        entries to NaN

    Returns
    -------
    :obj:`np.ndarray`
        Symmetric FC matrix with diagonal set to 1
    """
    fc = np.zeros((nodes, nodes), dtype=float)
    if exc_inter:
        half_nodes = nodes // 2
        fc[:half_nodes, :half_nodes][np.tril_indices(half_nodes, -1)] = (
            fc_tril[:fc_tril.shape[0] // 2]
        )
        fc[half_nodes:, half_nodes:][np.tril_indices(half_nodes, -1)] = (
            fc_tril[fc_tril.shape[0] // 2:]
        )
        fc[:half_nodes, half_nodes:] = np.nan
        fc[half_nodes:, :half_nodes] = np.nan
    else:
        fc[np.tril_indices(nodes, -1)] = fc_tril
    fc += fc.T
    np.fill_diagonal(fc, 1.0)
    return fc


def compute_group_measures(subjects, parc, exc_inter):
    """
    Computes group-level FC and FCD from parcellated resting-state BOLD.

    Per session, computes FC and FCD trils, then pools at two levels: within
    subjects (across sessions) and across subjects.

    Parameters
    ----------
    subjects: :obj:`list` of :obj:`str`
        HCP subject IDs
    parc: :obj:`str`
        Parcellation name (e.g. ``'schaefer-100'``)
    exc_inter: :obj:`bool`
        If True, exclude interhemispheric connections

    Returns
    -------
    :obj:`np.ndarray`
        Group-level FC matrix
    :obj:`np.ndarray`
        Group-level FCD tril vector
    """
    nodes = None
    fc_trils = {}
    fcd_trils = {}
    for sub in tqdm(subjects, desc='Computing subject FC/FCD trils'):
        fc_trils[sub] = []
        fcd_trils[sub] = []
        for scan_day in SCAN_DAYS:
            for pe in PEs:
                ses = f'REST{scan_day}_{pe}'
                bold_path = os.path.join(
                    HCP_OUTPUT_DIR, 'bold', sub, ses,
                    f'ctx_parc-{parc}_desc-bold.npz',
                )
                if not os.path.isfile(bold_path):
                    print(f'Missing {bold_path}, skipping')
                    continue
                bold = np.load(bold_path)['arr_0']
                if nodes is None:
                    nodes = bold.shape[0]
                fc_trils[sub].append(
                    calculate_fc(bold.copy(), exc_interhemispheric=exc_inter, return_tril=True)
                )
                fcd_trils[sub].append(
                    calculate_fcd(
                        bold.copy(),
                        FCD_WINDOW_SIZE_TR,
                        FCD_WINDOW_STEP_TR,
                        exc_interhemispheric=exc_inter,
                        return_tril=True,
                    )
                )
        fc_trils[sub] = pool_fc_trils(fc_trils[sub])
        fcd_trils[sub] = pool_fcd_trils(fcd_trils[sub])
    if len(fc_trils) == 0:
        raise RuntimeError('No subjects with data from both scan days')
    all_fc_tril = pool_fc_trils(list(fc_trils.values()))
    all_fcd_tril = pool_fcd_trils(list(fcd_trils.values()))
    all_fc = tril_to_fc(all_fc_tril, nodes, exc_inter=exc_inter)
    return all_fc, all_fcd_tril


def save_group_measures(group, parc, exc_inter, fc, fcd_tril):
    """
    Saves group-level FC and FCD to ``src/cubnm/data/hcp/``.

    Parameters
    ----------
    group: :obj:`str`
        Group name (e.g. ``'group-train706'``)
    parc: :obj:`str`
        Parcellation name
    exc_inter: :obj:`bool`
        If True, use the ``_exc-inter`` filename suffix
    fc: :obj:`np.ndarray`
        Group-level FC matrix
    fcd_tril: :obj:`np.ndarray`
        Group-level FCD tril vector
    """
    opts = '_exc-inter' if exc_inter else ''
    for measure, data, desc in [
        ('fc', fc, 'fc'),
        ('fcd', fcd_tril, 'fcdtril'),
    ]:
        out_dir = os.path.join(CUBNM_DATA_DIR, 'hcp', measure, group, SES)
        os.makedirs(out_dir, exist_ok=True)
        if measure == 'fcd':
            curr_opts = opts + f'_window-{FCD_WINDOW_SIZE}_step-{FCD_WINDOW_STEP}'
        else:
            curr_opts = opts
        out_path = os.path.join(
            out_dir,
            f'ctx_parc-{parc}{curr_opts}_desc-{desc}.npz',
        )
        np.savez_compressed(out_path, data)
        print(f'Saved {out_path}')


def prep_func_group(subjects, parc, group):
    """
    Computes and saves group-level FC and FCD for a subject list.

    Runs pooling with and without interhemispheric connections excluded.

    Parameters
    ----------
    subjects: :obj:`list` of :obj:`str`
        HCP subject IDs
    parc: :obj:`str`
        Parcellation name (e.g. ``'schaefer-100'``)
    group: :obj:`str`
        Group output name (e.g. ``'group-train706'``)
    """
    for exc_inter in (True, False):
        print(f'Computing group measures (exc_interhemispheric={exc_inter})...')
        fc, fcd_tril = compute_group_measures(subjects, parc, exc_inter)
        save_group_measures(group, parc, exc_inter, fc, fcd_tril)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Compute and pool group-level FC and FCD from parcellated HCP resting-state BOLD.'
        ),
    )
    parser.add_argument(
        '--subjects', required=True,
        help='Path to a text file with one subject ID per line.',
    )
    parser.add_argument(
        '--parc', default='schaefer-100',
        help="Parcellation name (e.g. 'schaefer-100').",
    )
    parser.add_argument(
        '--group', default=None,
        help=(
            'Group output name (e.g. group-train706). '
            'Defaults to group-{stem} from the subjects filename.'
        ),
    )
    args = parser.parse_args()

    if args.group:
        group = args.group
    else:
        stem = os.path.splitext(os.path.basename(args.subjects))[0]
        if stem.startswith('subjects-'):
            stem = stem[len('subjects-'):]
        group = f'group-{stem}'

    subjects = np.loadtxt(args.subjects, dtype=str)

    prep_func_group(subjects, args.parc, group)
