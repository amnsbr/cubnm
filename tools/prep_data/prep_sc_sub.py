import os
import argparse
import numpy as np
import pandas as pd
import dotenv

dotenv.load_dotenv()
HCP_SC_INPUT_DIR = os.getenv('HCP_SC_DATASET_PATH')
HCP_OUTPUT_DIR = os.getenv('HCP_OUTPUT_DIR')
PARCELLATIONS_DIR = os.path.join(os.path.dirname(__file__), 'parcellations')

def label_sc(
        participant_label, 
        parc, 
        measure, 
        hemi=None, 
        exc_subcortex=True
    ):
    """
    Labels structural connectivity with the given parcellation and excludes
    unwanted parcels.

    Parameters
    ----------
    participant_label: :obj:`str`
        HCP subject ID
    parc: :obj:`str`
        Parcellation name (e.g. ``'aparc'``, ``'schaefer-100'``)
    measure: :obj:`str`, {'strength', 'length'}
        SC measure to load from HCP output files
    hemi: :obj:`str` or :obj:`None`, {'L', 'R'}
        If set, retain only the left or right hemisphere. ``None`` keeps both
    exc_subcortex: :obj:`bool`
        If True, exclude subcortical parcels from the output connectome

    Returns
    -------
    :obj:`pd.DataFrame` or :obj:`None`
        Symmetric connectome with parcel labels as row and column names.
        Returns ``None`` if the output already exists or the input file is missing
    """
    # covert parcellation name to names used in HCP SC data
    if parc == 'aparc':
        hcp_parc_name = 'DesikanKilliany_68Parcels'
    elif 'schaefer' in parc:
        n = parc.split('-')[-1]
        hcp_parc_name = f'Schaefer2018_{n}Parcels_7Networks'
    else:
        raise NotImplementedError(f"{parc} not available")
    # read raw full SC (including cortex+subcortex)
    full_SC_prefix = os.path.join(
        HCP_SC_INPUT_DIR, 'output', 'SC', 'HCP', participant_label, 
        f'{hcp_parc_name}_10M_native')
    if measure == 'strength':
        full_SC_path = full_SC_prefix + '_count.csv'
    elif measure == 'length':
        full_SC_path = full_SC_prefix + '_length.csv'
    sub_out_dir = os.path.join(HCP_OUTPUT_DIR, 'SC', participant_label)
    out_prefix = f'ctx' if exc_subcortex else f'sctx_ctx'
    out_prefix += f'_parc-{parc}'
    if hemi:
        out_prefix += f'_hemi-{hemi}'
    out_path = os.path.join(sub_out_dir, out_prefix + f'_desc-{measure}.csv')
    if os.path.isfile(out_path):
        print(f'Skipping {participant_label}: output already exists ({out_path})')
        return None
    if not os.path.isfile(full_SC_path):
        print(f'Skipping {participant_label}: input not found ({full_SC_path})')
        return None
    print(f'Processing {participant_label}...')
    full_SC = np.loadtxt(full_SC_path, delimiter=',')
    # label full SC
    lut_sctx_mics = pd.read_csv(os.path.join(PARCELLATIONS_DIR, 'lut_subcortical-cerebellum_mics.csv'))
    lut_parc_mics = pd.read_csv(os.path.join(PARCELLATIONS_DIR, f'lut_{parc}_mics.csv'))
    lut_full = pd.concat([lut_parc_mics, lut_sctx_mics], axis=0)
    ## remove parcels not included in HCP full SC
    ## cerebellum (mics 100-1000) and midline (mics 1000, 2000)
    exc_parcels = lut_full.loc[(lut_full['mics'] >= 100) & (lut_full['mics'] <= 1000), 'mics'].values.tolist()
    exc_parcels.append(2000)
    ## for aparc remove L/R_corpuscallosum
    if parc == 'aparc':
        exc_parcels += [1004, 2004]
    full_SC_labels = lut_full.loc[~lut_full['mics'].isin(exc_parcels), 'label'].values
    full_SC_labeled = pd.DataFrame(full_SC, index=full_SC_labels, columns=full_SC_labels)
    # specify parcels to exclude
    ## exclude subcortex (mics 10-100) if indicated
    exc_labels = []
    if exc_subcortex:
        exc_labels += lut_full.loc[(lut_full['mics']) < 100, 'label'].values.tolist()
    ## exclude the cortical and subcortical parcels from the other hemisphere
    if hemi == 'L':
        exc_labels += lut_full.loc[(lut_full['mics']>=49) & (lut_full['mics']<100), 'label'].values.tolist()
        exc_labels += lut_full.loc[(lut_full['mics']>2000), 'label'].values.tolist()
    elif hemi == 'R':
        exc_labels += lut_full.loc[(lut_full['mics']<49), 'label'].values.tolist()
        exc_labels += lut_full.loc[(lut_full['mics']>1000) & (lut_full['mics']<2000), 'label'].values.tolist()
    # drops duplicates and get shared with parcels that actually exist in SC
    exc_labels = list(set(exc_labels) & set(full_SC_labeled.index))
    # get selected parcels and set diagonal to zero 
    SC = full_SC_labeled.drop(index=exc_labels, columns=exc_labels)
    SC.values[np.diag_indices_from(SC)] = 0

    # save
    os.makedirs(sub_out_dir, exist_ok=True)
    SC.to_csv(out_path)
    return SC


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Label SC matrices for HCP subjects using a given parcellation.'
    )
    parser.add_argument(
        '--subjects', nargs='*', default=[],
        help='Subject IDs to process. If not provided, all available subjects are processed.'
    )
    parser.add_argument(
        '--parc', required=True,
        help="Parcellation name (e.g. 'aparc', 'schaefer-100')."
    )
    parser.add_argument(
        '--measure', required=True, choices=['strength', 'length'],
        help='SC measure to use.'
    )
    parser.add_argument(
        '--hemi', default=None, choices=['L', 'R'],
        help='Restrict to a single hemisphere. By default both hemispheres are used.'
    )
    parser.add_argument(
        '--inc-subcortex', action='store_true', default=False,
        help='Include subcortical parcels (subcortex is excluded by default).'
    )
    args = parser.parse_args()

    exc_subcortex = not args.inc_subcortex

    if args.subjects:
        subjects = args.subjects
    else:
        hcp_sc_dir = os.path.join(HCP_SC_INPUT_DIR, 'output', 'SC', 'HCP')
        subjects = sorted(os.listdir(hcp_sc_dir))

    for subject in subjects:
        label_sc(subject, args.parc, args.measure, hemi=args.hemi, exc_subcortex=exc_subcortex)