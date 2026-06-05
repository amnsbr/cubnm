import os
import argparse
import numpy as np
import datalad.api

import dotenv

import prep_data_utils

dotenv.load_dotenv()
HCP_MAIN_DIR = os.path.join(os.getenv('HCP_INPUT_DATASET_PATH'), 'HCP1200')
HCP_OUTPUT_DIR = os.getenv('HCP_OUTPUT_DIR')


def post_hcp_rs(participant_label, parc, days=(1, 2), PEs=('LR', 'RL')):
    """
    Postprocesses HCP minimally processed resting-state fMRI data:
    downloads CIFTI timeseries via datalad, transforms to fs_LR space,
    parcellates, and saves compressed BOLD arrays.

    Parameters
    ----------
    participant_label: :obj:`str`
        HCP subject ID
    parc: :obj:`str` or :obj:`list` of :obj:`str`
        Parcellation name passed to :func:`prep_data_utils.parcellate_surf`
    days: :obj:`tuple` of :obj:`int`
        Resting-state session days to process (e.g. ``(1, 2)``)
    PEs: :obj:`tuple` of :obj:`str`
        Phase-encoding directions to process (e.g. ``('LR', 'RL')``)
    """
    # setup data dowload
    subdataset = os.path.join(HCP_MAIN_DIR, f'{participant_label}')
    subsubdataset = os.path.join(subdataset, 'MNINonLinear')
    # first get the subdirectories of the MNINonLinear directory
    datalad.api.get(subsubdataset, dataset=subdataset, get_data=False)
    os.chdir(subsubdataset)

    if isinstance(parc, str):
        parc = [parc]

    # get FC and FCDs for days 1 and 2 and LR and RL scans
    for scan_day in days:
        for PE in PEs:
            # create session-specific parcellated bold data
            ses = f'REST{scan_day}_{PE}'
            sub_bold_dir = os.path.join(HCP_OUTPUT_DIR, 'bold', participant_label, ses)
            os.makedirs(sub_bold_dir, exist_ok=True)
            for curr_parc in parc:
                out_prefix = f'ctx_parc-{curr_parc}'
                out_path = os.path.join(sub_bold_dir, f'{out_prefix}_desc-bold.npz')
                if os.path.exists(out_path):
                    print(f'{out_path} already exists')
                    continue
                
                # download data if not already downloaded
                cifti_path = os.path.join(subsubdataset, 'Results', f'rfMRI_{ses}', f'rfMRI_{ses}_Atlas_MSMAll_hp2000_clean.dtseries.nii')
                if not os.path.exists(os.path.join(subsubdataset, 'Results', f'rfMRI_{ses}')):
                    print(f'{ses} for {participant_label} does not exist')
                    continue
                datalad.api.get(cifti_path, dataset=subdataset)
                # parcellate
                bold_fsLR = prep_data_utils.hcp_to_fs_LR(cifti_path)
                parc_bold = prep_data_utils.parcellate_surf(bold_fsLR, curr_parc, space="fsLR", align_order=True, concat=True)
                np.savez_compressed(out_path, parc_bold.values)

    # drop the subdataset
    datalad.api.drop(subsubdataset, what='datasets', dataset=subdataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Postprocess HCP resting-state fMRI for HCP subjects using a given parcellation.'
    )
    parser.add_argument(
        '--subjects', nargs='*', default=[],
        help='Subject IDs to process. If not provided, all available subjects are processed.'
    )
    default_parcs = ['aparc', 'schaefer-100', 'schaefer-200', 'schaefer-400']
    parser.add_argument(
        '--parc', nargs='*', default=default_parcs,
        help="Parcellation name(s) (e.g. 'aparc', 'schaefer-100')."
    )
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(os.listdir(HCP_MAIN_DIR))

    for subject in subjects:
        post_hcp_rs(subject, args.parc)