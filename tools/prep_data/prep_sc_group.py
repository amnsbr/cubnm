import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()
HCP_OUTPUT_DIR = os.getenv('HCP_OUTPUT_DIR')
CUBNM_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'cubnm', 'data')
)

MEASURES = ('strength', 'length')


def prep_sc_group(subjects, parc, group, hemi=None, exc_subcortex=True):
    """
    Computes and saves group-level SC strength and length for a subject list.

    Median-pools per-subject SC CSVs from ``$HCP_OUTPUT_DIR/SC/`` and writes
    group connectomes to ``src/cubnm/data/hcp/sc/{group}/``.

    Parameters
    ----------
    subjects: :obj:`list` of :obj:`str`
        HCP subject IDs
    parc: :obj:`str`
        Parcellation name (e.g. ``'schaefer-100'``)
    group: :obj:`str`
        Group output name (e.g. ``'group-test303'``)
    hemi: :obj:`str` or :obj:`None`
        Hemisphere tag passed to subject SC paths, if any
    exc_subcortex: :obj:`bool`
        If True, use cortex-only subject SC files
    """
    out_prefix = 'ctx' if exc_subcortex else 'sctx_ctx'
    out_prefix += f'_parc-{parc}'
    if hemi:
        out_prefix += f'_hemi-{hemi}'

    print('Computing group SC (median across subjects)...')
    group_sc = {}
    for measure in MEASURES:
        mats = []
        index = None
        columns = None
        for sub in tqdm(subjects, desc=f'Loading {measure}'):
            path = os.path.join(
                HCP_OUTPUT_DIR, 'SC', sub,
                out_prefix + f'_desc-{measure}.csv',
            )
            if not os.path.isfile(path):
                print(f'Missing {path}, skipping')
                continue
            sc = pd.read_csv(path, index_col=0)
            if mats and sc.values.shape != mats[0].shape:
                raise ValueError(
                    f'Shape mismatch for {sub} {measure}: '
                    f'{sc.values.shape} vs {mats[0].shape}'
                )
            if index is None:
                index = sc.index
                columns = sc.columns
            mats.append(sc.values)
        if len(mats) == 0:
            raise RuntimeError(f'No subjects with {measure} data')
        median = np.median(np.array(mats), axis=0)
        group_sc[measure] = pd.DataFrame(median, index=index, columns=columns)

    out_dir = os.path.join(CUBNM_DATA_DIR, 'hcp', 'sc', group)
    os.makedirs(out_dir, exist_ok=True)
    for measure in MEASURES:
        out_path = os.path.join(out_dir, out_prefix + f'_desc-{measure}.csv')
        group_sc[measure].to_csv(out_path)
        print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Compute group-level SC strength and length by median-pooling subject SC CSVs.'
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
            'Group output name (e.g. group-test303). '
            'Defaults to group-{stem} from the subjects filename.'
        ),
    )
    parser.add_argument(
        '--hemi', default=None, choices=['L', 'R'],
        help='Use hemisphere-restricted subject SC files. By default both hemispheres are used.',
    )
    parser.add_argument(
        '--inc-subcortex', action='store_true', default=False,
        help='Use subject SC files that include subcortical parcels (excluded by default).',
    )
    args = parser.parse_args()

    if args.group:
        group = args.group
    else:
        stem = os.path.splitext(os.path.basename(args.subjects))[0]
        if stem.startswith('subjects-'):
            stem = stem[len('subjects-'):]
        group = f'group-{stem}'

    exc_subcortex = not args.inc_subcortex

    with open(args.subjects) as f:
        subjects = [line.strip() for line in f if line.strip()]

    prep_sc_group(subjects, args.parc, group, hemi=args.hemi, exc_subcortex=exc_subcortex)
