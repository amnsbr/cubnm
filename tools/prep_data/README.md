# Preparation of the data included in the package

This directory contains scripts to prepare the data included in the cuBNMpackage.

## Structural Connectivity (SC)

The SC data is prepared by running the [vbc-mri-pipeline](https://jugit.fz-juelich.de/inm7/public/vbc-mri-pipeline.git) on the [HCP open access data](https://github.com/datalad-datasets/human-connectome-project-openaccess), using
parameters defined in e.g. "vbc_mri_pipeline_example_input.txt". This data is currently stored on a local cluster and is not accessible publicly. Therefore this script can only be run by maintainers who have access to the cluster.

To prepare the SC data, after defining the variables `$HCP_SC_DATASET_PATH` and `$HCP_SC_DATASET_URL` in the `.env` file (at project root) and installing `datalad` (e.g. via miniforge), run the following command to download the processed SC data:

```bash
bash download_sc.sh
```

Then label and save the SC data as CSV files in the `$HCP_OUTPUT_DIR/SC` directory:

```bash
uv run prep_sc_sub.py --parc <parc> --measure <measure> [--inc-subcortex]
```

where `<parc>` is the parcellation name (e.g. "aparc", "schaefer-100") and `<measure>` is the SC measure to use (e.g. "strength", "length").

The `--inc-subcortex` argument can be used to create SC files that include subcortical parcels.

To median-pool subject SC matrices into a group connectome and save them under `src/cubnm/data/hcp/sc/`:

```bash
uv run prep_sc_group.py --subjects <subjects_list.txt> --parc <parc>
```

## Functional data

For the functional data, minimally processed resting-state BOLD data from the HCP open access dataset is used without any further preprocessing.

To prepare the functional data, after defining the variables `$HCP_INPUT_DATASET_PATH` and `$HCP_OUTPUT_DIR` in the `.env` file (at project root) and installing `datalad` (e.g. via miniforge), run the following command to download the minimally processed resting-state BOLD data:

```bash
bash download_func.sh
```

Then parcellate individual subject-session BOLD data by running:

```bash
bash run_prep_func_sub_all.sh
```

Lastly, compute group-level FC and FCD trils and save them under `src/cubnm/data/hcp/`:

```bash
uv run prep_func_group.py --subjects <subjects_list.txt> --parc <parc>
```

where `<subjects_list.txt>` is a text file with one subject ID per line (e.g. see [`subjects-train706.txt`](subjects-train706.txt)) and `<parc>` is the parcellation name (e.g. "aparc", "schaefer-100").

## Example subject data

To copy subject-level BOLD and SC data from `$HCP_OUTPUT_DIR` into `src/cubnm/data/hcp/`:

```bash
bash copy_sub_data.sh --subjects 100206 100307
```

## Maps

Heterogeneity maps used by ``cubnm.datasets.load_maps`` are stored under
``src/cubnm/data/maps/``. Continuous maps (``myelinmap``, ``fcgradient01``) are
fetched from neuromaps in fsLR space, parcellated, and saved as raw values
(``ctx_parc-{parc}_desc-{map}.txt``). The ``yeo7`` map is
derived from Schaefer parcel labels and only supports Schaefer parcellations. Prepare maps with:

```bash
uv run prep_maps.py
```
