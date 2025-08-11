This project is a fork of [griffin-goodwin/Projection-Effects](https://github.com/griffin-goodwin/Projection-Effects).
The owner, Griffin Goodwin, gave me permission to reuse and modify his code in
an email on 2025-07-22.

The original code is in the `master` branch, which is protected as it isn't
meant to be altered. In order to run the code, I needed to modify it; my
modifications are in the `main` branch.

The dependencies for this repo are listed in `env.yml`. A `conda` environment
with those dependencies can be created by running the following command from the
top-level directory:
```
./make_env.sh
```

The code uses the SWAN-SF dataset, which is described in [this paper](https://doi.org/10.1038/s41597-020-0548-x).
Running the command below from `data/` will download the data into a
subdirectory of `data/`:
```
./download_data.sh
```
The code that was used to generate the dataset can be found [here](https://bitbucket.org/gsudmlab/workspace/projects/FP),
in several repositories it seems. In particular, code for computing `HC_ANGLE`
can be found in [this script](https://bitbucket.org/gsudmlab/armvtsprep/src/main/mvts/add_TMFI_column.py).
