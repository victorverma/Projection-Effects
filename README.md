This project is a fork of [griffin-goodwin/Projection-Effects](https://github.com/griffin-goodwin/Projection-Effects),
which contains the code for [this paper](https://dx.doi.org/10.3847/1538-4357/adb4f6).
The owner of that repository, Griffin Goodwin, gave me permission to reuse and
modify his code in an email on 2025-07-22.

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
Running the command below from `data/raw/` will download the data into
`data/raw/swan_sf/`:
```
./download_raw_data.sh
```
The code that was used to generate the dataset can be found [here](https://bitbucket.org/gsudmlab/workspace/projects/FP),
in several repositories it seems. In particular, code for computing `HC_ANGLE`
can be found in [this script](https://bitbucket.org/gsudmlab/armvtsprep/src/main/mvts/add_TMFI_column.py).

The SWAN-SF dataset is divided across many CSVs, each of which contains a
12-hour multivariate time series that preceded either a flare or a quiet period.
In the experiments in the paper by Goodwin et al., each row of the training and
test sets contained summary statistics computed from a single CSV. The
CSVs that were used are in the directories matching `data/raw/swan_sf/partition[1-5]/(FL|NF)/`.
Running the command below from `data/processed/` makes four data frames for each
partition. One contains the data in all the CSVs for the partition and another
contains all of the summary statistic data. The other two are similar, except
they incorporate corrections computed using the polynomials given in the paper.
The data frames are saved in files matching `data/processed/partition[1-5]/(corrected_)?(full|summary)_df.parquet`.
```
./make_dfs.sh
```
