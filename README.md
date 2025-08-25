## Overview ##

This project is a fork of [griffin-goodwin/Projection-Effects](https://github.com/griffin-goodwin/Projection-Effects),
which contains the code for [this paper](https://dx.doi.org/10.3847/1538-4357/adb4f6).

The original code is in the `master` branch, which is protected as it isn't
meant to be altered. In order to run the code, I needed to modify it; my
modifications are in the `main` branch.

## Setting Up the Conda Environment ##

The dependencies for this repo are listed in `env.yml`. A `conda` environment
with those dependencies can be created by running the following command from the
top-level directory:
```
./make_env.sh
```

## Downloading the Data ##

The code uses the SWAN-SF dataset, which is described in [this paper](https://doi.org/10.1038/s41597-020-0548-x).
Running the command below from `data/raw/` will download the data into
`data/raw/swan_sf/`:
```
./download_raw_data.sh
```
The code that was used to generate the dataset can be found [here](https://bitbucket.org/gsudmlab/workspace/projects/FP),
in several repositories it seems. In particular, code for computing `HC_ANGLE`
can be found in [this script](https://bitbucket.org/gsudmlab/armvtsprep/src/main/mvts/add_TMFI_column.py).

## Putting the Data in Data Frames ##

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

## Running the Experiment ##

Goodwin et al. performed an experiment in which a support vector classifier was
trained on each partition and tested on the others. This was done using both the
original data for the partitions and the corrected data for them. The code in
`experiment/` performs this experiment.

### Feature Selection ###

For a given partition, the features used in the support vector
classifiers were the 25 features with the biggest Fisher score improvements for
that partition. To compute the improvements and rank all the features, run the
code in the notebook `experiment/feature_selection/feature_rankings.ipynb`.

### Training the Models

To train the models using the top features, run this command from `experiment/model_training/`:
```
./train_models.sh
```
This can take a long time to complete. To run the code on an HPC using Slurm, a
job like `train_models.sbat` can be submitted. Make sure to export the
environment variable `SBATCH_ACCOUNT` first.

### Testing the Models ###

To test the trained models, run this command from `experiment/model_testing/`:
```
./test_models.sh
```
To run this on an HPC, modify and submit the Slurm job `test_models.sbat`.

### Plotting the Results ###

The code in `experiment/model_testing/all_results_plots.ipynb` generates plots
for various performance metrics. In each plot, metric values are plotted for
each pair of a training partition and a testing partition, with one curve being
for the original data and the other being for the corrected data.
