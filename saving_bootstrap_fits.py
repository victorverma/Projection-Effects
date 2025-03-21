# Import necessary libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from scipy.optimize import least_squares, curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import interpolate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import dill
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pickle
import sys

# List of parameters to analyze
params = ["EPSX"]

# Loop through each parameter
for param in params:
    # Load the dictionary of fits from a pickle file
    with open('dictionary_fits/subsampledictionary.pkl', 'rb') as f:
        results = pickle.load(f)

    # Define a polynomial function to fit data
    def fun(x, t, y):
        return 1 + x[0]*t + x[1]*t**2 + x[2]*t**3 + x[3]*t**4 + x[4]*t**5 + x[5]*t**6 + x[6]*t**7 - y

    # Generate data using a polynomial function
    def generate_data(t, b, c, d, e, f, g, h):
        y = 1 + b*t + c*t**2 + d*t**3 + e*t**4 + f*t**5 + g*t**6 + h*t**7
        return y

    # Create bootstrap samples for fitting
    def create_bootstrap_samples(df, n_samples=1000):
        sample_fits = []
        x0 = [-9.46218617e-02, 8.66111744e-03, -3.38727155e-04, 7.14326020e-06, -8.33216960e-08, 4.94618271e-10, -1.15452287e-12]
        for i in range(n_samples):
            df_bootstrap_sample = df.groupby("AR").sample(n=len(df['AR'].unique()), replace=True)
            sample_fits.append(least_squares(fun, x0, loss='linear', args=(df_bootstrap_sample['HC_ANGLE'], df_bootstrap_sample[param])).x)
        fits = np.asarray(sample_fits)
        mean = fits.mean(axis=0)
        std = fits.std(axis=0)
        return fits, mean, std

    # Select ARs with RMS less than 0.3
    arsToSelect = results[results['RMS'] < .3]['AR'].to_numpy()
    datToCorrect = pd.read_csv("ARData.csv", low_memory=False)
    datToCorrect = datToCorrect[datToCorrect['HC_ANGLE'] <= 73]
    datToCorrect.dropna(subset=[param], inplace=True)
    datToCorrect[param] = datToCorrect[param].abs()
    datToCorrect = datToCorrect[datToCorrect['AR'].isin(arsToSelect)]

    # Special handling for "R_VALUE" parameter
    if param == "R_VALUE":
        datToCorrect.replace([0], np.nan, inplace=True)
        datToCorrect.dropna(subset=['R_VALUE'], inplace=True)

    prefitARs = []
    for ar in datToCorrect['AR'].unique():
        d = datToCorrect[datToCorrect['AR'] == ar]
        if d['HC_ANGLE'].min() <= 10:
            prefitARs.append(ar)
            out = d[d["HC_ANGLE"] < d["HC_ANGLE"].min() + 3]
            removedOut = out[np.abs(stats.zscore(out[param])) < 3]
            datToCorrect.loc[datToCorrect['AR'] == ar, param] = d[param] / (removedOut[param].mean())

    prefitData = datToCorrect[datToCorrect['AR'].isin(prefitARs)]

    # Create bootstrap samples for prefit data
    curves, mean, std = create_bootstrap_samples(prefitData, n_samples=500)

    datToCorrectLinear = datToCorrect.copy()

    for ar in datToCorrect['AR'].unique():
        d = datToCorrect[datToCorrect['AR'] == ar]
        if d['HC_ANGLE'].min() > 10:
            out = d[d["HC_ANGLE"] < d["HC_ANGLE"].min() + 3]
            removedOut = out[np.abs(stats.zscore(out[param])) < 3]
            datToCorrectLinear.loc[datToCorrectLinear['AR'] == ar, param] = (d[param] / (removedOut[param].mean())) * np.mean(generate_data(np.arange(d["HC_ANGLE"].min(), d["HC_ANGLE"].min() + 3, .1), *mean))

    # Create bootstrap samples for corrected data
    curves, mean, std = create_bootstrap_samples(datToCorrectLinear, n_samples=500)

    # Generate fit data
    datToCorrectLinear['Fit'] = generate_data(datToCorrectLinear['HC_ANGLE'], *mean)

    stdfits = pd.DataFrame(columns=['HC_ANGLE', param])

    for cur in curves:
        temp = pd.DataFrame({'HC_ANGLE': np.arange(0, 73.5, .1), param: generate_data(np.arange(0, 73.5, .1), *cur)})
        stdfits = pd.concat([stdfits, temp])
    stdfits['Fit'] = generate_data(stdfits['HC_ANGLE'], *mean)
    aboveCurvefits = stdfits[stdfits[param] > stdfits['Fit']]
    aboveCurvefitsfinal = aboveCurvefits[['HC_ANGLE', param]].groupby(pd.cut(aboveCurvefits['HC_ANGLE'], np.arange(0, 73.5, .5), include_lowest=True)).median()
    belowCurvefits = stdfits[stdfits[param] < stdfits['Fit']]
    belowCurvefitsfinal = belowCurvefits[['HC_ANGLE', param]].groupby(pd.cut(belowCurvefits['HC_ANGLE'], np.arange(0, 73.5, .5), include_lowest=True)).median()
    aboveCurve = datToCorrectLinear[datToCorrectLinear[param] > datToCorrectLinear['Fit']]
    aboveCurvefinal = aboveCurve[['HC_ANGLE', param]].groupby(pd.cut(aboveCurve['HC_ANGLE'], np.arange(0, 73.5, .5), include_lowest=True)).median()
    belowCurve = datToCorrectLinear[datToCorrectLinear[param] < datToCorrectLinear['Fit']]
    belowCurvefinal = belowCurve[['HC_ANGLE', param]].groupby(pd.cut(belowCurve['HC_ANGLE'], np.arange(0, 73.5, .5), include_lowest=True)).median()

    # Plot the data and fits
    plt.scatter(datToCorrectLinear['HC_ANGLE'], datToCorrectLinear[param], s=.5, c='lightgrey', alpha=.5)
    for cur in curves:
        plt.plot(np.arange(0, 73, .1), generate_data(np.arange(0, 73, .1), *cur), c="coral", alpha=.25)
    plt.plot(np.arange(0, 73, .1), generate_data(np.arange(0, 73, .1), *mean), c="maroon", alpha=1)
    plt.plot(aboveCurvefinal['HC_ANGLE'], aboveCurvefinal[param], c="coral", linestyle="--")
    plt.plot(belowCurvefinal['HC_ANGLE'], belowCurvefinal[param], c="coral", linestyle="--")

    plt.ylim([0, 5])
    plt.xlim([-2.5, 75])
    plt.xlabel("Heliocentric Angle (degrees)")
    plt.ylabel("Normalized " + param)
    plt.tight_layout()
    plt.close()

    # Save the outcome to a pickle file
    outcome = {'Mean Fit': mean, 'Curves': curves, 'Above Curve Data Median': aboveCurvefinal, 'Below Curve Data Median': belowCurvefinal, 'Above Curve Fits Median': aboveCurvefitsfinal, 'Below Curve Median': belowCurvefitsfinal}
    # with open('dictionary_fits/saved_dictionary_' + param + '.pkl', 'wb') as f:
    #     pickle.dump(outcome, f)

# Plot the final data and fits
plt.scatter(datToCorrectLinear['HC_ANGLE'], datToCorrectLinear[param], s=.5, c='lightgrey', alpha=.5)
for cur in curves:
    plt.plot(np.arange(0, 73, .1), generate_data(np.arange(0, 73, .1), *cur), c="coral", alpha=.25)
plt.plot(np.arange(0, 73, .1), generate_data(np.arange(0, 73, .1), *mean), c="maroon", alpha=1)
plt.plot(aboveCurvefinal['HC_ANGLE'], aboveCurvefinal[param], c="maroon", linestyle="--")
plt.plot(belowCurvefinal['HC_ANGLE'], belowCurvefinal[param], c="maroon", linestyle="--")
plt.ylim([0, 5])
plt.xlim([-2.5, 75])
plt.xlabel("Heliocentric Angle (degrees)")
plt.ylabel("Normalized " + param)
plt.tight_layout()