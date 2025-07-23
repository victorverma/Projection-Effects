# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats


# %%
def fun(x, t, y):
    """
    Polynomial function to fit data.

    Parameters:
    x (list): Coefficients of the polynomial.
    t (float): Independent variable.
    y (float): Dependent variable.

    Returns:
    float: Result of the polynomial function minus y.
    """
    return 1 + x[0] * t + x[1] * t ** 2 + x[2] * t ** 3 + x[3] * t ** 4 + x[4] * t ** 5 + x[5] * t ** 6 + x[
        6] * t ** 7 - y


def generate_data(t, b, c, d, e, f, g, h):
    """
    Generate data using a polynomial function.

    Parameters:
    t (float): Independent variable.
    b, c, d, e, f, g, h (float): Coefficients of the polynomial.

    Returns:
    float: Result of the polynomial function.
    """
    y = 1 + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5 + g * t ** 6 + h * t ** 7
    return y


# %%
# Load the dictionary of fits from a pickle file
with open('dictionary_fits/subsampledictionary.pkl', 'rb') as f:
    data = pickle.load(f)

# Select ARs with RMS less than 0.3
arsToSelect = data[data['RMS'] < .3]['AR'].to_numpy()
print(len(arsToSelect))

# Load and filter the data to correct
datToCorrect = pd.read_csv("ARData.csv")
datToCorrect = datToCorrect[datToCorrect['HC_ANGLE'] <= 73]

# %%
# List of parameters to analyze
params = ["ABSNJZH", "EPSX", "EPSY", "EPSZ", "MEANALP", "MEANGAM", "MEANGBH", "MEANGBT", "MEANGBZ", "MEANJZD",
          "MEANJZH", "MEANPOT", "MEANSHR", "R_VALUE", "SAVNCPP", "SHRGT45", "TOTBSQ", "TOTFX", "TOTFY", "TOTFZ",
          "TOTPOT", "TOTUSJH", "TOTUSJZ", "USFLUX"]

# %%
# Set the plot style
plt.style.use('dark_background')

# %%
# Create subplots for each parameter
im, ax = plt.subplots(6, 4, figsize=(16, 14), sharex=True)
im.patch.set_alpha(0.75)

for i, param in enumerate(params):
    # Filter and process data for the current parameter
    dat = datToCorrect.dropna(subset=[param])
    dat[param] = dat[param].abs()
    dat = dat[datToCorrect['AR'].isin(arsToSelect)]

    # Load results from pickle files
    with open(f'dictionary_fits/saved_dictionary_{param}.pkl', 'rb') as f:
        results = pickle.load(f)
    with open(f'dictionary_fits/saved_dictionary_prefit_{param}.pkl', 'rb') as f:
        pre_results = pickle.load(f)

    for ar in dat['AR'].unique():
        d = dat[dat['AR'] == ar]
        if d['HC_ANGLE'].min() <= 10:
            out = d[d["HC_ANGLE"] < d["HC_ANGLE"].min() + 3]
            removedOut = out[np.abs(stats.zscore(out[param])) < 3]
            dat.loc[dat['AR'] == ar, param] = d[param] / (removedOut[param].mean())
        if d['HC_ANGLE'].min() > 10:
            out = d[d["HC_ANGLE"] < d["HC_ANGLE"].min() + 3]
            removedOut = out[np.abs(stats.zscore(out[param])) < 3]
            dat.loc[dat['AR'] == ar, param] = (d[param] / (removedOut[param].mean())) * np.mean(
                generate_data(np.arange(d["HC_ANGLE"].min(), d["HC_ANGLE"].min() + 3, .1), *pre_results['Mean Fit']))

    # Replace infinite values and drop NaNs
    dat.replace(np.inf, np.nan, inplace=True)
    dat.dropna(subset=[param], inplace=True)

    # Plot the data
    ax[i // 4, i % 4].set_facecolor((0, 0, 0, 0.75))
    ax[i // 4, i % 4].axhline(1, c="#DCD427", alpha=.75, linestyle="dashdot", linewidth=2)
    ax[i // 4, i % 4].scatter(x=dat['HC_ANGLE'], y=dat[param], s=.1, c='white', alpha=.2)
    ax[i // 4, i % 4].plot(np.arange(0, 73, .1), generate_data(np.arange(0, 73, .1), *results['Mean Fit']), c="#FF3333",
                           alpha=1, linewidth=3)
    ax[i // 4, i % 4].plot(results['Above Curve Data Median']['HC_ANGLE'], results['Above Curve Data Median'][param],
                           c="#FF3333", alpha=.75, linestyle=(0, (1, 1)), linewidth=2)
    ax[i // 4, i % 4].plot(results['Below Curve Data Median']['HC_ANGLE'], results['Below Curve Data Median'][param],
                           c="#FF3333", alpha=.75, linestyle=(0, (1, 1)), linewidth=2)
    ax[i // 4, i % 4].set_title(param, {'fontsize': 20})
    ax[i // 4, i % 4].tick_params(axis='both', which='major', labelsize=15)
    ax[i // 4, i % 4].grid(False)
    mx = np.nanmax(results['Above Curve Data Median'][param])
    mn = np.nanmin(results['Below Curve Data Median'][param])
    rnge = mx - mn
    ax[i // 4, i % 4].set_ylim([0, mx + rnge * .25])

# Add common labels
ax = im.add_subplot(111, frameon=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"$\theta$ (degrees)", labelpad=25, fontsize=20)
ax.set_ylabel("Normalized Value", labelpad=33, fontsize=20)
plt.tight_layout()
plt.savefig("ProjectionEffectEachParam3.png", dpi=700)

# %%
# Create subplots for corrected data
im, ax = plt.subplots(6, 4, figsize=(10, 10), sharex=True, sharey=True)
for i, param in enumerate(params):
    dat = datToCorrect.dropna(subset=[param])
    dat[param] = dat[param].abs()
    dat = dat[datToCorrect['AR'].isin(arsToSelect)]

    # Load corrected results from pickle files
    with open(f'dictionary_fits/saved_dictionary_corrected_{param}.pkl', 'rb') as f:
        results = pickle.load(f)

    # Plot the corrected data
    ax[i // 4, i % 4].axhline(1, c="black", alpha=1, linestyle="dashdot")
    ax[i // 4, i % 4].scatter(x=results['HC Data'], y=results['Param Data'], s=.1, c='lightgrey', alpha=.25)
    for cur in results['Curves']:
        ax[i // 4, i % 4].plot(np.arange(0, 73, .1), generate_data(np.arange(0, 73, .1), *cur), c="coral", alpha=.1)
    ax[i // 4, i % 4].plot(np.arange(0, 73, .1), generate_data(np.arange(0, 73, .1), *results['Mean Fit']), c="maroon",
                           alpha=1)
    ax[i // 4, i % 4].plot(results['Above Curve Data Median']['HC_ANGLE'], results['Above Curve Data Median'][param],
                           c="maroon", alpha=.5, linestyle=(0, (1, 1)))
    ax[i // 4, i % 4].plot(results['Below Curve Data Median']['HC_ANGLE'], results['Below Curve Data Median'][param],
                           c="maroon", alpha=.5, linestyle=(0, (1, 1)))
    ax[i // 4, i % 4].set_title(param)
    ax[i // 4, i % 4].grid()
    ax[i // 4, i % 4].set_ylim([0, 3])

# Add common labels
ax = im.add_subplot(111, frameon=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Heliocentric Angle (deg)", labelpad=20, fontsize=12)
ax.set_ylabel("Normalized Value", labelpad=20, fontsize=12)
plt.tight_layout()
plt.savefig("ProjectionEffectEachParamCorrected.png", dpi=500)

# %%
# Reload the data
datToCorrect = pd.read_csv("ARData.csv", low_memory=False)

# %%
# Print the number of unique AR groups
print(len(datToCorrect.groupby('AR')))


# %%
def refinedData(dat, min_lon, max_lon):
    """
    Refine data based on longitude range.

    Parameters:
    dat (DataFrame): Data to refine.
    min_lon (float): Minimum longitude.
    max_lon (float): Maximum longitude.

    Returns:
    DataFrame: Refined data.
    """
    minimum = dat.groupby(['AR'])['LON_AVG'].min()
    maximum = dat.groupby(['AR'])['LON_AVG'].max()
    lon = minimum.to_frame().merge(maximum.to_frame(), left_index=True, right_index=True)
    lon['AR'] = lon.index
    goodARs = []
    for ar in lon['AR'].unique():
        if (lon[lon['AR'] == ar]['LON_AVG_x'][0] <= min_lon) and (lon[lon['AR'] == ar]['LON_AVG_y'][0] >= max_lon):
            goodARs.append(ar)
    print(len(goodARs))
    return dat[dat['AR'].isin(goodARs)]


# %%
# Refine the data based on longitude range
refined = refinedData(datToCorrect, -73, 73)

# %%
# Adjust HC_ANGLE for refined data
minHC = refined.groupby(["AR"])['HC_ANGLE'].idxmin()
for ind in minHC.index:
    refined.loc[(refined.index <= minHC.loc[ind]) & (refined['AR'] == ind), 'HC_ANGLE'] = refined.loc[(refined.index <=
                                                                                                       minHC.loc[
                                                                                                           ind]) & (
                                                                                                                  refined[
                                                                                                                      'AR'] == ind), 'HC_ANGLE'] * -1

# %%
# Filter ARs based on HC_ANGLE ranges
arr = refined[((refined['HC_ANGLE'] <= 30) & (refined['HC_ANGLE'] >= 0))]['AR'].unique()
arr2 = refined[((refined['HC_ANGLE'] >= -30) & (refined['HC_ANGLE'] <= 0))]['AR'].unique()
arr3 = refined[((refined['HC_ANGLE'] <= 100) & (refined['HC_ANGLE'] >= 73))]['AR'].unique()
arr4 = refined[((refined['HC_ANGLE'] >= -100) & (refined['HC_ANGLE'] <= -73))]['AR'].unique()
inter = np.intersect1d(arr, arr2)
inter2 = np.intersect1d(inter, arr3)
inter3 = np.intersect1d(inter2, arr4)
print(len(inter3))
refined = refined[refined['AR'].isin(inter3)]
refined = refined[(refined['HC_ANGLE'] <= 73) & (refined['HC_ANGLE'] >= -73)]