#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# Function to calculate TSS and HSS metrics and group data by bins
def gridGraph(file, bins):
    data = pd.read_csv(file)
    # Calculate True Positives, True Negatives, False Positives, and False Negatives
    data['TP'] = data.apply(lambda row: 1 if row['LABEL'] == 1 and row['PRED'] == 1 else 0, axis=1)
    data['TN'] = data.apply(lambda row: 1 if row['LABEL'] == 0 and row['PRED'] == 0 else 0, axis=1)
    data['FP'] = data.apply(lambda row: 1 if row['LABEL'] == 0 and row['PRED'] == 1 else 0, axis=1)
    data['FN'] = data.apply(lambda row: 1 if row['LABEL'] == 1 and row['PRED'] == 0 else 0, axis=1)
    res = data.groupby(pd.cut(data['HC_ANGLE_mean'], np.arange(0, 80, bins))).sum()
    tp = res['TP']
    tn = res['TN']
    fp = res['FP']
    fn = res['FN']

    # Calculate TSS and HSS metrics
    tp_rate = tp / (tp + fn)
    fp_rate = fp / (fp + tn)
    hss2numer = 2 * ((tp * tn) - (fn * fp))
    hss2denom = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss2 = hss2numer / (hss2denom)

    res['TSS'] = tp_rate - fp_rate
    res['HSS'] = hss2
    return res

# Function to calculate total counts and group data by bins
def gridGraph2(file, bins):
    data = pd.read_csv(file)
    # Calculate True Positives, True Negatives, False Positives, and False Negatives
    data['TP'] = data.apply(lambda row: 1 if row['LABEL'] == 1 and row['PRED'] == 1 else 0, axis=1)
    data['TN'] = data.apply(lambda row: 1 if row['LABEL'] == 0 and row['PRED'] == 0 else 0, axis=1)
    data['FP'] = data.apply(lambda row: 1 if row['LABEL'] == 0 and row['PRED'] == 1 else 0, axis=1)
    data['FN'] = data.apply(lambda row: 1 if row['LABEL'] == 1 and row['PRED'] == 0 else 0, axis=1)
    res = data.groupby(pd.cut(data['HC_ANGLE_mean'], np.arange(0, 80, bins))).sum()
    tp = res['TP']
    tn = res['TN']
    fp = res['FP']
    fn = res['FN']

    # Calculate total counts
    total = tp + tn + fp + fn
    res['total'] = total
    return res

# List of file names for uncorrected data
part = [
    'Partition12TestingData.csv', 'Partition13TestingData.csv', 'Partition14TestingData.csv',
    'Partition15TestingData.csv', 'Partition21TestingData.csv', 'Partition23TestingData.csv',
    'Partition24TestingData.csv', 'Partition25TestingData.csv', 'Partition31TestingData.csv',
    'Partition32TestingData.csv', 'Partition34TestingData.csv', 'Partition35TestingData.csv',
    'Partition41TestingData.csv', 'Partition42TestingData.csv', 'Partition43TestingData.csv',
    'Partition45TestingData.csv', 'Partition51TestingData.csv', 'Partition52TestingData.csv',
    'Partition53TestingData.csv', 'Partition54TestingData.csv'
]

# List of file names for corrected data
partCorr = [
    'Partition12TestingDataCorr.csv', 'Partition13TestingDataCorr.csv', 'Partition14TestingDataCorr.csv',
    'Partition15TestingDataCorr.csv', 'Partition21TestingDataCorr.csv', 'Partition23TestingDataCorr.csv',
    'Partition24TestingDataCorr.csv', 'Partition25TestingDataCorr.csv', 'Partition31TestingDataCorr.csv',
    'Partition32TestingDataCorr.csv', 'Partition34TestingDataCorr.csv', 'Partition35TestingDataCorr.csv',
    'Partition41TestingDataCorr.csv', 'Partition42TestingDataCorr.csv', 'Partition43TestingDataCorr.csv',
    'Partition45TestingDataCorr.csv', 'Partition51TestingDataCorr.csv', 'Partition52TestingDataCorr.csv',
    'Partition53TestingDataCorr.csv', 'Partition54TestingDataCorr.csv'
]

# Calculate TSS and HSS for uncorrected data
all_res = pd.DataFrame()
for p in part:
    print(p)
    res = gridGraph(p, 10)
    res['file'] = p
    all_res = pd.concat([all_res, res[['TSS', 'HSS']]], axis=1)

# Aggregate total counts for uncorrected data
all_res = pd.DataFrame()
for p in part:
    print(p)
    res = gridGraph2(p, 10)
    res['file'] = p
    all_res = pd.concat([all_res, res['total']], axis=1)

# Print aggregated results
print(all_res)

# Calculate TSS and HSS for corrected data
all_resCorr = pd.DataFrame()
for p in partCorr:
    print(p)
    res = gridGraph(p, 5)
    res['file'] = p
    all_resCorr = pd.concat([all_resCorr, res[['TSS', 'HSS']]], axis=1)

# Load pre-saved results
all_res = pd.read_csv('all_res2.csv')
all_resCorr = pd.read_csv('all_resCorr2.csv')
all_res.index = all_res['HC_ANGLE_mean']
all_resCorr.index = all_resCorr['HC_ANGLE_mean']

# Prepare data for plotting
tss_scores = all_res.filter(like='TSS').T.reset_index(drop=True)
hss_scores = all_res.filter(like='HSS').T.reset_index(drop=True)
tss_scoresCorr = all_resCorr.filter(like='TSS').T.reset_index(drop=True)
hss_scoresCorr = all_resCorr.filter(like='HSS').T.reset_index(drop=True)

# Combine TSS and HSS scores for uncorrected and corrected data
df = pd.concat([tss_scores, tss_scoresCorr], keys=["df1", "df2"]).reset_index()
df.drop('level_1', axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df.index = df['level_0']
df.drop('level_0', axis=1, inplace=True)

# Stack and rename columns for TSS and HSS scores
tssnew = tss_scores.stack().droplevel(0).reset_index().rename(columns={'index': 'field', 0: 'data'})
tssnew2 = tss_scoresCorr.stack().droplevel(0).reset_index().rename(columns={'index': 'field', 0: 'data'})
hssnew = hss_scores.stack().droplevel(0).reset_index().rename(columns={'index': 'field', 0: 'data'})
hssnew2 = hss_scoresCorr.stack().droplevel(0).reset_index().rename(columns={'index': 'field', 0: 'data'})

# Label the data as corrected or uncorrected
tssnew['LABEL'] = "UNCORR"
tssnew2['LABEL'] = "CORR"
hssnew['LABEL'] = "UNCORR"
hssnew2['LABEL'] = "CORR"

# Combine TSS and HSS data
tss = pd.concat([tssnew, tssnew2])
hss = pd.concat([hssnew, hssnew2])

# Plot TSS and HSS scores
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Plot box plots for TSS scores
sns.boxplot(data=tss, x='HC_ANGLE_mean', y='data', hue='LABEL', palette={"UNCORR": "#BDD358", "CORR": "#F07167"}, linecolor='black', ax=axs[0], linewidth=2, showmeans=True, meanprops={'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '7'}, showfliers=False, fill=True, gap=.1)
axs[0].set_ylabel('TSS')
axs[0].grid(axis='y')
axs[0].yaxis.set_minor_locator(AutoMinorLocator())
axs[0].tick_params(which='minor', length=2)
axs[0].grid(which='minor', linestyle='--', linewidth='0.5', color='grey', alpha=0.5)

# Plot box plots for HSS scores
sns.boxplot(data=hss, x='HC_ANGLE_mean', y='data', hue='LABEL', palette={"UNCORR": "#BDD358", "CORR": "#F07167"}, linecolor='black', ax=axs[1], linewidth=2, showmeans=True, meanprops={'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '7'}, showfliers=False, fill=True, gap=.1, legend=False)
axs[1].set_ylabel(r'HSS$_2$')
axs[1].grid(axis='y')
axs[1].set_xlabel(r'$\theta$ (degrees)')
axs[1].yaxis.set_minor_locator(AutoMinorLocator())
axs[1].tick_params(which='minor', length=2)
axs[1].grid(which='minor', linestyle='--', linewidth='0.5', color='grey', alpha=0.5)

# Set axis below grid
axs[0].set_axisbelow(True)
axs[1].set_axisbelow(True)

# Add legend
handles = [plt.Rectangle((0, 0), 1, 1, color='#BDD358', lw=2, ec='black'), plt.Rectangle((0, 0), 1, 1, color='#F07167', lw=2, ec='black')]
axs[0].legend(handles, ['Uncorrected', 'Corrected'])

plt.tight_layout()
plt.savefig('tss_hss_box_plots.png', dpi=600)
plt.show()