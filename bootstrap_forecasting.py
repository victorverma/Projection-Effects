# Import necessary libraries
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, make_scorer
from gsulabpackage.util import metrics as util


# Define a function to calculate TSS and HSS2 metrics
def get_tss_and_hss2(y_true, y_pred, labels=None):
    if labels is None:
        labels = [0, 1]

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    # Calculate true positive rate and false positive rate
    tp_rate = tp / float(tp + fn) if tp > 0 else 0
    fp_rate = fp / float(fp + tn) if fp > 0 else 0

    # Calculate TSS
    tss = tp_rate - fp_rate

    # Calculate HSS2
    numer = 2 * ((tp * tn) - (fn * fp))
    denom = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss2 = numer / float(denom)

    return tss + hss2


# Define a function to train the model
def trainModel(training, params, corrBool):
    # Load training data
    flaring = pd.read_csv("partition" + training + "_corr_boot/PointInTime/flaring.csv", sep="\t")
    nonflaring = pd.read_csv("partition" + training + "_corr_boot/PointInTime/nonflaring.csv", sep="\t")
    trainingdata = pd.concat([flaring, nonflaring])

    # Label the data
    trainingdata.loc[(trainingdata["LABEL"] == "M") | (trainingdata["LABEL"] == "X"), "LABEL"] = 1
    trainingdata.loc[(trainingdata["LABEL"] != 1), "LABEL"] = 0
    trainingdata.dropna(inplace=True)

    # Create a scorer
    score = make_scorer(util.get_tss)

    # Assign AR group
    trainingdata['AR'] = 1
    for i in range(0, len(trainingdata)):
        trainingdata['AR'].iloc[i] = trainingdata['FILE'].iloc[i][
                                     trainingdata['FILE'].iloc[i].find("_ar"):trainingdata['FILE'].iloc[i].find("_s")]

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(trainingdata[params])
    scaled = pd.DataFrame(scaler.transform(trainingdata[params]))
    scaled.columns = trainingdata[params].columns

    # Select classifier based on training and correction flag
    if corrBool == False:
        if training == "1":
            clf = SVC(class_weight="balanced", kernel="rbf", C=10, gamma=0.0001)
        elif training == "2":
            clf = SVC(class_weight="balanced", kernel="rbf", C=0.01, gamma='scale')
        elif training == "3":
            clf = SVC(class_weight="balanced", kernel="rbf", C=100, gamma=0.0001)
        elif training == "4":
            clf = SVC(class_weight="balanced", kernel="rbf", C=0.001, gamma=0.01)
        else:
            clf = SVC(class_weight="balanced", kernel="rbf", C=0.001, gamma=0.01)
    else:
        if training == "1":
            clf = SVC(class_weight="balanced", kernel="rbf", C=0.01, gamma=0.01)
        elif training == "2":
            clf = SVC(class_weight="balanced", kernel="rbf", C=0.1, gamma=0.0001)
        elif training == "3":
            clf = SVC(class_weight="balanced", kernel="rbf", C=10, gamma=0.0001)
        elif training == "4":
            clf = SVC(class_weight="balanced", kernel="rbf", C=0.0001, gamma='scale')
        else:
            clf = SVC(class_weight="balanced", kernel="rbf", C=0.1, gamma=0.0001)

    # Train the classifier
    clf.fit(scaled[params].to_numpy(), trainingdata['LABEL'].astype('int'))
    return clf, scaler


# Define a function to forecast using the trained model
def forecast(training, testing, params, clf, scaler):
    print("Training Data: " + training)
    print("Testing Data: " + testing)

    # Load testing data
    flaring = pd.read_csv("partition" + testing + "_corr_boot/PointInTime/flaring.csv", sep="\t")
    nonflaring = pd.read_csv("partition" + testing + "_corr_boot/PointInTime/nonflaring.csv", sep="\t")
    testingdata = pd.concat([flaring, nonflaring])

    # Label the data
    testingdata.loc[(testingdata["LABEL"] == "M") | (testingdata["LABEL"] == "X"), "LABEL"] = 1
    testingdata.loc[(testingdata["LABEL"] != 1), "LABEL"] = 0
    testingdata.dropna(inplace=True)

    # Scale the testing data
    scaledtest = pd.DataFrame(scaler.transform(testingdata[params]))
    scaledtest.columns = testingdata[params].columns

    # Predict using the classifier
    pred = clf.predict(scaledtest[params].to_numpy())
    testingdata['PRED'] = pred

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(testingdata['LABEL'].astype('int'), pred, labels=[0, 1]).ravel()

    # Calculate HSS2
    hss2numer = 2 * ((tp * tn) - (fn * fp))
    hss2denom = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss2 = hss2numer / float(hss2denom)

    # Calculate true positive rate and false positive rate
    tp_rate = tp / float(tp + fn) if tp > 0 else 0
    fp_rate = fp / float(fp + tn) if fp > 0 else 0

    # Print metrics
    print(tp_rate - fp_rate)
    print(hss2)
    print("TP " + str(tp))
    print("TN " + str(tn))
    print("FP " + str(fp))
    print("FN " + str(fn))
    print("\n")

    return [tn, fp, fn, tp], training, testing, testingdata


# Define a function to get features based on importance
def getFeatures(partition, numFeatures, corrected):
    data = pd.read_csv("Partition" + partition + "Improvements.csv")
    if corrected:
        top = data[['Specs', 'CORR_Specs', 'IMP']].head(numFeatures)
        features = []
        for index, row in top.iterrows():
            if row['IMP'] >= 0:
                features.append(row['CORR_Specs'])
            else:
                features.append(row['Specs'])
    else:
        features = data['Specs'].head(numFeatures).to_numpy().tolist()
    return features


# Define a function to get final features
def getFeaturesFinal(corrected):
    data = pd.read_csv("PartitionAVGALLImprovements.csv")
    if corrected:
        features = data['CORR_Specs'].to_numpy().tolist()
    else:
        features = data['Specs'].to_numpy().tolist()
    return features


# Define a function to get features with only improvements
def getFeaturesOnlyImp(partition, numFeatures, corrected):
    data = pd.read_csv("Partition" + partition + "Improvements.csv")
    if corrected:
        top = data[['Specs', 'CORR_Specs', 'IMP']][data['IMP'] > 0].head(numFeatures)
        features = top['CORR_Specs'].to_numpy().tolist()
    else:
        top = data[['Specs', 'CORR_Specs', 'IMP']][data['IMP'] > 0].head(numFeatures)
        features = top['Specs'].to_numpy().tolist()
    return features


# Initialize results dictionary
results = {}

# Get features and train model for partition 1
feat = getFeaturesOnlyImp("1", 25, False)
clf, scalar = trainModel("1", feat, False)

# Forecast and save results for different partitions
conf, training, testing, testingdata = forecast("1", "2", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition12TestingData.csv")

conf, training, testing, testingdata = forecast("1", "3", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition13TestingData.csv")

conf, training, testing, testingdata = forecast("1", "4", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition14TestingData.csv")

conf, training, testing, testingdata = forecast("1", "5", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition15TestingData.csv")

# Repeat the process for other partitions
feat = getFeaturesOnlyImp("2", 25, False)
clf, scalar = trainModel("2", feat, False)
conf, training, testing, testingdata = forecast("2", "1", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition21TestingData.csv")

conf, training, testing, testingdata = forecast("2", "3", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition23TestingData.csv")

conf, training, testing, testingdata = forecast("2", "4", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition24TestingData.csv")

conf, training, testing, testingdata = forecast("2", "5", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition25TestingData.csv")

feat = getFeaturesOnlyImp("3", 25, False)
clf, scalar = trainModel("3", feat, False)
conf, training, testing, testingdata = forecast("3", "1", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition31TestingData.csv")

conf, training, testing, testingdata = forecast("3", "2", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition32TestingData.csv")

conf, training, testing, testingdata = forecast("3", "4", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition34TestingData.csv")

conf, training, testing, testingdata = forecast("3", "5", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition35TestingData.csv")

feat = getFeaturesOnlyImp("4", 25, False)
clf, scalar = trainModel("4", feat, False)
conf, training, testing, testingdata = forecast("4", "1", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition41TestingData.csv")

conf, training, testing, testingdata = forecast("4", "2", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition42TestingData.csv")

conf, training, testing, testingdata = forecast("4", "3", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition43TestingData.csv")

conf, training, testing, testingdata = forecast("4", "5", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition45TestingData.csv")

feat = getFeaturesOnlyImp("5", 25, False)
clf, scalar = trainModel("5", feat, False)
conf, training, testing, testingdata = forecast("5", "1", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition51TestingData.csv")

conf, training, testing, testingdata = forecast("5", "2", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition52TestingData.csv")

conf, training, testing, testingdata = forecast("5", "3", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition53TestingData.csv")

conf, training, testing, testingdata = forecast("5", "4", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition54TestingData.csv")

# Repeat the process for corrected data
results = {}

feat = getFeaturesOnlyImp("1", 25, True)
clf, scalar = trainModel("1", feat, True)
conf, training, testing, testingdata = forecast("1", "2", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition12TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("1", "3", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition13TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("1", "4", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition14TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("1", "5", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition15TestingDataCorr.csv")

feat = getFeaturesOnlyImp("2", 25, True)
clf, scalar = trainModel("2", feat, True)
conf, training, testing, testingdata = forecast("2", "1", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition21TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("2", "3", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition23TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("2", "4", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition24TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("2", "5", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition25TestingDataCorr.csv")

feat = getFeaturesOnlyImp("3", 25, True)
clf, scalar = trainModel("3", feat, True)
conf, training, testing, testingdata = forecast("3", "1", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition31TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("3", "2", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition32TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("3", "4", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition34TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("3", "5", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition35TestingDataCorr.csv")

feat = getFeaturesOnlyImp("4", 25, True)
clf, scalar = trainModel("4", feat, True)
conf, training, testing, testingdata = forecast("4", "1", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition41TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("4", "2", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition42TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("4", "3", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition43TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("4", "5", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition45TestingDataCorr.csv")

feat = getFeaturesOnlyImp("5", 25, True)
clf, scalar = trainModel("5", feat, True)
conf, training, testing, testingdata = forecast("5", "1", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition51TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("5", "2", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition52TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("5", "3", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition53TestingDataCorr.csv")

conf, training, testing, testingdata = forecast("5", "4", feat, clf, scalar)
results[training + ":" + testing] = conf
testingdata.to_csv("Partition54TestingDataCorr.csv")