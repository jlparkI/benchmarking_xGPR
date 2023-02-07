"""Prepares tabular datasets with existing deep learning performance benchmarks
in the literature for training with xGPR."""
import os
import calendar

import pandas as pd
import numpy as np


def prep_tabular_datasets(start_dir, dataset_dir, dataset_filename = "raw_data.csv"):
    """Preps the tabular datasets used for comparisons with various
    deep learning models."""
    os.chdir(start_dir)
    os.chdir(os.path.join("benchmark_evals", dataset_dir,
                "raw_data"))
    if dataset_filename not in os.listdir():
        raise ValueError(f"{dataset_dir} dataset not retrieved yet.")

    if "song" in dataset_dir:
        raw = pd.read_csv(dataset_filename, header=None, delimiter=",")
        ydata = raw.iloc[:,0].values
        xdata = raw.iloc[:,1:].values
        cutoff = int(0.8 * ydata.shape[0])
        trainy, trainx = ydata[:cutoff], xdata[:cutoff,:]
        testy, testx = ydata[cutoff:], xdata[cutoff:,:]
        trainx_mean, trainx_std = np.mean(trainx, axis=0), np.std(trainx, axis=0)
        trainx = (trainx - trainx_mean[None,:]) / trainx_std[None,:]
        testx = (testx - trainx_mean[None,:]) / trainx_std[None,:]

    elif "rossman" in dataset_dir:
        raw = pd.read_csv(dataset_filename)
        trainx, testx, trainy, testy = preprocess_rossman(raw)
    elif "sulfur" in dataset_dir:
        raw = pd.read_csv(dataset_filename)
        trainx, trainy = raw.iloc[:,0:5].values, raw.iloc[:,5].values

    rng = np.random.default_rng(123)
    idx = rng.permutation(trainy.shape[0])
    trainx, trainy = trainx[idx,:], trainy[idx]

    os.chdir("..")
    if "standard" not in os.listdir():
        os.mkdir("standard")

    os.chdir("standard")
    if "train" not in os.listdir():
        os.mkdir("train")
    if "test" not in os.listdir() and "sulfur" not in dataset_dir:
        os.mkdir("test")

    os.chdir("train")
    generate_block_files(trainx, trainy, blocksize = 2000)
    if "sulfur" in dataset_dir:
        return
    os.chdir(os.path.join("..", "test"))
    generate_block_files(testx, testy, blocksize = 2000)



def generate_block_files(xdata, ydata, blocksize):
    """Save 'chunks' of data to numpy files for later
    retrieval."""
    fnum = 0
    for i in range(0, xdata.shape[0], blocksize):
        np.save(f"{fnum}_xvalues.npy", xdata[i:i+blocksize,:])
        np.save(f"{fnum}_yvalues.npy", ydata[i:i+blocksize])
        fnum += 1
        print(f"Block {fnum} complete")

def get_str_column_names(df):
    str_names = []
    for col in df.columns:
        for xitem in df[col]:
            if isinstance(xitem, str):
                str_names.append(col)
                break

    return str_names


def fix_strs(df, cat_names):
    for col in cat_names:
        df = pd.get_dummies(df, prefix=[col], columns=[col])
    return df


def preprocess_rossman(df):
    """The rossman dataset requires some special preprocessing. We
    follow closely the preprocessing done for the Catboost paper
    at https://github.com/catboost, but with a few differences,
    because unlike gradient boosted trees, GPs are most definitely
    not scale-invariant, so the features have to be on the same
    scale, and redundant features are more problematic for GPs."""

    month_abbrs = calendar.month_abbr[1:]
    month_abbrs[8] = "Sept"
    #Unlike Catboost, we don't load the "store features" -- those are
    #redundant since that info is already captured in the store number.
    df['Year'] = [int(s.split("-")[0]) for s in df["Date"].tolist()]
    df["Month"] = [int(s.split("-")[1]) for s in df["Date"].tolist()]
    df = df.drop(['Date'], axis=1)
    #Open is redundant, since there are no customers when stores are not open.
    df = df.drop(["Open"], axis=1)
    df = df[df["Year"] >= 2014].copy()

    df = df.fillna(0)
    str_cat_columns = ["Store", "DayOfWeek", "StateHoliday",
            "SchoolHoliday", "Month"]
    df = fix_strs(df, str_cat_columns)

    train_inds = df[df['Year'] == 2014].index
    test_inds = df[df['Year'] == 2015].index

    train_df = df.iloc[train_inds].copy()
    test_df = df.iloc[test_inds].copy()
    del df

    train_df = train_df.drop(["Year"], axis=1)
    test_df = test_df.drop(["Year"], axis=1)
    trainy, testy = train_df["Sales"].values, test_df["Sales"].values
    train_df = train_df.drop(["Sales"], axis=1)
    test_df = test_df.drop(["Sales"], axis=1)

    trainx, testx = train_df.values.astype(np.float32), test_df.values.astype(np.float32)
    trainx[:,1:] *= trainx[:,0:1]
    testx[:,1:] *= testx[:,0:1]
    trainx, testx = trainx[:,1:], testx[:,1:]
    trainx_mean = np.mean(trainx, axis=0)
    trainx_std = np.std(trainx, axis=0)

    #To ensure all on same scale, multiply all one-hot
    #encoded features by num customers and standardize.
    trainx = (trainx - trainx_mean[None,:]) / \
                trainx_std[None,:]
    testx = (testx - trainx_mean[None,:]) / \
                trainx_std[None,:]
    print(f"Train dpoints: {trainx.shape[0]}")
    print(f"Test dpoints: {testx.shape[0]}")
    return trainx, testx, trainy, testy
