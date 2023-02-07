"""Provides the prep_UCI_dataset function for preparing three UCI datasets
for training for comparison with GPyTorch."""
import os
import pandas as pd
import numpy as np

def prep_UCI_dataset(start_dir, dataset_dir, split = None):
    """Preps the UCI datasets used for comparisons with GPyTorch."""
    os.chdir(start_dir)
    os.chdir(os.path.join("benchmark_evals", dataset_dir,
                "raw_data"))
    if "raw_data.csv" not in os.listdir() and "raw_data_.csv" not in os.listdir():
        raise ValueError(f"{dataset_dir} dataset not retrieved yet.")

    if "raw_data.csv" in os.listdir():
        raw_datafile = "raw_data.csv"
    else:
        raw_datafile = "raw_data_.csv"

    raw = pd.read_csv(raw_datafile, header=None, delimiter=",")
    ydata = raw.iloc[:,0].values
    xdata = raw.iloc[:,1:].values
    xdata = (xdata - np.mean(xdata, axis=0)[None,:]) / np.std(xdata, axis=0)[None,:]
    ydata = (ydata - np.mean(ydata)) / np.std(ydata)
    if "song" not in dataset_dir:
        rng = np.random.default_rng(123)
        idx = rng.choice(xdata.shape[0], xdata.shape[0], replace=False)
        ydata, xdata = ydata[idx], xdata[idx,:]
    cutoff = int(0.8 * ydata.shape[0])
    trainy, trainx = ydata[:cutoff], xdata[:cutoff,:]
    testy, testx = ydata[cutoff:], xdata[cutoff:,:]

    os.chdir("..")
    if split is not None:
        if split not in os.listdir():
            os.mkdir(split)
        os.chdir(split)
    if "train" not in os.listdir():
        os.mkdir("train")
    if "test" not in os.listdir():
        os.mkdir("test")
    os.chdir("train")
    generate_block_files(trainx, trainy, blocksize = 2000)
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
