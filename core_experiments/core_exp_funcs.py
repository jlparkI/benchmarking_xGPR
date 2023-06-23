"""This script contains functions used by multiple experiments /
routines. This is part of the benchmarking suite used to test
the xGPR library for Parkinson et al."""
import os
import time
import numpy as np
from scipy.stats import spearmanr

from xGPR.xGP_Regression import xGPRegression
from xGPR.data_handling.dataset_builder import build_offline_fixed_vector_dataset
from xGPR.data_handling.dataset_builder import build_offline_sequence_dataset




def get_target_qm9_files(target_dir, data_type = "u298_atom"):
    """Gets a list of files in a target directory for QM9."""
    os.chdir(target_dir)
    xlist = [os.path.abspath(f) for f in os.listdir() if f.endswith("xfolded.npy")]
    ylist = [os.path.abspath(f) for f in os.listdir() if f.endswith(f"{data_type}_yvalues.npy")]
    xlist.sort()
    ylist.sort()
    return xlist, ylist



def get_qm9_train_dataset(start_dir, target_dir):
    """Gets the train dataset for QM9."""
    os.chdir(target_dir)
    train_xfiles, train_yfiles = get_target_qm9_files("train")
    os.chdir("..")
    valid_xfiles, valid_yfiles = get_target_qm9_files("valid")
    os.chdir(start_dir)
    train_dset = build_dataset(train_xfiles, train_yfiles, conv_kernel = True)
    valid_dset = build_dataset(valid_xfiles, valid_yfiles, conv_kernel = True)
    return train_dset, valid_dset




def get_train_test_datasets(start_dir, target_dir, kernel):
    """Gets the train and test datasets for a specified target directory."""
    conv_kernel = False
    if "conv" in kernel.lower():
        conv_kernel = True

    os.chdir(target_dir)
    train_xfiles, train_yfiles = get_file_list("train")
    test_xfiles, test_yfiles = get_file_list("test")
    train_dset = build_dataset(train_xfiles, train_yfiles, conv_kernel)
    os.chdir(start_dir)
    return train_dset, test_xfiles, test_yfiles


def get_train_dataset(start_dir, target_dir, kernel):
    """Gets the train dataset for a specified target directory when
    there is no test dataset."""
    conv_kernel = False
    if "conv" in kernel.lower():
        conv_kernel = True

    os.chdir(target_dir)
    train_xfiles, train_yfiles = get_file_list("train")
    train_dset = build_dataset(train_xfiles, train_yfiles, conv_kernel)
    os.chdir(start_dir)
    return train_dset


def get_file_list(target_dir):
    """Gets a list of files in a specified target directory."""
    current_dir = os.getcwd()
    os.chdir(target_dir)
    xfiles = [os.path.abspath(f) for f in os.listdir() if f.endswith("xvalues.npy")]
    yfiles = [os.path.abspath(f) for f in os.listdir() if f.endswith("yvalues.npy")]
    if len(xfiles) == 0 and len(yfiles) == 0:
        xfiles = [os.path.abspath(f) for f in os.listdir() if f.endswith("X.npy")]
        yfiles = [os.path.abspath(f) for f in os.listdir() if f.endswith("Y.npy")]
    xfiles.sort()
    yfiles.sort()
    os.chdir(current_dir)
    return xfiles, yfiles


def build_dataset(xfiles, yfiles, conv_kernel = False):
    """Converts the input file lists into dataset objects
    that can be fed to xGPR's hyperparameter tuning and fitting
    functions."""
    if conv_kernel:
        dataset = build_offline_sequence_dataset(xfiles, yfiles,
                            chunk_size = 2000, skip_safety_checks = True)
    else:
        dataset = build_offline_fixed_vector_dataset(xfiles, yfiles,
                            chunk_size = 2000, skip_safety_checks = True)
    return dataset


def tune_model_bayes(train_dset, low_train_rffs, high_train_rffs,
                    kernel = "RBF", random_seed = 123,
                    nmll_rank = 1024, pretransform_dir = None,
                    nmll_mode = "srht_2"):
    """Tunes the model using a specified train dataset, using
    the crude_bayes algorithm, and fine-tunes if so specified.

    Args:
        train_dset: An xGPR Dataset object with the disk locations
            of the data to load.
        low_train_rffs (int): The number of rffs to use for tuning.
        high_train_rffs: Either None or an int. If an int, the initial
            result of tuning with low_train_rffs is now retuned over
            a smaller bounding box around the region of the first
            best solution using a larger number of rffs.
        random_seed (int): A seed for the random number generator.
        nmll_rank (int): The rank for nmll estimation if fine
            tuning is desired.
        pretransform_dir: The location for pretransformation (if desired)
            for fine-tuning.
        nmll_mode (str): The mode for preconditioner construction.
            'srht_2' is slower to construct (but speeds up fitting);
            'srht' is faster to construct (but fitting may take longer).

    Returns:
        xgp: An xGPRegression object with tuned hyperparameters.
        hyperparams (str): The tuned hyperparamters as a string ready
            to write to file.
        nfev (int): The number of function evaluations during tuning.
        tuning_wclock (float): The time in seconds required to tune.
        nmll (float): The best score achieved.
    """
    tuning_wclock = time.time()
    print(f"Now tuning using kernel {kernel}", flush=True)
    xgp = xGPRegression(training_rffs = low_train_rffs, fitting_rffs = 512,
            variance_rffs = 16,
            kernel_choice = kernel,
            kernel_specific_params =
            {"matern_nu":5.0/2.0, "conv_width":9,
                "polydegree":3},
            device = "gpu", verbose = True)

    _, nfev, nmll, _ = xgp.tune_hyperparams_crude_bayes(train_dset,
                                max_bayes_iter = 50,
                                random_seed = random_seed)

    if high_train_rffs is not None:
        xgp.training_rffs = high_train_rffs
        _, nfev1, nmll = xgp.tune_hyperparams_fine_direct(train_dset,
                         optim_method = "Powell",
                         random_seed = 123, max_iter = 75,
                         nmll_rank = nmll_rank, nmll_tol = 1e-6,
                         pretransform_dir = pretransform_dir,
                         preconditioner_mode = nmll_mode)

        nfev += nfev1


    tuning_wclock = time.time() - tuning_wclock
    hyperparams = xgp.kernel.get_hyperparams(logspace=True)
    hyperparams = "_".join([str(z) for z in hyperparams.tolist()])
    print(hyperparams, flush=True)
    return xgp, hyperparams, nfev, tuning_wclock, nmll



def tune_model_lbfgs(train_dset, low_train_rffs,
                    kernel = "MiniARD", random_seed = 123):
    """Tunes the model using a specified train dataset, using
    the crude_lbfgs algorithm. Only used for kernels with > 3
    hyperparameters (i.e., only the MiniARD kernel).

    Args:
        train_dset: An xGPR Dataset object with the disk locations
            of the data to load.
        low_train_rffs (int): The number of rffs to use for tuning.
        random_seed (int): A seed for the random number generator.

    Returns:
        xgp: An xGPRegression object with tuned hyperparameters.
        hyperparams (str): The tuned hyperparamters as a string ready
            to write to file.
        nfev (int): The number of function evaluations during tuning.
        tuning_wclock (float): The time in seconds required to tune.
        nmll (float): The best score achieved.
    """
    tuning_wclock = time.time()
    print(f"Now tuning using kernel {kernel}", flush=True)
    #The split points here are for the GB1 dataset -- the only
    #dataset where it would make sense to use MiniARD.
    xgp = xGPRegression(training_rffs = low_train_rffs, fitting_rffs = 512,
            variance_rffs = 16,
            kernel_choice = kernel,
            kernel_specific_params =
            {"split_points":[21,42,63]},
            device = "gpu", verbose = True)

    _, nfev, nmll = xgp.tune_hyperparams_crude_lbfgs(train_dset,
                                max_iter = 50, n_restarts = 3,
                                random_seed = random_seed)


    tuning_wclock = time.time() - tuning_wclock
    hyperparams = xgp.kernel.get_hyperparams(logspace=True)
    hyperparams = "_".join([str(z) for z in hyperparams.tolist()])
    print(hyperparams, flush=True)
    return xgp, hyperparams, nfev, tuning_wclock, nmll



def fit_and_score(xgp, train_dset, test_xfiles, test_yfiles, score_type = "standard",
        precond_max_rank = 1024, hparams = None, random_seed = 123,
        pretransform_dir = None, mode = "cg", tol = 1e-5,
        precond_method = "srht_2"):
    """Fits a model to a specified train dataset, scores it on a specified
    test dataset, and returns the results and the time required to fit (seconds)."""

    wallclock_time = time.time()
    if pretransform_dir is not None:
        dataset = xgp.pretransform_data(train_dset, pretransform_dir,
                random_seed = random_seed, preset_hyperparams = hparams)
    else:
        dataset = train_dset

    preconditioner, ratio = xgp.build_preconditioner(dataset,
                    max_rank = precond_max_rank, preset_hyperparams = hparams,
                    random_state = random_seed, method = precond_method)
    print(ratio, flush=True)
    xgp.fit(dataset, preconditioner = preconditioner,
            preset_hyperparams = hparams, random_seed = random_seed,
            mode = mode, tol = tol, suppress_var = True,
            max_iter = 500)
    wallclock_time = time.time() - wallclock_time

    all_preds, all_y = [], []

    for xfile, yfile in zip(test_xfiles, test_yfiles):
        xdata, ydata = np.load(xfile), np.load(yfile)
        all_preds.append(xgp.predict(xdata, get_var=False))
        all_y.append(ydata)
    all_y = np.concatenate(all_y)
    all_preds = np.concatenate(all_preds)

    if score_type == "standard":
        spearman_score = spearmanr(all_preds, all_y)[0]
        mae_score = np.mean(np.abs(all_preds - all_y))
        return spearman_score, mae_score, wallclock_time

    mse_score = np.mean( (all_preds - all_y)**2 )
    mae_score = np.mean(np.abs(all_preds - all_y))

    return mse_score, mae_score, wallclock_time


def online_fit_and_score(xgp, dataset, testx, testy, precond_max_rank = 1024,
                random_seed = 123):
    """Fits a model to a specified train dataset, scores it on a batch of
    test data stored in memory, and returns the results and the time
    required to fit (seconds)."""

    wallclock_time = time.time()

    preconditioner, ratio = xgp.build_preconditioner(dataset,
                    max_rank = precond_max_rank, random_state = random_seed,
                    method = "srht_2")
    print(ratio, flush=True)
    xgp.fit(dataset, preconditioner = preconditioner,
            random_seed = random_seed, mode = "cg",
            tol = 1e-7, suppress_var = True, max_iter = 500)
    wallclock_time = time.time() - wallclock_time

    all_preds = xgp.predict(testx, get_var = False)
    mse_score = np.mean( (all_preds - testy)**2 )
    mae_score = np.mean(np.abs(all_preds - testy))
    return mse_score, mae_score, wallclock_time
