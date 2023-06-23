"""Contains the tools needed to reproduce protein engineering experiments."""
import os

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, norm, sem

from xGPR.xGP_Regression import xGPRegression
from xGPR.data_handling.dataset_builder import build_online_dataset
from .constants import uq_constants
from .core_exp_funcs import tune_model_bayes, tune_model_lbfgs, fit_and_score
from .core_exp_funcs import get_train_test_datasets, get_train_dataset




def protein_tune(start_dir, dataset_specs, logfile_name):
    """Tunes hyperparameters for specified datasets."""
    os.chdir(start_dir)

    for dataset_specifier, tune_details in dataset_specs.items():
        dpath = os.path.join(*dataset_specifier)
        target_dir = os.path.join(start_dir, "benchmark_evals", dpath)

        kernel = tune_details[0]
        print(f"{kernel},{dpath}")
        for random_seed in [123, 124, 125]:
            print(f"Working on {dpath}, kernel {kernel}", flush = True)
            train_dset = get_train_dataset(start_dir, target_dir, kernel)
            if kernel != "MiniARD":
                _, hyperparams, _, tuning_wclock, nmll = tune_model_bayes(train_dset,
                            low_train_rffs = tune_details[1],
                            high_train_rffs = tune_details[2],
                            kernel = kernel, random_seed = random_seed)
            else:
                _, hyperparams, _, tuning_wclock, nmll = tune_model_lbfgs(train_dset,
                            low_train_rffs = tune_details[1],
                            kernel = kernel, random_seed = random_seed)

            with open(os.path.join(start_dir, "final_results", logfile_name),
                    "a+", encoding="utf8") as output_file:
                output_file.write(f"{dpath},{kernel},"
                        f"{hyperparams},"
                        f"{tune_details[1]},"
                        f"{tuning_wclock},{nmll},"
                        f"{random_seed}\n")


def protein_fit(start_dir, logfile_name, hparam_file, temp_dir = None):
    """Fits models for the protein datasets."""
    os.chdir(start_dir)

    os.chdir("final_results")
    kernel_sel = pd.read_csv(hparam_file)

    for dset in kernel_sel["Dataset"].unique().tolist():
        dset_results = kernel_sel[kernel_sel["Dataset"]==dset].copy()
        dset_results.reset_index(drop=True, inplace=True)
        if dset_results.shape[0] != 3:
            print("There are more / fewer than 3 hyperparameter tuning "
                    f"results for dset {dset}. Please clean up the "
                    "hyperparameter tuning log before running the fitting.")
            continue

        for i in range(dset_results.shape[0]):
            hparams = [float(s) for s in dset_results["Hyperparams"][i].split("_")]
            hparams = np.asarray(hparams)
            data_fpath = os.path.join(start_dir, "benchmark_evals", dset_results["Dataset"][i])
            kernel = dset_results["Kernel_Type"][i]
            print(f"Working on dataset {dset}, using kernel {kernel}", flush = True)
            if "conv" in kernel.lower():
                pretransform_dir = temp_dir
            else:
                pretransform_dir = None

            for fitting_rffs in [8192, 16384]:
                train_dset, test_xfiles, test_yfiles = get_train_test_datasets(start_dir,
                                 data_fpath, kernel)
                xgp = xGPRegression(training_rffs = 512,
                        fitting_rffs = fitting_rffs,
                        variance_rffs = 64, kernel_choice = kernel,
                        device = "gpu", verbose = True,
                        kernel_specific_params = {"conv_width":9, "matern_nu":5/2,
                            "split_points":[21,42,63]})


                spearman_score, _, fitting_wclock = fit_and_score(xgp, train_dset,
                        test_xfiles, test_yfiles, hparams = hparams,
                        random_seed = dset_results["random_seed"].values[i],
                        pretransform_dir = pretransform_dir,
                        precond_max_rank = 1024, tol = 1e-6)

                with open(os.path.join(start_dir, "final_results", logfile_name),
                        "a+", encoding="utf8") as output_file:
                    output_file.write(f"{dset},{kernel},"
                            f"{hparams},"
                            f"NA,{fitting_rffs},"
                            f"{fitting_wclock},{spearman_score},"
                            f"{dset_results['random_seed'].values[i]}\n")



def uncertainty_calibration(start_dir):
    """Evaluate uncertainty calibration. We primarily do this to
    compare with Greenman et al but also to provide some
    additional useful statistics.

    Args:
        start_dir (str): The dir where this script is located.
    """

    os.chdir(start_dir)

    for dataset_specifier, params in uq_constants.target_datasets.items():
        print(dataset_specifier)
        print(params)
        data_fpath = os.path.join(*dataset_specifier)
        train_dset, test_xfiles, test_yfiles = get_train_test_datasets(start_dir,
                os.path.join(start_dir, "benchmark_evals", data_fpath), params[0])

        xgp = xGPRegression(training_rffs = 512,
                        fitting_rffs = 8192,
                        variance_rffs = 2048, kernel_choice = params[0],
                        device = "gpu", verbose = True,
                        kernel_specific_params = {"conv_width":9, "split_points":[21,42,63]})


        spr_score, mae, auce, upper_bound, y_data = uncertainty_eval(xgp, train_dset,
                        test_xfiles, test_yfiles, hparams = np.array(params[1]))

        with open(os.path.join(start_dir, "final_results", "uncert_quant.txt"),
                            "a+", encoding="utf8") as output_file:
            output_file.write(f"{data_fpath},{params[0]},"
                        f"{'_'.join([str(z) for z in params[1]])},16384,"
                        f"{mae},{spr_score},"
                        f"{auce}\n")


def active_learning(start_dir):
    """Runs an experiment on the GB1 dataset in which we apply active
    learning and determine how long it takes us to find a really good
    sequence using Bayesian optimization over a 4-amino acid space."""
    os.chdir(os.path.join(start_dir, "benchmark_evals", "active_learn",
        "encoded_data", "GB1"))
    all_x_data = np.load("0_block_xvalues.npy").astype(np.float64)
    all_y_data = np.load("0_block_yvalues.npy").astype(np.float64)

    for i in range(50):
        y_means, y_maxes = active_learning_run(all_x_data, all_y_data,
                        123 + i)
        with open(os.path.join(start_dir, "final_results",
            "active_learn_log.txt"), "a+", encoding="utf8") as logfile:
            for j, (ymean, ymax) in enumerate(zip(y_means, y_maxes)):
                logfile.write(f"{123+i},{j},{str(ymean)},{str(ymax)}\n")

        print(f"Iteration {i} complete.", flush=True)



def active_learning_run(all_x_data, all_y_data, random_seed):
    """Performs a single active learning run."""
    model = xGPRegression(training_rffs = 1024, fitting_rffs = 8192, device = "gpu",
                    variance_rffs = 1024, kernel_choice = "RBF", verbose = False)
    tdset, t_x, t_y = active_learn_init_sample(all_x_data, all_y_data, random_seed)
    y_means, y_maxes = [], []

    for j in range(5):
        _ = model.tune_hyperparams_crude_bayes(tdset, max_bayes_iter=30)
        preconditioner, _ = model.build_preconditioner(tdset,
                                         max_rank = 256, method = 'srht')
        model.fit(tdset, preconditioner = preconditioner,
                        mode = "cg", tol=1e-6)
        tdset, t_x, t_y, sampled_y = active_learn_sample_and_stack(tdset.xdata_, tdset.ydata_,
                t_x, t_y, model)
        y_means.append(np.mean(sampled_y))
        y_maxes.append(np.max(sampled_y))
        #Report best so far.
        y_maxes[-1] = np.max(y_maxes)

    return y_means, y_maxes

def active_learn_init_sample(all_x, all_y, random_seed):
    """Randomly selects 384 samples from the GB1 dataset
    on which to train an initial model."""
    rng = np.random.default_rng(random_seed)
    ind = rng.permutation(all_x.shape[0])
    ind_train, ind_test = ind[:384], ind[384:]
    testx, testy = all_x[ind_test,:], all_y[ind_test]
    train_dset = build_online_dataset(all_x[ind_train,:],
                                      all_y[ind_train], chunk_size=2000)
    return train_dset, testx, testy


def active_learn_sample_and_stack(init_trainx, init_trainy, test_x, test_y, model):
    """Uses model predictions for the GB1 dataset to select a batch of 96
    sequences for 'experimental' evaluation, and adds these to the training
    set for the next round."""
    preds, var = model.predict(test_x, get_var = True, chunk_size=2000)
    best_idx = np.argsort(preds + 1.96 * var)[-96:]
    sampled_y = test_y[best_idx]

    train_x = np.vstack([init_trainx, test_x[best_idx,:]])
    train_y = np.concatenate([init_trainy, test_y[best_idx]])
    mask = np.ones(test_x.shape[0], dtype=bool)
    mask[best_idx] = False
    new_test_x = test_x[mask,:]
    new_test_y = test_y[mask]
    new_train_dset = build_online_dataset(train_x, train_y, chunk_size=2000)
    return new_train_dset, new_test_x, new_test_y, sampled_y



def uncertainty_eval(xgp, train_dset, test_xfiles, test_yfiles,
        hparams = None):
    """Performs experiments for quantifying uncertainty calibration on
    the same datasets used by Greenman et al. (to have a direct comparison).
    The Greenman et al. paper used several metrics, two of which are in our view
    not very informative; we therefore calculate the metrics that we believe
    have better justification."""
    preconditioner, _ = xgp.build_preconditioner(train_dset,
                    max_rank = 1024, preset_hyperparams = hparams,
                    random_state = 123, method = "srht_2")
    xgp.fit(train_dset, preconditioner = preconditioner,
            preset_hyperparams = hparams, random_seed = 123,
            mode = "cg", tol = 1e-6, suppress_var = False,
            max_iter = 500)

    all_preds, all_y, all_var = [], [], []

    for xfile, yfile in zip(test_xfiles, test_yfiles):
        xdata, ydata = np.load(xfile), np.load(yfile)
        preds, var = xgp.predict(xdata, get_var=True)
        all_y.append(ydata)
        all_preds.append(preds)
        all_var.append(var)

    all_y = np.concatenate(all_y)
    all_preds = np.concatenate(all_preds)
    all_var = np.concatenate(all_var)

    spearman_score = spearmanr(all_preds, all_y)[0]
    mae_score = np.mean(np.abs(all_preds - all_y))
    auce = calc_miscalibration(all_preds, all_y, all_var)

    upper_bound = norm.ppf((1+0.95)/2) * all_var + all_preds

    return spearman_score, mae_score, auce, upper_bound, all_y


def calc_miscalibration(preds, ground_truth, var):
    """Calculates the miscalibration area of each model."""
    epe = np.linspace(0, 1, 100).tolist()
    residuals = np.abs(preds - ground_truth)
    calibration_err = []
    for epe_val in epe:
        cutoff = norm.ppf((1+epe_val)/2)
        in_interval = (residuals <= (cutoff * np.sqrt(var)))
        fraction_in_interval = float(in_interval.sum()) / \
                float(residuals.shape[0])
        miscalibration = np.abs(fraction_in_interval - epe_val)
        calibration_err.append(miscalibration)
    return np.trapz(y=np.array(calibration_err),
            x = np.array(epe))
