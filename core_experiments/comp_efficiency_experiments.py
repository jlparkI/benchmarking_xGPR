"""Contains tools for performing experiments specifically related
to computational efficiency (e.g. preconditioned CG, approximate NMLL
etc)."""
import os
import time

import numpy as np
import cupy as cp

from xGPR.xGP_Regression import xGPRegression
from .constants import general_constants, cg_constants
from .core_exp_funcs import get_train_test_datasets


def run_approx_nmll_tests(start_dir):
    """Test the approximate nmll estimation procedure.

    Args:
        start_dir (str): The dir where this script is located.
    """

    os.chdir(start_dir)

    xgp = xGPRegression(training_rffs = 4096, fitting_rffs = 4096,
                        kernel_choice = "RBF", device = "gpu",
                        verbose = True)
    rng = np.random.default_rng(123)

    for data_fpath in cg_constants.target_datasets.keys():
        train_dset, _, _ = get_train_test_datasets(start_dir, os.path.join(start_dir,
            "benchmark_evals", data_fpath), "RBF")
        #In general the user will not initialize a kernel outside of a tuning
        #or fitting procedure; we are only doing so here for testing purposes.
        xgp.kernel = xgp._initialize_kernel(xgp.kernel_choice, train_dset.get_xdim(),
                    xgp.training_rffs, 123)
        bounds = xgp.kernel.get_bounds()

        print(f"Now working on {data_fpath}")
        hparam_samples = [rng.uniform(low=bounds[:,0], high=bounds[:,1], size=3)
                        for i in range(8)]

        for hparams in hparam_samples:
            #We're going to access some protected member methods here that the
            #end user will not generally need to access
            train_dset.device = "gpu"
            approx_nmll = xgp.approximate_nmll(hparams, train_dset,
                    max_rank = 512, nsamples = 25, random_seed = 123,
                    tol = 1e-5)
            print("Now calculating exact NMLL...this might take a second...")
            exact_nmll = xgp.exact_nmll(hparams, train_dset)

            z_trans_z, _, _ = calc_design_mat(train_dset, xgp.kernel)
            z_trans_z.flat[::z_trans_z.shape[0]+1] += xgp.kernel.get_lambda()**2
            _, s_1, _ = cp.linalg.svd(z_trans_z)
            condition_number = s_1.max() / s_1.min()
            print(f"{z_trans_z.shape}")

            with open(os.path.join(start_dir, "final_results",
                "approx_nmll_log.txt"), "a+", encoding="utf8") as output_file:
                output_file.write(f"{data_fpath},RBF,"
                    f"{'_'.join([str(z) for z in hparams.tolist()])},"
                    f"{xgp.training_rffs},{exact_nmll},{approx_nmll},"
                    f"{condition_number}\n")


def run_fitcomp_tests(start_dir):
    """Compare several important fitting methods.

    Args:
        start_dir (str): The dir where this script is located.
    """

    os.chdir(start_dir)
    logpath = os.path.abspath(os.path.join("final_results", "fitcomp_log.txt"))

    for data_fpath, (hparams, max_rank) in cg_constants.fitcomp_preset_hyperparams.items():
        train_dset, _, _ = get_train_test_datasets(start_dir, os.path.join(start_dir,
            "benchmark_evals", data_fpath), "RBF")
        print(f"Now working on {data_fpath}, {max_rank}")
        xgp = xGPRegression(training_rffs = 512,
                        variance_rffs = 64, fitting_rffs = 16384,
                        kernel_choice = "RBF", device = "gpu",
                        verbose = True)
        preconditioner, ratio = xgp.build_preconditioner(train_dset,
                        max_rank = max_rank, preset_hyperparams = hparams)

        _, losses = xgp.fit(train_dset, tol = 1e-6,
                        preset_hyperparams = hparams,
                        run_diagnostics = True,
                        max_iter = 100,
                        mode = "cg")
        losses = "_".join([str(z) for z in losses])
        hyperparams = "_".join([str(z) for z in hparams.tolist()])
        with open(logpath, "a+", encoding="utf8") as output_file:
            output_file.write(f"{data_fpath},RBF,"
                        f"{hyperparams},"
                        f"16384,CG,NA,"
                        f"no_preconditioner,{losses}\n")

        _, losses = xgp.fit(train_dset, tol = 1e-6,
                        preset_hyperparams = hparams,
                        run_diagnostics = True,
                        max_iter = 100,
                        mode = "sgd")
        losses = "_".join([str(z) for z in losses])
        hyperparams = "_".join([str(z) for z in hparams.tolist()])
        with open(logpath, "a+", encoding="utf8") as output_file:
            output_file.write(f"{data_fpath},RBF,"
                        f"{hyperparams},"
                        f"16384,SVRG,NA,"
                        f"no_preconditioner,{losses}\n")

        _, losses = xgp.fit(train_dset, tol = 1e-6,
                        preset_hyperparams = hparams,
                        run_diagnostics = True,
                        max_iter = 100,
                        mode = "amsgrad")
        losses = "_".join([str(z) for z in losses])
        hyperparams = "_".join([str(z) for z in hparams.tolist()])
        with open(logpath, "a+", encoding="utf8") as output_file:
            output_file.write(f"{data_fpath},RBF,"
                        f"{hyperparams},"
                        f"16384,AMSGrad,NA,"
                        f"no_preconditioner,{losses}\n")

        _, losses = xgp.fit(train_dset, tol = 1e-6,
                        preset_hyperparams = hparams,
                        run_diagnostics = True,
                        max_iter = 100,
                        preconditioner = preconditioner,
                        mode = "cg")
        losses = "_".join([str(z) for z in losses])
        hyperparams = "_".join([str(z) for z in hparams.tolist()])
        with open(logpath, "a+", encoding="utf8") as output_file:
            output_file.write(f"{data_fpath},RBF,"
                        f"{hyperparams},"
                        f"16384,CG,{ratio},"
                        f"{max_rank},{losses}\n")

        _, losses = xgp.fit(train_dset, tol = 1e-6,
                        preset_hyperparams = hparams,
                        run_diagnostics = True,
                        max_iter = 100,
                        preconditioner = preconditioner,
                        mode = "sgd")
        losses = "_".join([str(z) for z in losses])
        hyperparams = "_".join([str(z) for z in hparams.tolist()])
        with open(logpath, "a+", encoding="utf8") as output_file:
            output_file.write(f"{data_fpath},RBF,"
                        f"{hyperparams},"
                        f"16384,SVRG,{ratio},"
                        f"{max_rank},{losses}\n")



def run_lbfgs_tests(start_dir):
    """Test lbfgs procedures and save the number of iterations
    required to converge.

    Args:
        start_dir (str): The dir where this script is located.
    """

    os.chdir(start_dir)

    for data_fpath, hparams in cg_constants.target_datasets.items():
        train_dset, _, _ = get_train_test_datasets(start_dir, os.path.join(start_dir,
                "benchmark_evals", data_fpath), "RBF")
        for fitting_rff in cg_constants.fitting_rffs:
            print(f"Now working on {data_fpath}, L-BFGS")
            wclock = time.time()
            xgp = xGPRegression(training_rffs = 512,
                        variance_rffs = 64, fitting_rffs = fitting_rff,
                        kernel_choice = "RBF", device = "gpu",
                        verbose = True)
            niter, _ = xgp.fit(train_dset,
                        tol = 1e-6,
                        preset_hyperparams = hparams,
                        run_diagnostics = True,
                        max_iter = 1000,
                        mode = "lbfgs")
            wclock = time.time() - wclock
            hyperparams = "_".join([str(z) for z in hparams.tolist()])
            with open(os.path.join(start_dir, "final_results", "cglog.txt"),
                    "a+", encoding="utf8") as output_file:
                output_file.write(f"LBFGS,{data_fpath},RBF,"
                        f"{hyperparams},"
                        f"{fitting_rff},"
                        f"{wclock},{niter},"
                        "NA,NA,NA\n")


def run_cg_tests(start_dir):
    """Test cg procedures and save the number of iterations
    required to converge.

    Args:
        start_dir (str): The dir where this script is located.
    """

    os.chdir(start_dir)

    for data_fpath, hparams in cg_constants.target_datasets.items():
        train_dset, _, _ = get_train_test_datasets(start_dir, os.path.join(start_dir,
            "benchmark_evals", data_fpath), "RBF")
        for fitting_rff in cg_constants.fitting_rffs:
            for method in ["gauss", "srht", "srht_2"]:
                if method != "srht" and fitting_rff > 16384:
                    continue
                for max_rank in cg_constants.precond_rank:
                    print(f"Now working on {data_fpath}, {max_rank}")
                    wclock = time.time()
                    xgp = xGPRegression(training_rffs = 512,
                        variance_rffs = 64, fitting_rffs = fitting_rff,
                        kernel_choice = "RBF", device = "gpu",
                        verbose = True)
                    if max_rank == 0:
                        if method != "srht":
                            continue
                        ratio = 0
                        preconditioner = None
                    else:
                        preconditioner, ratio = xgp.build_preconditioner(train_dset,
                            max_rank = max_rank, preset_hyperparams = hparams,
                            method = method)
                    niter, _ = xgp.fit(train_dset,
                            preconditioner = preconditioner,
                            tol = 1e-5,
                            preset_hyperparams = hparams,
                            run_diagnostics = True,
                            max_iter = 1000,
                            mode = "cg")
                    wclock = time.time() - wclock
                    hyperparams = "_".join([str(z) for z in hparams.tolist()])
                    with open(os.path.join(start_dir, "final_results", "cglog.txt"),
                            "a+", encoding="utf8") as output_file:
                        output_file.write(f"CG,{data_fpath},RBF,"
                            f"{hyperparams},"
                            f"{fitting_rff},"
                            f"{wclock},{niter},"
                            f"{ratio},{max_rank},{method}\n")




def run_optimizer_tests(start_dir):
    """Tests some of the various optimizers to check 1)
    the best score they are able to achieve and 2) number
    of func evals required to do it. Run using double-precision,
    predefined bounds for clarity.

    Args:
        start_dir (str): The filepath where this script is located.
    """
    os.chdir(start_dir)
    logpath = os.path.abspath(os.path.join("final_results", "optimizelog.txt"))

    for data_fpath in cg_constants.target_datasets.keys():
        train_dset, _, _ = get_train_test_datasets(start_dir, os.path.join(start_dir,
            "benchmark_evals", data_fpath), "RBF")
        for optimizer in ["bayes", "bfgs"]:
            print(f"Now working on {data_fpath}, {optimizer}.")
            tuning_wclock = time.time()
            xgp = xGPRegression(training_rffs = 512, fitting_rffs = 512,
                        variance_rffs = 64, kernel_choice = "RBF",
                        device = "gpu", verbose = True)
            if optimizer == "bayes":
                #Set eigval_quotient to a very large value and min_eigval to very small
                #to completely turn off filtering for small eigenvalues, which is beneficial
                #for stability but has a small (<1%) impact on best marginal likelihood
                hparam, nfev, mnll, _ = xgp.tune_hyperparams_crude_bayes(train_dset,
                                    bounds = general_constants.OPTIMIZER_TEST_BOUNDS,
                                    bayes_tol = 1e-2, n_pts_per_dim = 30, n_cycles = 4,
                                    eigval_quotient = 1e10, min_eigval = 1e-6)
            elif optimizer == "bfgs":
                hparam, nfev, mnll = xgp.tune_hyperparams_crude_lbfgs(train_dset,
                                    n_restarts = 5,
                                    bounds = general_constants.OPTIMIZER_TEST_BOUNDS)
            tuning_wclock = time.time() - tuning_wclock
            hparam = "_".join([str(z) for z in hparam.tolist()])
            with open(logpath, "a+", encoding="utf8") as output_file:
                data_name = "_".join(data_fpath.split("/")[-3:])
                output_file.write(f"{data_name},RBF,"
                    f"{hparam},"
                    f"512,"
                    f"{tuning_wclock},{nfev},"
                    f"{optimizer},{mnll}\n")
