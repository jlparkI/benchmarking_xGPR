"""Contains the tools needed to reproduce other miscellaneous experiments."""
import os

from .constants import uci_constants
from .core_exp_funcs import tune_model_bayes, fit_and_score
from .core_exp_funcs import get_train_test_datasets


def run_uci_tests(start_dir):
    """Run tests with the UCI datasets for comparison with GPyTorch."""
    for data_fpath, nmll_settings in uci_constants.nmll_settings.items():
        train_dset, test_xfiles, test_yfiles = get_train_test_datasets(start_dir,
                        os.path.join(start_dir, "benchmark_evals", data_fpath), "RBF")
        for training_rff in zip(uci_constants.training_low_rffs,
                                    uci_constants.training_high_rffs):
            print(f"Now working on {data_fpath}, {training_rff}")
            xgp, hyperparams, nfev, tuning_wclock, _ = tune_model_bayes(train_dset,
                                training_rff[0], training_rff[1], kernel = "RBF",
                                nmll_rank = nmll_settings[0],
                                nmll_mode = nmll_settings[1])
            for fitting_rff in uci_constants.fitting_rffs:
                xgp.fitting_rffs = fitting_rff
                spearman_score, mae_score, fitting_wclock = fit_and_score(xgp, train_dset,
                            test_xfiles, test_yfiles, mode = "cg", tol = 1e-6,
                            precond_method = nmll_settings[1],
                            precond_max_rank = nmll_settings[0])
                with open(os.path.join(start_dir, "final_results",
                        "ucilog.txt"), "a+", encoding="utf8") as output_file:
                    output_file.write(f"{data_fpath},RBF,"
                        f"{hyperparams},"
                        f"{training_rff[1]},"
                        f"{fitting_rff},"
                        f"{tuning_wclock},{nfev},"
                        f"{fitting_wclock},{spearman_score},"
                        f"{mae_score},bayes\n")
