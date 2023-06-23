"""Contains the tools needed to reproduce experiments specific to QM9."""
import os
import time

import pandas as pd
import numpy as np

from xGPR.xGP_Regression import xGPRegression
from xGPR.data_handling.dataset_builder import build_offline_sequence_dataset
from .core_exp_funcs import get_qm9_train_dataset, get_target_qm9_files



def molfit(start_dir, pretransform_dir = None):
    """Fits the model for QM9."""
    os.chdir(start_dir)
    logpath = os.path.abspath(os.path.join("final_results", "molfit.txt"))
    hparam_df = pd.read_csv(os.path.join("final_results", "moltune.txt"))

    for i in range(hparam_df.shape[0]):
        preset_hparams = np.asarray([float(f) for f in hparam_df.iloc[i,2].split("_")])
        kernel = hparam_df.iloc[i,1]
        dataset_specifier = hparam_df.iloc[i,0]
        target_dir = os.path.join(start_dir, "benchmark_evals", "chemdata", "full_soap")
        split_pts = []
        os.chdir(target_dir)
        qm9_model = xGPRegression(training_rffs = 12, fitting_rffs = 16384,
                            device = "gpu", kernel_choice = kernel,
                            verbose = True,
                            kernel_specific_params = {"split_points":split_pts, "polydegree":2})

        for data_type in ["u298_atom", "h298_atom", "u0_atom", "zpve", "cv", "g298_atom"]:

            start_time = time.time()
            train_xfiles, train_yfiles = get_target_qm9_files("train", data_type)
            os.chdir("..")
            test_xfiles, test_yfiles = get_target_qm9_files("test", data_type)
            os.chdir(target_dir)

            train_dset = build_offline_sequence_dataset(train_xfiles, train_yfiles,
                            skip_safety_checks=True)

            for fitting_rff in [16384, 32768, 65536]:
                qm9_model.fitting_rffs = fitting_rff
                pre_dset = qm9_model.pretransform_data(train_dset,
                        pretransform_dir = pretransform_dir,
                        preset_hyperparams = preset_hparams)
                preconditioner, _ = qm9_model.build_preconditioner(pre_dset, max_rank = 3500,
                                               preset_hyperparams = preset_hparams, method="srht_2")

                qm9_model.fit(pre_dset, preconditioner=preconditioner, mode="cg",
                        preset_hyperparams = preset_hparams, suppress_var = True, tol=1e-9,
                        max_iter = 1000)

                end_time = time.time()
                fitting_wclock = end_time - start_time
                print(f"Fitting wallclock time: {fitting_wclock}", flush=True)

                all_preds, all_y = [], []
                for xfile, yfile in zip(test_xfiles, test_yfiles):
                    xdata, ydata = np.load(xfile), np.load(yfile)
                    preds = qm9_model.predict(xdata, get_var = False, chunk_size=100)
                    all_preds.append(preds)
                    all_y.append(ydata)

                all_preds = np.concatenate(all_preds)
                all_y = np.concatenate(all_y)
                mae = np.mean(np.abs(all_y - all_preds))
                print(f"MAE for data type {data_type} on {dataset_specifier} is {mae}", flush=True)

                with open(logpath, "a+", encoding="utf8") as output_file:
                    output_file.write(f"{dataset_specifier},{kernel},{hparam_df.iloc[i,2]},"
                            f"{fitting_rff},{fitting_wclock},{mae},{data_type}\n")
                pre_dset.delete_dataset_files()
