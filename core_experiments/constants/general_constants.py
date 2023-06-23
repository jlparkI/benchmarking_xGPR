"""Defines preset search parameters for the protein datasets."""
import os
import numpy as np

#Low train rffs is used for Bayesian optimization, followed by L-BFGS with
#high train rffs. Precond rank is the rank for preconditioner construction
#(larger = more expensive, but likely greater reduction in number of
#fitting iterations).
dset_settings = {"song_dataset":{"low_train_rffs":3000, "high_train_rffs":8192,
                    "precond_rank":1024, "fitting_rff":32768,
                    "precond_method":"srht"},
                "rossman":{"low_train_rffs":3000, "high_train_rffs":8192,
                    "precond_rank":2048, "fitting_rff":32768,
                     "precond_method":"srht_2"},
                "sulfur":{"low_train_rffs":3000, "high_train_rffs":8192,
                    "precond_rank":1024, "fitting_rff":32768,
                    "precond_method":"srht_2"}
                }

#Hyperparameter boundaries for running the optimizer test.
OPTIMIZER_TEST_BOUNDS = np.log(np.asarray([[1e-3,1e1], [0.2,5], [1e-6,1e2]]))
