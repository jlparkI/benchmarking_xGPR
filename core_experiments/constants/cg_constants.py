"""Parameters for use in testing conjugate gradients with and without
preconditioning."""
import os
import numpy as np

fitting_rffs = [16384, 32768]
precond_rank = [0, 256, 512, 1024]

#This dictionary maintains the list of datasets used for testing how fast
#conjugate gradients fits with and without preconditioning. The exact hyperparameters
#used do not matter to this comparison, but we use the ones obtained from prior
#tuning experiments for convenience.
target_datasets = {os.path.join("kin40k_dataset", "y_norm"):
            np.array([-1.4701868288258673,1.488503077974549,-0.5559783514308828]),
        os.path.join("song_dataset", "y_norm"):
            np.array([-0.18500497156080836,-0.9688407884695313,-2.3766176234444494]),
        os.path.join("uci_protein_dataset", "y_norm"):
            np.array([-0.41483738691418204,-0.19847253218546643,0.45796802733361996]),
        os.path.join("fluorescence_eval", "conv_features", "onehot_conv", "standard"):
            np.array([-0.7729011,1.4412146999999997,-4.501004]),
        os.path.join("gb1_eval", "onehot", "three_vs_rest"):
            np.array([-0.8832280464765179,0.23252013335197366,-0.7564112993375892]),
        os.path.join("gb1_eval", "onehot", "two_vs_rest"):
            np.array([-1.0572382378575833,0.17987230132295984,-0.6770193344307492]),
        os.path.join("aav_eval", "conv_features", "onehot_conv", "mut_des_split"):
            np.array([-0.6826396999999998,-0.11605590000000002,-4.8745935]),
        os.path.join("aav_eval", "conv_features", "onehot_conv", "seven_vs_many_split"):
            np.array([-0.7277703999999999,0.24900159999999985,-5.0760609])
        }


#This list and dictionary store parameters for a more detailed comparison of
#conjugate gradients with stochastic gradient descent methods. The hyperparameters
#and then the max rank for preconditioner construction are indexed by the filepath.
fitcomp_max_rank = [1600, 1024, 1024, 1024]
fitcomp_preset_hyperparams = {os.path.join("kin40k_dataset", "y_norm"):
            (np.array([-1.4701868288258673,1.488503077974549,-0.5559783514308828]),
                1600),
        os.path.join("song_dataset", "y_norm"):
            (np.array([-0.18500497156080836,-0.9688407884695313,-2.3766176234444494]),
                1024),
        os.path.join("uci_protein_dataset", "y_norm"):
            (np.array([-0.41483738691418204,-0.19847253218546643,0.45796802733361996]),
                1024),
        os.path.join("aav_eval", "conv_features", "onehot_conv", "mut_des_split"):
            (np.array([-0.6826396999999998,-0.11605590000000002,-4.8745935]),
                1024)
        }
