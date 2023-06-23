"""Parameters for use for the uncertainty calibration test."""

#Stores the locations of target datasets, the best-performing kernel
#from the tune / train experiments and its associated hyperparameters.
#We hard-code the hyperparameters here to avoid any ambiguity.
target_datasets = {("aav_eval", "esm_reps", "conv_mut_des_split"):
                    ["RBF",[-0.7729011,0.2490016,-7.228606]],
                ("aav_eval", "onehot_conv", "seven_vs_many_split"):
                    ["FHTConv1d",[-0.7277704,-1.6094379,-0.5075]],
                #GB1
                ("gb1_eval", "onehot", "three_vs_rest"):
                    ["MiniARD",[-0.783322,  0.71016635, -1.08017348, -1.45222903, -0.41554267, -0.43705396]],
                ("gb1_eval", "onehot", "two_vs_rest"):
                    ["MiniARD",[-0.92332823,  0.89961381, -1.40687356, -1.18507571, -0.45145798, -0.63646826]],
                #Stability
                ("stability", "esm_reps", "standard"):
                    ["RBF",[-0.3039924, 0.4634613, -1.8111963]],
                #Thermostability
                ("thermostability", "esm_reps", "mixed_split"):
                    ["RBF",[-0.5923784, -0.2538536, -1.8111963]],
                }
