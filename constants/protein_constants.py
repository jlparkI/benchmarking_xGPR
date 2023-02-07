"""Defines preset search parameters for the protein datasets."""
kernel_select_settings = {("aav_eval", "conv_features", "des_mut_split"):["RBF"],
        ("aav_eval", "onehot_conv", "des_mut_split"):["Conv1d", "FHTConv1d"],
        ("aav_eval", "onehot", "des_mut_split"):["RBF"],
        #AAV: mut_des_split
        ("aav_eval", "conv_features", "mut_des_split"):["RBF"],
        ("aav_eval", "onehot_conv", "mut_des_split"):["Conv1d", "FHTConv1d"],
        ("aav_eval", "onehot", "mut_des_split"):["RBF"],
        #AAV: seven_vs_many
        ("aav_eval", "conv_features", "seven_vs_many_split"):["RBF"],
        ("aav_eval", "onehot_conv", "seven_vs_many_split"):["Conv1d", "FHTConv1d"],
        ("aav_eval", "onehot", "seven_vs_many_split"):["RBF"],
        #AAV: two_vs_many
        ("aav_eval", "conv_features", "two_vs_many_split"):["RBF"],
        ("aav_eval", "onehot_conv", "two_vs_many_split"):["Conv1d", "FHTConv1d"],
        ("aav_eval", "onehot", "two_vs_many_split"):["RBF"],
        #AAV: one_vs_many
        ("aav_eval", "conv_features", "one_vs_many_split"):["RBF"],
        ("aav_eval", "onehot_conv", "one_vs_many_split"):["Conv1d", "FHTConv1d"],
        ("aav_eval", "onehot", "one_vs_many_split"):["RBF"],
        #FLR
        ("fluorescence_eval", "onehot", "standard"):["RBF"],
        ("fluorescence_eval", "conv_features", "standard"):["RBF"],
        ("fluorescence_eval", "onehot_conv", "standard"):["Conv1d", "FHTConv1d"],
        #GB1: Three vs rest
        ("gb1_eval", "onehot", "three_vs_rest"):["RBF"],
        #GB1: Two vs rest
        ("gb1_eval", "onehot", "two_vs_rest"):["RBF"],
        #Stability
        ("stability", "conv_features", "standard"):["RBF"],
        ("stability", "onehot_conv", "standard"):["Conv1d", "FHTConv1d"],
        #Thermostability: human
        ("thermostability", "conv_features", "human"):["RBF"],
        #Thermostability: mixed:
        ("thermostability", "conv_features", "mixed_split"):["RBF"]}

#Settings are stored as (train_rffs, fit_rffs, preconditioner_rank, tune method).
#Larger preconditioner ranks speed up fitting but make preconditioner
#construction more expensive; we use a smaller rank whenever we can get
#away with it. L-BFGS is great for small datasets, much slower than bayes for large.
fitting_settings = {"aav_eval_des_mut_split":(3000, 8192, 1024, "bayes"),
                    "aav_eval_mut_des_split":(3000, 8192, 1024, "bayes"),
                    "aav_eval_seven_vs_many_split":(3000, 8192, 1024, "bayes"),
                    "aav_eval_two_vs_many_split":(3000, 8192, 1024, "bayes"),
                    "aav_eval_one_vs_many_split":(3000, 8192, 512, "bayes"),
                    "fluorescence_eval_standard":(3000, 8192, 1024, "bayes"),
                    "gb1_eval_three_vs_rest":(2048, 8192, 1024, "bayes"),
                    "gb1_eval_two_vs_rest":(2048, 8192, 512, "bayes"),
                    "stability_standard":(2048, 8192, 1024, "bayes"),
                    "thermostability_mixed_split":(3000, 8192, 1024, "bayes"),
                    "thermostability_human":(3000, 8192, 1024, "bayes")
                    }


MATERN_NU = 5/2
MAX_BAYES_ITER = 35
