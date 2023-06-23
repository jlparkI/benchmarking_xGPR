"""Defines tuning params for protein datasets fit with convolution kernels."""
conv_tuning_params = {("aav_eval", "conv_features", "onehot_conv","des_mut_split"):["RBF", 3000, None],
        ("aav_eval", "onehot_conv", "des_mut_split"):["FHTConv1d", 3000, None],
        ("aav_eval", "esm_reps", "conv_des_mut_split"):["RBF", 3000, None],
        ("aav_eval", "esm_reps", "des_mut_split"):["RBF", 3000, None],
        #AAV: mut_des_split
        ("aav_eval", "onehot_conv", "mut_des_split"):["FHTConv1d", 3000, None],
        ("aav_eval", "conv_features", "onehot_conv", "mut_des_split"):["RBF", 3000, None],
        ("aav_eval", "esm_reps", "conv_mut_des_split"):["RBF", 3000, None],
        ("aav_eval", "esm_reps", "mut_des_split"):["RBF", 3000, None],
        #AAV: seven_vs_many
        ("aav_eval", "conv_features", "onehot_conv", "seven_vs_many_split"):["RBF", 3000, None],
        ("aav_eval", "onehot_conv", "seven_vs_many_split"):["FHTConv1d", 3000, None],
        ("aav_eval", "esm_reps", "conv_seven_vs_many_split"):["RBF", 3000, None],
        ("aav_eval", "esm_reps", "seven_vs_many_split"):["RBF", 3000, None],
        #AAV: two_vs_many
        ("aav_eval", "conv_features", "onehot_conv", "two_vs_many_split"):["RBF", 3000, None],
        ("aav_eval", "onehot_conv", "two_vs_many_split"):["FHTConv1d", 3000, None],
        ("aav_eval", "esm_reps", "conv_two_vs_many_split"):["RBF", 3000, None],
        ("aav_eval", "esm_reps", "two_vs_many_split"):["RBF", 3000, None],
        #AAV: one_vs_many
        ("aav_eval", "conv_features", "onehot_conv", "one_vs_many_split"):["RBF", 3000, None],
        ("aav_eval", "onehot_conv", "one_vs_many_split"):["FHTConv1d", 3000, None],
        ("aav_eval", "esm_reps", "conv_one_vs_many_split"):["RBF", 3000, None],
        ("aav_eval", "esm_reps", "one_vs_many_split"):["RBF", 3000, None],
        #GB1 -- the smallest dataset -- with just 4 aas, it's small enough we can
        #use a separate lengthscale for each aa when using one-hot encoding.
        ("gb1_eval", "onehot", "three_vs_rest"):["MiniARD", 1024, None],
        ("gb1_eval", "onehot", "two_vs_rest"):["MiniARD", 1024, None],
        ("gb1_eval", "esm_reps", "three_vs_rest"):["RBF", 1024, None],
        ("gb1_eval", "esm_reps", "two_vs_rest"):["RBF", 1024, None],
        #Stability
        ("stability", "conv_features", "onehot_conv", "standard"):["RBF", 3000, None],
        ("stability", "onehot_conv", "standard"):["FHTConv1d", 3000, None],
        ("stability", "esm_reps", "conv"):["RBF", 3000, None],
        ("stability", "esm_reps", "standard"):["RBF", 3000, None],
        #Thermostability: human
        ("thermostability", "conv_features", "onehot_conv", "human"):["RBF", 3000, None],
        ("thermostability", "esm_reps", "conv_human"):["RBF", 3000, None],
        ("thermostability", "esm_reps", "human"):["RBF", 3000, None],
        #Thermostability: mixed:
        ("thermostability", "conv_features", "onehot_conv", "mixed_split"):["RBF", 3000, None],
        ("thermostability", "esm_reps", "conv_mixed_split"):["RBF", 3000, None],
        ("thermostability", "esm_reps", "mixed_split"):["RBF", 3000, None],
        #Fluorescence
        ("fluorescence_eval", "conv_features", "onehot_conv", "standard"):["RBF", 3000, None],
        ("fluorescence_eval", "onehot_conv", "standard"):["FHTConv1d", 3000, None],
        ("fluorescence_eval", "esm_reps", "conv"):["RBF", 3000, None],
        ("fluorescence_eval", "esm_reps", "standard"):["RBF", 3000, None],
        }
