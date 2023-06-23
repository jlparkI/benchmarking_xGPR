"""Contains some settings for the UCI dataset tests."""
training_low_rffs = [3000, 3000]
training_high_rffs = [None, 8192]
fitting_rffs = [8192, 16384, 32768]
nmll_settings = {"kin40k_dataset/y_norm":[1500,"srht_2"],
               "song_dataset/y_norm":[1024,"srht"],
               "uci_protein_dataset/y_norm":[1024,"srht"]}
