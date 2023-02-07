"""Preps the thermostability dataset for evaluation."""
import os
import pandas as pd
from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit

#There are a few (three or four) outrageously long sequences in this
#dataset. Luckily they are all in the training set, so we can
#exclude them without affecting our ability to evaluate the model.
#We exclude these so that we do not end up needing to zero pad to
#outrageously long lengths.
MAX_SEQLEN = 6994


def prep_thermostab(start_dir):
    """Preps the thermostability dataset for evaluation."""
    os.chdir(os.path.join("benchmark_evals", "thermostability", "raw_data"))
    if "human.csv" not in os.listdir():
        raise ValueError("The thermostability dataset has not been downloaded yet!")
    raw_data = load_raw_data()
    splits = ["human", "mixed_split"]
    for dataset, split in zip(raw_data, splits):
        os.chdir(start_dir)
        os.chdir(os.path.join("benchmark_evals", "thermostability"))
        print(f"Now encoding split {split}")
        if "onehot_conv" not in os.listdir():
            os.mkdir("onehot_conv")
        generate_split_arrays(start_dir, "onehot_conv",
                            split, "onehot_conv", dataset)
    print("Thermostab encoding is complete.")


def generate_split_arrays(start_dir, dest_dir, split,
                            enc_type, raw_data):
    """Generate arrays with encodings for a specific train-test
    split."""
    encoder = MSAEncodingToolkit(enc_type.split("_conv")[0])

    os.chdir(dest_dir)
    subset = raw_data[raw_data["set"].notna()]
    train = subset[subset["set"]=="train"]

    validation = train[train["validation"].notna()]
    test = subset[subset["set"]=="test"]
    input_dfs = [train, validation, test]
    dfnames = ["train", "valid", "test"]
    if split not in os.listdir():
        os.mkdir(split)
    os.chdir(split)

    for (input_df, dfname) in zip(input_dfs, dfnames):
        raw_seqs = input_df["sequence"].tolist()
        yvals = input_df["target"].values
        if dfname not in os.listdir():
            os.mkdir(dfname)
        encoder.encode_sequence_list(raw_seqs, yvals, dfname,
                    blocksize=200, mode="conv", fixed_len = MAX_SEQLEN)
        os.chdir("..")
    os.chdir(start_dir)


def load_raw_data():
    """Load the raw data and eliminate sequences that contain
    U or X (since we do not know what amino acid is present
    we cannot really model these)"""
    human = pd.read_csv("human.csv")
    mixed = pd.read_csv("mixed_split.csv")
    mixed["len"] = mixed.sequence.str.len()
    human["len"] = human.sequence.str.len()

    human = human[~human.sequence.str.contains("U")]
    mixed = mixed[~mixed.sequence.str.contains("U")]
    mixed = mixed[~mixed.sequence.str.contains("X")]
    human = human[~human.sequence.str.contains("X")]

    human = human[human["len"]<=6994].copy()
    mixed = mixed[mixed["len"]<=6994].copy()
    return [human, mixed]
