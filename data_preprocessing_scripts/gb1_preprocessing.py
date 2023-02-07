"""Preps the GB1 data for evaluation."""
import os
import pandas as pd
from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit


def prep_gb1(start_dir):
    """Preps the GB1 test data for modeling."""
    os.chdir(os.path.join("benchmark_evals", "gb1_eval", "raw_data"))
    if "four_mutations_full_data.csv" not in os.listdir():
        raise ValueError("The GB1 dataset has not been downloaded yet!")

    raw_data = pd.read_csv("four_mutations_full_data.csv")
    splits = ["one_vs_rest", "two_vs_rest", "three_vs_rest"]
    target_dirs = ["onehot"]

    for i, enc_type in enumerate(target_dirs):
        for split in splits:
            os.chdir(start_dir)
            os.chdir(os.path.join("benchmark_evals", "gb1_eval"))
            print(f"Now encoding split {split} using {enc_type}")
            if target_dirs[i] not in os.listdir():
                os.mkdir(target_dirs[i])
            generate_split_arrays(start_dir, target_dirs[i],
                            split, enc_type, raw_data)

    print("GB1 encoding is complete.")


def generate_split_arrays(start_dir, dest_dir, split,
                            enc_type, raw_data):
    """Encodes the data for a given train-test split."""
    os.chdir(dest_dir)
    if "conv" in enc_type:
        mode = "conv"
        encoding_name = enc_type.split("_conv")[0]
    else:
        mode = "flat"
        encoding_name = enc_type

    encoder = MSAEncodingToolkit(encoding_name)
    subset = raw_data[raw_data[split].notna()]
    is_valid_colname = split + "_validation"

    train = subset[subset[split]=="train"]
    validation = train[train[is_valid_colname].notna()]
    train = train[train[is_valid_colname].isna()]
    test = subset[subset[split]=="test"]

    input_dfs = [train, validation, test]
    dfnames = ["train", "valid", "test"]
    if split not in os.listdir():
        os.mkdir(split)
    os.chdir(split)

    for (input_df, dfname) in zip(input_dfs, dfnames):
        raw_seqs = input_df["Variants"].tolist()
        yvals = input_df["Fitness"].values
        if dfname not in os.listdir():
            os.mkdir(dfname)
        encoder.encode_sequence_list(raw_seqs, yvals, dfname,
                blocksize=2000, mode = mode)
        os.chdir("..")
    os.chdir(start_dir)
