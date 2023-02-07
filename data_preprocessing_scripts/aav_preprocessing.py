"""Preps the AAV dataset for evaluation."""
import os
import pandas as pd
from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit


def prep_aav(start_dir):
    """Prep all AAV data for multiple evaluation types, including
    convolution & fixed vector kernels, for all splits."""
    os.chdir(os.path.join("benchmark_evals", "aav_eval", "raw_data"))
    if "full_data_.csv" not in os.listdir():
        raise ValueError("The AAV dataset has not been downloaded yet!")

    raw_data = generate_msa()
    splits = ["seven_vs_many_split",
            "mut_des_split", "des_mut_split",
            "two_vs_many_split", "one_vs_many_split"]
    target_dirs = ["onehot", "onehot_conv"]

    for i, enc_type in enumerate(target_dirs):
        for split in splits:
            os.chdir(start_dir)
            os.chdir(os.path.join("benchmark_evals", "aav_eval"))
            print(f"Now encoding split {split} using {enc_type}")
            if target_dirs[i] not in os.listdir():
                os.mkdir(target_dirs[i])
            generate_split_arrays(start_dir, target_dirs[i],
                            split, enc_type, raw_data)
    print("AAV encoding is complete.")


def generate_split_arrays(start_dir, dest_dir, split,
                            enc_type, raw_data):
    """Generates the zero-padded encodings for convolution
    for a given train-test split."""
    encoder = MSAEncodingToolkit(enc_type.split("_conv")[0])

    os.chdir(dest_dir)
    subset = raw_data[raw_data[split].notna()]
    is_valid_colname = split + "_validation"
    subset = subset[subset["aligned_seqs"]!="_PROBLEM_SEQUENCE_"]
    train = subset[subset[split]=="train"]

    validation = train[train[is_valid_colname].notna()]
    train = train[train[is_valid_colname].isna()]
    test = subset[subset[split]=="test"]
    input_dfs = [train, validation, test]
    dfnames = ["train", "valid", "test"]
    if split not in os.listdir():
        os.mkdir(split)
    os.chdir(split)

    for (dframe, dfname) in zip(input_dfs, dfnames):
        raw_seqs = dframe["aligned_seqs"].tolist()
        raw_seqs = [s.replace("-", "") for s in raw_seqs]
        yvals = dframe["score"].values
        if dfname not in os.listdir():
            os.mkdir(dfname)
        #57 is the largest length of an AAV
        #sequence in this dataset, so we zero-pad everything to
        #this.
        mode = "conv"
        if enc_type == "onehot":
            mode = "flat"
        encoder.encode_sequence_list(raw_seqs, yvals, dfname,
                    blocksize=2000, mode=mode, fixed_len = 57)
        os.chdir("..")
    os.chdir(start_dir)


def generate_msa():
    """The AAV data CAN be converted to a multiple sequence alignment.
    Although we don't really use this, because we prefer convolution
    kernels in any case where generating an MSA is not straightforward,
    this function is used to generate an MSA (which we did not
    in fact end up using)."""
    if "full_data_alignments_added.csv" in os.listdir():
        raw_data = pd.read_csv("full_data_alignments_added.csv")
        return raw_data
    print("Now converting raw AAV data to an MSA.")
    raw_data = pd.read_csv("full_data_.csv")
    mutregs = raw_data["mutated_region"].tolist()
    raw_data.drop("full_aa_sequence", axis=1, inplace=True)
    wild_type = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"

    prob_seqs, prob_idx = [], set()
    for i, mutreg in enumerate(mutregs):
        if len(mutreg) == 29 and "*" in mutreg and \
                mutreg.replace("*", "") == wild_type:
            mutregs[i] = mutreg.replace("*", "")
            prob_seqs.append(mutregs[i])
            prob_idx.add(i)


    corrected_mutregs = []
    for i, mutreg in enumerate(mutregs):
        pos_counter, prev_insert = 0, False
        corrected_mutreg = ["-" for i in range(57)]
        if i in prob_idx:
            corrected_mutregs.append("_PROBLEM_SEQUENCE_")
            continue
        for letter in mutreg:
            if letter.islower():
                prev_insert = True
            elif not prev_insert:
                pos_counter += 1
            if not letter.islower():
                prev_insert = False
            if letter != "*":
                corrected_mutreg[pos_counter] = letter.upper()
            pos_counter += 1
        corrected_mutregs.append("".join(corrected_mutreg))

    raw_data["aligned_seqs"] = corrected_mutregs
    raw_data.to_csv("full_data_alignments_added.csv", index=False)
    return raw_data
