"""Preps the AAV dataset for evaluation."""
import os
import pandas as pd
import esm
from xGPR.static_layers.fast_conv import FastConv1d
from .build_esm_reps import generate_esm_embeddings, save_conv_reps
from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit


def prep_aav(start_dir):
    """Prep all AAV data for multiple evaluation types, including
    convolution & fixed vector kernels, for all splits."""
    os.chdir(os.path.join("benchmark_evals", "aav_eval", "raw_data"))
    if "full_data_.csv" not in os.listdir():
        raise ValueError("The AAV dataset has not been downloaded yet!")

    esm_model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    esm_model.cuda()

    statlayer = FastConv1d(seq_width = 1280, device = "gpu",
                        conv_width = [9], num_features = 5000,
                        mode = "maxpool")

    raw_data = sequence_cleanup()
    splits = ["seven_vs_many_split",
            "mut_des_split", "des_mut_split",
            "two_vs_many_split", "one_vs_many_split"]
    target_dirs = ["onehot_conv", "esm_reps"]

    for i, enc_type in enumerate(target_dirs):
        for split in splits:
            os.chdir(start_dir)
            os.chdir(os.path.join("benchmark_evals", "aav_eval"))
            print(f"Now encoding split {split} using {enc_type}")
            if target_dirs[i] not in os.listdir():
                os.mkdir(target_dirs[i])
            if "esm" not in enc_type:
                generate_split_arrays(start_dir, target_dirs[i],
                            split, enc_type, raw_data)
            else:
                generate_esm_arrays(start_dir, target_dirs[i], split, raw_data,
                        esm_model, alphabet, batch_converter,
                        statlayer)
    print("AAV encoding is complete.")


def generate_esm_arrays(start_dir, dest_dir, split, raw_data,
        esm_model, alphabet, batch_converter, statlayer):
    """Generate ESM embeddings for a specific split."""
    os.chdir(dest_dir)
    subset = raw_data[raw_data[split].notna()]
    is_valid_colname = split + "_validation"
    subset = subset[subset["clean_seqs"]!="_PROBLEM_SEQUENCE_"]
    train = subset[subset[split]=="train"]

    validation = train[train[is_valid_colname].notna()]
    train = train[train[is_valid_colname].isna()]
    test = subset[subset[split]=="test"]
    input_dfs = [train, validation, test]
    dfnames = ["train", "valid", "test"]

    if split not in os.listdir():
        os.mkdir(split)

    conv_split = f"conv_{split}"
    if conv_split not in os.listdir():
        os.mkdir(conv_split)

    rep_dir = os.getcwd()

    for (dframe, dfname) in zip(input_dfs, dfnames):
        os.chdir(rep_dir)
        os.chdir(split)
        raw_seqs = dframe["clean_seqs"].tolist()
        yvals = dframe["score"].values
        if dfname not in os.listdir():
            os.mkdir(dfname)

        conv_reps = generate_esm_embeddings(dfname, raw_seqs, yvals, esm_model, alphabet,
                batch_converter, statlayer, 50)

        os.chdir(os.path.join(rep_dir, conv_split))
        if dfname not in os.listdir():
            os.mkdir(dfname)
        save_conv_reps(dfname, conv_reps, yvals)
    os.chdir(start_dir)



def generate_split_arrays(start_dir, dest_dir, split,
                            enc_type, raw_data):
    """Generates the zero-padded encodings for convolution
    for a given train-test split."""
    encoder = MSAEncodingToolkit(enc_type.split("_conv")[0])

    os.chdir(dest_dir)
    subset = raw_data[raw_data[split].notna()]
    is_valid_colname = split + "_validation"
    subset = subset[subset["clean_seqs"]!="_PROBLEM_SEQUENCE_"]
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
        raw_seqs = dframe["clean_seqs"].tolist()
        yvals = dframe["score"].values
        if dfname not in os.listdir():
            os.mkdir(dfname)
        #57 is the largest length of an AAV
        #sequence in this dataset, so we zero-pad everything to
        #this.
        mode = "conv"
        if "conv" not in enc_type:
            mode = "flat"
        encoder.encode_sequence_list(raw_seqs, yvals, dfname,
                    blocksize=2000, mode=mode, fixed_len = 57)
        os.chdir("..")
    os.chdir(start_dir)


def sequence_cleanup():
    """Cleans up the input data so it is ready to encode."""
    raw_data = pd.read_csv("full_data_.csv")
    raw_data.drop("full_aa_sequence", axis=1, inplace=True)
    wild_type = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"

    corrected_mutregs = []
    num_duds = 0
    #There are a few (29) weird sequences where
    #a deletion is indicated but the sequence length
    #matches the wild type, which is weird and suggests
    #some error. For now, excluding these.
    for mutreg in raw_data["mutated_region"].tolist():
        if len(mutreg) == 29 and "*" in mutreg and \
                mutreg.replace("*", "") == wild_type:
            num_duds += 1
            corrected_mutregs.append("_PROBLEM")
            continue
        clean_mutreg = mutreg.replace("*", "")
        clean_mutreg = clean_mutreg.upper().replace("-", "")
        corrected_mutregs.append(clean_mutreg)

    print(f"{num_duds} duds")
    raw_data["clean_seqs"] = corrected_mutregs
    raw_data = raw_data[raw_data["clean_seqs"]!="_PROBLEM"].copy()
    return raw_data
