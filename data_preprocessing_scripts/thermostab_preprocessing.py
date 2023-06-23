"""Preps the thermostability dataset for evaluation."""
import os
import numpy as np
import pandas as pd
import esm
from xGPR.static_layers.fast_conv import FastConv1d
from .build_esm_reps import generate_esm_embeddings, save_conv_reps
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

    esm_model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    esm_model.cuda()

    statlayer = FastConv1d(seq_width = 1280, device = "gpu",
                        conv_width = [9], num_features = 5000,
                        mode = "maxpool")

    target_dirs = ["onehot_conv", "esm_reps"]

    for enc_type in target_dirs:
        os.chdir(start_dir)
        os.chdir(os.path.join("benchmark_evals", "thermostability"))
        if enc_type not in os.listdir():
            os.mkdir(enc_type)
        os.chdir(enc_type)
        dpath = os.getcwd()

        for dataset, split in zip(raw_data, splits):
            os.chdir(dpath)
            print(f"Now encoding split {split}")
            if "esm" not in enc_type:
                generate_split_arrays(start_dir, split, enc_type, dataset)
            else:
                generate_esm_arrays(start_dir, split, dataset,
                        esm_model, alphabet, batch_converter,
                        statlayer)
    print("Thermostab encoding is complete.")


def generate_esm_arrays(start_dir, split, raw_data,
        esm_model, alphabet, batch_converter, statlayer):
    """Generate ESM embeddings for a specific split."""

    subset = raw_data[raw_data["set"].notna()]
    train = subset[subset["set"]=="train"]

    test = subset[subset["set"]=="test"]
    input_dfs = [train, test]
    dfnames = ["train", "test"]
    if split not in os.listdir():
        os.mkdir(split)

    conv_split = f"conv_{split}"
    if conv_split not in os.listdir():
        os.mkdir(conv_split)

    rep_dir = os.getcwd()

    for (input_df, dfname) in zip(input_dfs, dfnames):
        os.chdir(rep_dir)
        os.chdir(split)
        raw_seqs = input_df["sequence"].tolist()
        yvals = input_df["target"].tolist()
        if dfname not in os.listdir():
            os.mkdir(dfname)

        #ESM can ONLY embed sequences up to 1024 in length; therefore we have to
        #discard all which are longer than this. In fact, it yields errors processing
        #sequences of length 1023! See FAIR-ESM documentation.
        zipped_values = [(yvals[i], raw_seqs[i]) for i in range(len(yvals)) if len(raw_seqs[i]) < 1020]
        zipped_values.sort(key=lambda x: len(x[1]))
        yvals = np.array([s[0] for s in zipped_values])
        filtered_seqs = [s[1] for s in zipped_values]
        conv_reps = generate_esm_embeddings(dfname, filtered_seqs, yvals, esm_model, alphabet,
                batch_converter, statlayer, 1, large = True)

        os.chdir(os.path.join(rep_dir, conv_split))
        if dfname not in os.listdir():
            os.mkdir(dfname)
        save_conv_reps(dfname, conv_reps, yvals)
    os.chdir(start_dir)



def generate_split_arrays(start_dir, split,
                            enc_type, raw_data):
    """Generate arrays with encodings for a specific train-test
    split."""
    encoder = MSAEncodingToolkit(enc_type.split("_conv")[0])

    subset = raw_data[raw_data["set"].notna()]
    train = subset[subset["set"]=="train"]

    test = subset[subset["set"]=="test"]
    input_dfs = [train, test]
    dfnames = ["train", "test"]
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
