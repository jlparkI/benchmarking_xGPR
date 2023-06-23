"""Preps the GB1 data for evaluation."""
import os
import pandas as pd
import numpy as np
import torch
import esm
from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit
from .build_esm_reps import tokenize

def prep_gb1(start_dir):
    """Preps the GB1 test data for modeling."""
    os.chdir(os.path.join("benchmark_evals", "gb1_eval", "raw_data"))
    if "four_mutations_full_data.csv" not in os.listdir():
        raise ValueError("The GB1 dataset has not been downloaded yet!")

    raw_data = pd.read_csv("four_mutations_full_data.csv")
    splits = ["two_vs_rest", "three_vs_rest"]
    target_dirs = ["onehot", "esm_reps"]

    esm_model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    esm_model.cuda()

    for i, enc_type in enumerate(target_dirs):
        for split in splits:
            os.chdir(start_dir)
            os.chdir(os.path.join("benchmark_evals", "gb1_eval"))
            print(f"Now encoding split {split} using {enc_type}")
            if target_dirs[i] not in os.listdir():
                os.mkdir(target_dirs[i])
            if "esm" in enc_type:
                os.chdir("esm_reps")
                generate_esm_arrays(split, raw_data,
                    esm_model, alphabet, batch_converter)
            else:
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


def generate_esm_arrays(split, raw_data,
        esm_model, alphabet, batch_converter):
    """Generate ESM embeddings for a specific split."""

    train = raw_data[raw_data[split]=="train"].copy()

    test = raw_data[raw_data[split]=="test"].copy()
    input_dfs = [train, test]
    dfnames = ["train", "test"]
    if split not in os.listdir():
        os.mkdir(split)

    rep_dir = os.getcwd()

    for (input_df, dfname) in zip(input_dfs, dfnames):
        os.chdir(rep_dir)
        os.chdir(split)
        #Cut off early in the sequence to avoid a long sequence that
        #will incur too much memory with ESM. The variable regions are all
        #in the first part of the sequence.
        raw_seqs = [s[:55] for s in input_df["sequence"].tolist()]
        yvals = input_df["Fitness"].values
        if dfname not in os.listdir():
            os.mkdir(dfname)

        _ = gb1_esm_embeddings(dfname, raw_seqs, yvals, esm_model, alphabet,
                batch_converter, statlayer = None, batch_size = 25)


def gb1_esm_embeddings(dest_dir, seqs, yvals, esm_model, alphabet,
        batch_converter, statlayer, batch_size = 50):
    """Encodes the data using esm-based embeddings with settings specific
    to GB1. For GB1, unlike the other benchmarks, we are modifying only
    4 amino acids, so we only want the embeddings for those 4."""

    if dest_dir not in os.listdir():
        os.mkdir(dest_dir)
    zipped_data = [(y,x) for (x,y) in zip(seqs, yvals)]
    yvals = np.array(yvals)

    tokens, batch_lens = tokenize(zipped_data, batch_converter,
                alphabet)
    reps = gb1_genrep(tokens, batch_lens, esm_model, batch_size, statlayer)
    os.chdir(dest_dir)

    for i, cnum in enumerate(range(0, reps.shape[0], 2000)):
        np.save(f"{i}_block_xvalues.npy", reps[cnum:cnum+2000,:].astype(np.float32))
        np.save(f"{i}_block_yvalues.npy", yvals[cnum:cnum+2000])
    os.chdir("..")

def gb1_genrep(batch_tokens, batch_len, model, batch_size, print_updates = True):
    """Generates the ESM-1v representations for a batch of input data, without using the
    FastConv-1d statlayer (for GB1) and extrcting only reps for 4 specific AAs."""
    seqreps = []
    #The four positions we need to extract.
    relevant_idx = torch.Tensor([38,39,40,53]).long() + 1
    with torch.no_grad():
        for j in range(0, batch_tokens.shape[0], batch_size):
            results = model(batch_tokens[j:j+batch_size].cuda(), repr_layers=[33],
                    return_contacts=True)
            token_representations = results["representations"][33].cpu()

            sequence_representations = []
            for i, tokens_len in enumerate(batch_len[j:j+batch_size]):
                sequence_representations.append(token_representations[i,
                    relevant_idx].reshape((1, 4 * 1280)))

            seqreps += sequence_representations
            if j % 1000 == 0 and print_updates:
                print(f"ESM embeddings: {j} complete")

    seqreps = np.vstack([t.numpy() for t in seqreps])
    return seqreps
