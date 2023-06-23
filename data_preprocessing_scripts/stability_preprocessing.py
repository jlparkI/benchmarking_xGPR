"""Preps the stability dataset."""
import os
import json

import numpy as np
import esm
from xGPR.static_layers.fast_conv import FastConv1d

from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit
from .build_esm_reps import generate_esm_embeddings, save_conv_reps


def prep_stability(start_dir):
    """Preps the stability dataset for evaluation with a convolution kernel."""
    os.chdir(os.path.join(start_dir, "benchmark_evals", "stability", "raw_data"))

    esm_model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    esm_model.cuda()

    statlayer = FastConv1d(seq_width = 1280, device = "gpu",
                        conv_width = [9], num_features = 5000,
                        mode = "maxpool")

    if "stability_valid.json" not in os.listdir():
        raise ValueError("The stability dataset has not been downloaded yet!")
    with open("stability_train.json", "r") as fhandle:
        train = json.load(fhandle)
    with open("stability_valid.json", "r") as fhandle:
        train += json.load(fhandle)
    with open("stability_test.json", "r") as fhandle:
        test = json.load(fhandle)

    target_dirs = ["onehot_conv"]

    for enc_type in target_dirs:
        os.chdir(os.path.join(start_dir, "benchmark_evals",
                        "stability"))
        if enc_type not in os.listdir():
            os.mkdir(enc_type)
        os.chdir(enc_type)
        if "standard" not in os.listdir():
            os.mkdir("standard")
        os.chdir("standard")
        dpath = os.getcwd()
        for json_data, data_type in zip([train, test], ["train", "test"]):
            os.chdir(dpath)
            generate_arrays(data_type, json_data, enc_type)

    print("Now working on embeddings...this might take a minute.")

    os.chdir(os.path.join(start_dir, "benchmark_evals",
                        "stability"))
    if "esm_reps" not in os.listdir():
        os.mkdir("esm_reps")
    os.chdir("esm_reps")
    if "conv" not in os.listdir():
        os.mkdir("conv")
    if "standard" not in os.listdir():
        os.mkdir("standard")
    os.chdir("standard")

    for json_data, data_type in zip([train, test], ["train", "test"]):
        os.chdir(os.path.join(start_dir, "benchmark_evals",
                        "stability", "esm_reps", "standard"))
        seqs = [s["primary"] for s in json_data]
        yvals = np.asarray([s["stability_score"] for s in json_data]).flatten()
        conv_reps = generate_esm_embeddings(data_type, seqs, yvals, esm_model, alphabet,
                batch_converter, statlayer)
        os.chdir(os.path.join(start_dir, "benchmark_evals",
                        "stability", "esm_reps", "conv"))
        save_conv_reps(data_type, conv_reps, yvals)

    print("Stability encoding is complete.")


def generate_arrays(dest_dir, raw_data, enc_type):
    """Encodes the data as onehot arrays for use by the
    convolution kernels."""
    encoder = MSAEncodingToolkit(enc_type.split("_conv")[0])

    if dest_dir not in os.listdir():
        os.mkdir(dest_dir)
    seqs = [s["primary"] for s in raw_data]
    yvals = np.asarray([s["stability_score"] for s in raw_data]).flatten()
    #We zero pad to 50, the longest sequence present.
    encoder.encode_sequence_list(seqs, yvals, dest_dir,
                    blocksize=2000, mode="conv", fixed_len = 50)
