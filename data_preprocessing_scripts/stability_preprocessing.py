"""Preps the stability dataset."""
import os
import json

import numpy as np

from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit


def prep_stability(start_dir):
    """Preps the stability dataset for evaluation with a convolution kernel."""
    os.chdir(os.path.join(start_dir, "benchmark_evals", "stability", "raw_data"))

    if "stability_valid.json" not in os.listdir():
        raise ValueError("The stability dataset has not been downloaded yet!")
    with open("stability_train.json", "r") as fhandle:
        train = json.load(fhandle)
    with open("stability_valid.json", "r") as fhandle:
        train += json.load(fhandle)
    with open("stability_test.json", "r") as fhandle:
        test = json.load(fhandle)
    os.chdir(os.path.join(start_dir, "benchmark_evals", "stability"))

    if "onehot_conv" not in os.listdir():
        os.mkdir("onehot_conv")

    for json_data, data_type in zip([train, test], ["train", "test"]):
        os.chdir(start_dir)
        os.chdir(os.path.join("benchmark_evals", "stability", "onehot_conv"))
        if "standard" not in os.listdir():
            os.mkdir("standard")
        os.chdir("standard")
        generate_onehot_arrays(data_type, json_data)
    print("Stability encoding is complete.")


def generate_onehot_arrays(dest_dir, raw_data):
    """Encodes the data as onehot arrays for use by the
    convolution kernels."""
    encoder = MSAEncodingToolkit("onehot")

    if dest_dir not in os.listdir():
        os.mkdir(dest_dir)
    seqs = [s["primary"] for s in raw_data]
    yvals = np.asarray([s["stability_score"] for s in raw_data]).flatten()
    #We zero pad to 50, the longest sequence present.
    encoder.encode_sequence_list(seqs, yvals, dest_dir,
                    blocksize=2000, mode="conv", fixed_len = 50)
