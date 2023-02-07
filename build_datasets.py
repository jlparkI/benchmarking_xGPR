"""Downloads datasets where needed and calls the appropriate functions for
prepping the data from data_preprocessing_scripts."""
import os
import shutil
import sys
import argparse
import tarfile
import zipfile
import subprocess

import wget
from data_preprocessing_scripts.aav_preprocessing import prep_aav
from data_preprocessing_scripts.thermostab_preprocessing import prep_thermostab
from data_preprocessing_scripts.fluorescence_preprocessing import prep_fluorescence
from data_preprocessing_scripts.gb1_preprocessing import prep_gb1
from data_preprocessing_scripts.uci_datasets_prep import prep_UCI_dataset
from data_preprocessing_scripts.stability_preprocessing import prep_stability
from data_preprocessing_scripts.generate_conv_transform import get_conv_transform
from data_preprocessing_scripts.tabular_data_preprocessing import prep_tabular_datasets

def gen_arg_parser():
    """Set up the argument parser for the command line input."""
    parser = argparse.ArgumentParser(description="Use this command line app to "
                "setup the experiments described in the paper -- to get "
                "data from repositories, move it to the correct locations "
                "and preprocess it.")
    parser.add_argument("--getraw", action="store_true", help=
            "Download the raw data from the appropriate locations.")
    parser.add_argument("--aav", action="store_true", help=
            "Preprocess the AAV dataset.")
    parser.add_argument("--thermostab", action="store_true", help=
            "Preprocess the thermostability dataset.")
    parser.add_argument("--stab", action="store_true", help=
            "Preprocess the stability dataset.")
    parser.add_argument("--fluor", action="store_true", help=
            "Preprocess the fluorescence dataset.")
    parser.add_argument("--gb1", action="store_true", help=
            "Preprocess the GB1 dataset.")
    parser.add_argument("--uci", action="store_true", help=
            "Preprocess the three UCI datasets for comparison with GPyTorch.")
    parser.add_argument("--molecules", action="store_true", help=
            "Preprocess the qm9 small molecules dataset.")
    parser.add_argument("--tabular", action="store_true", help=
            "Preprocess the additional tabular datasets for comparison with "
            "other algorithms.")
    return parser

def get_raw_data(start_dir):
    """Retrieve data from links from which it was originally downloaded (for
    datasets which are not included)."""
    qm9_yvals_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    qm9_xyz_url = "https://figshare.com/ndownloader/files/3195389"

    os.chdir(os.path.join("benchmark_evals", "aav_eval", "raw_data"))
    if "full_data_.csv" not in os.listdir():
        print("Now downloading the AAV dataset.")
        subprocess.run(["git", "clone", "https://github.com/J-SNACKKB/FLIP"])

        shutil.move(os.path.join("FLIP", "splits", "aav", "full_data.csv.zip"), "full_data.csv.zip")
        fname = "full_data.csv.zip"

        with zipfile.ZipFile(fname, "r") as zip_ref:
            zip_ref.extractall()

        os.remove("full_data.csv.zip")
        os.rename("full_data.csv", "full_data_.csv")
        shutil.rmtree("FLIP")
    print("AAV downloaded.")

    os.chdir(start_dir)
    os.chdir(os.path.join("benchmark_evals", "gb1_eval", "raw_data"))
    if "four_mutations_full_data.csv" not in os.listdir():
        print("Now downloading the GB1 dataset.")
        subprocess.run(["git", "clone", "https://github.com/J-SNACKKB/FLIP"])

        shutil.move(os.path.join("FLIP", "splits", "gb1", "four_mutations_full_data.csv.zip"),
                "four_mutations_full_data.csv.zip")
        fname = "four_mutations_full_data.csv.zip"

        with zipfile.ZipFile(fname, "r") as zip_ref:
            zip_ref.extractall()

        os.remove("four_mutations_full_data.csv.zip")
        shutil.rmtree("FLIP")
    print("GB1 downloaded.")

    os.chdir(start_dir)
    os.chdir("benchmark_evals")
    if "chemdata" not in os.listdir():
        os.mkdir("chemdata")
    os.chdir("chemdata")
    if "qm9_.csv" not in os.listdir():
        fname = wget.download(qm9_yvals_url)
        os.rename("qm9.csv", "qm9_.csv")
    if "qm9_mols" not in os.listdir():
        fname = wget.download(qm9_xyz_url)
        with tarfile.open(fname) as tarball:
            tarball.extractall("qm9_mols")
        os.remove(fname)
    print("QM9 downloaded.")

    os.chdir(start_dir)
    os.chdir(os.path.join("benchmark_evals", "song_dataset", "raw_data"))
    if "raw_data_.csv" not in os.listdir():
        fname = wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip")
        with zipfile.ZipFile(fname, "r") as zip_ref:
            zip_ref.extractall()
        os.rename("YearPredictionMSD.txt", "raw_data_.csv")
        os.remove(fname)
    print("SONG dataset downloaded.")

    os.chdir(start_dir)


def main():
    start_dir = os.path.dirname(os.path.abspath(__file__))
    parser = gen_arg_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.getraw:
        get_raw_data(start_dir)
    if args.aav:
        prep_aav(start_dir)
        get_conv_transform(start_dir, "aav_eval")
    if args.fluor:
        prep_fluorescence(start_dir)
        get_conv_transform(start_dir, "fluorescence_eval")
    if args.gb1:
        prep_gb1(start_dir)
        get_conv_transform(start_dir, "gb1_eval", 3)
    if args.thermostab:
        prep_thermostab(start_dir)
        get_conv_transform(start_dir, "thermostability")
    if args.stab:
        prep_stability(start_dir)
        get_conv_transform(start_dir, "stability")
    if args.molecules:
        os.chdir(os.path.join(start_dir, "chemtools"))
        benchmark_path = os.path.join(start_dir, "benchmark_evals")
        chemdata_path = os.path.join(benchmark_path, "chemdata")
        if "cleaned_qm9_mols" not in os.listdir():
            subprocess.run(["python", "molcleaner.py", chemdata_path], check = True)

        os.chdir(os.path.join(start_dir, "chemtools"))
        subprocess.run(["python", "soap_maker.py", chemdata_path,
                    start_dir], check = True)
        os.chdir(start_dir)
    if args.uci:
        for dataset in ["kin40k_dataset", "uci_protein_dataset",
                        "song_dataset"]:
            os.chdir(start_dir)
            prep_UCI_dataset(start_dir, dataset, "y_norm")
    if args.tabular:
        for dataset in ["song_dataset", "rossman", "sulfur"]:
            os.chdir(start_dir)
            #Song data is under a different filename for raw data.
            if "song" in dataset:
                prep_tabular_datasets(start_dir, dataset, "raw_data_.csv")
            else:
                prep_tabular_datasets(start_dir, dataset)


if __name__ == "__main__":
    main()
