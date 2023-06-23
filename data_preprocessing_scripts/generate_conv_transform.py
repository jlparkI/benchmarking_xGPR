"""Provides the get_conv_transform function for preparing benchmark
datasets for use with a fixed-vector kernel by performing feature
extraction with a convolution kernel."""
import os
from xGPR.static_layers.fast_conv import FastConv1d
from xGPR.data_handling.dataset_builder import build_offline_sequence_dataset

def get_conv_transform(start_dir, target_dataset, conv_width = 9):
    """Uses convolution-based feature extraction to generate features for
    input to a Matern / RBF kernel."""
    print("Now working on convolutional feature extraction.")
    os.chdir(os.path.join(start_dir, "benchmark_evals",
                target_dataset))

    if "onehot_conv" not in os.listdir():
        print("Either dataset has not yet been preprocessed, "
                "or convolution not planned for this dataset (due to "
                "sequence length etc).")
        return

    if "conv_features" not in os.listdir():
        os.mkdir("conv_features")

    conv_raw_dirs = [f for f in os.listdir() if f.endswith("_conv")]

    for conv_raw_dir in conv_raw_dirs:
        print(f"Now working on {conv_raw_dir}")
        os.chdir(os.path.join(start_dir, "benchmark_evals", target_dataset,
                        conv_raw_dir))
        splits = [f for f in os.listdir()]
        for split in splits:
            for data_dir in ["train", "test", "valid"]:
                os.chdir(os.path.join(start_dir, "benchmark_evals", target_dataset,
                        conv_raw_dir, split))
                if data_dir not in os.listdir():
                    continue
                os.chdir(data_dir)
                xfiles = [f for f in os.listdir() if f.endswith("xvalues.npy")
                            and "CONV" not in f]
                yfiles = [f for f in os.listdir() if f.endswith("yvalues.npy")]
                xfiles.sort()
                yfiles.sort()
                dset = build_offline_sequence_dataset(xfiles, yfiles, chunk_size = 2000)
                x_shape = dset.get_xdim()
                extractor = FastConv1d(seq_width = x_shape[2], device = "gpu",
                        conv_width = [conv_width], num_features = 5000,
                        mode = "maxpool_loc")
                os.chdir(os.path.join(start_dir, "benchmark_evals",
                        target_dataset, "conv_features"))

                if conv_raw_dir not in os.listdir():
                    os.mkdir(conv_raw_dir)
                os.chdir(conv_raw_dir)
                if split not in os.listdir():
                    os.mkdir(split)
                os.chdir(split)
                if data_dir not in os.listdir():
                    os.mkdir(data_dir)
                _ = extractor.conv1d_pretrain_feat_extract(dset, data_dir)
    print("Convolutional feature extraction is complete.")
