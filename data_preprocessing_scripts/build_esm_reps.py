"""Contains code used across preprocessing scripts to generate
ESM embeddings and FastConv-1d representations of ESM embeddings."""
import os
import numpy as np
import torch



def generate_esm_embeddings(dest_dir, seqs, yvals, esm_model, alphabet,
        batch_converter, statlayer, batch_size = 50, large = False):
    """Encodes the data using esm-based embeddings."""

    if dest_dir not in os.listdir():
        os.mkdir(dest_dir)
    zipped_data = [(y,x) for (x,y) in zip(seqs, yvals)]
    yvals = np.array(yvals)

    if not large:
        tokens, batch_lens = tokenize(zipped_data, batch_converter,
                alphabet)
        reps, conv_reps = genrep(tokens, batch_lens, esm_model, batch_size, statlayer)
    else:
        reps, conv_reps = [], []
        for i in range(0, len(zipped_data), 10):
            tokens, batch_lens = tokenize(zipped_data[i:i+10], batch_converter,
                alphabet)
            if max(batch_lens) < 350:
                rep_batch, conv_rep_batch = genrep(tokens, batch_lens, esm_model, 5, statlayer,
                        print_updates = False)
            else:
                rep_batch, conv_rep_batch = genrep(tokens, batch_lens, esm_model, 1, statlayer,
                        print_updates = False)

            if i % 1000 == 0:
                print(f"{i} complete.")

            reps.append(rep_batch)
            conv_reps.append(conv_rep_batch)

        reps = np.vstack(reps)
        conv_reps = np.vstack(conv_reps)

    os.chdir(dest_dir)

    for i, cnum in enumerate(range(0, reps.shape[0], 2000)):
        np.save(f"{i}_block_xvalues.npy", reps[cnum:cnum+2000,:].astype(np.float32))
        np.save(f"{i}_block_yvalues.npy", yvals[cnum:cnum+2000])
    os.chdir("..")
    return conv_reps



def tokenize(zipped_data, batch_converter, alphabet):
    """Converts input data into tokens for use by the ESM model."""
    _, _, tokens = batch_converter(zipped_data)
    batch_lens = (tokens != alphabet.padding_idx).sum(1)
    return tokens, batch_lens


def genrep(batch_tokens, batch_len, model, batch_size, statlayer, print_updates = True):
    """Generates the ESM-1v representations for a batch of input data.
    Also applies the FastConv-1d kernel input generator."""
    seqreps = []
    fastconv_1d_seqreps = []
    with torch.no_grad():
        for j in range(0, batch_tokens.shape[0], batch_size):
            results = model(batch_tokens[j:j+batch_size].cuda(), repr_layers=[33],
                    return_contacts=True)
            token_representations = results["representations"][33].cpu()

            sequence_representations = []
            for i, tokens_len in enumerate(batch_len[j:j+batch_size]):
                sequence_representations.append(token_representations[i,
                    1 : tokens_len - 1].mean(0))
                np_token_rep = token_representations[i:i+1,1:tokens_len - 1].numpy()
                fastconv_1d_seqreps.append(statlayer.conv1d_x_feat_extract(np_token_rep))

            seqreps += sequence_representations
            if j % 1000 == 0 and print_updates:
                print(f"ESM embeddings: {j} complete")

    seqreps = np.vstack([t.numpy() for t in seqreps])
    fastconv_1d_seqreps = np.vstack(fastconv_1d_seqreps)
    return seqreps, fastconv_1d_seqreps


def save_conv_reps(dest_dir, conv_reps, yvals):
    """Saves the convolution reps to a separate location."""
    if dest_dir not in os.listdir():
        os.mkdir(dest_dir)
    os.chdir(dest_dir)
    for i, cnum in enumerate(range(0, conv_reps.shape[0], 2000)):
        np.save(f"{i}_block_xvalues.npy", conv_reps[cnum:cnum+2000,:].astype(np.float32))
        np.save(f"{i}_block_yvalues.npy", yvals[cnum:cnum+2000])
    os.chdir("..")
