"""Convenience functions that are sometimes useful when working
with proteins. Most end users will likely write their own, but
these are provided in case helpful."""
import os
import sys

import numpy as np
from Bio import SeqIO

#The wildcard import here is normally undesirable, but
#in this case, pfasum_matrices contains only a short
#list of classes specified by the generate pssm loadfiles
#script.
from .pfasum_matrices import *


def get_offline_fitting_filelist(target_directory):
    """Gets a list of filenames ending with "xvalues" and
    'yvalues' in a target directory. Used for benchmarking,
    unlikely to be useful to most end users."""
    start_dir = os.getcwd()
    try:
        os.chdir(target_directory)
    except:
        raise Exception("Target directory for offline fitting not found.")
    xlist = [os.path.abspath(f) for f in os.listdir() if f.endswith("xvalues.npy")]
    ylist = [os.path.abspath(f) for f in os.listdir() if f.endswith("yvalues.npy")]
    if len(ylist) != len(xlist):
        raise ValueError("The number of files ending with 'yvalues.npy' "
                    "does not match the number of files ending with "
                    "'xvalues.npy' in the target directory.")
    xlist.sort()
    ylist.sort()
    os.chdir(start_dir)
    return xlist, ylist




def _str_to_class(classname):
    """Converts a string to a classname. Useful for converting inputs
    to MSAEncodingToolkit into class names for loading an appropriate
    PSSM."""
    pfasum_module = [m for m in sys.modules if "pfasum_matrices" in m]
    return getattr(sys.modules[pfasum_module[0]], classname)



class MSAEncodingToolkit():
    """Encodes input sequences either using one-hot encoding
    or a PSSM. Unlikely to be useful to end users, primarily
    for running the benchmark tests.

    Attributes:
        encoding_type (str): Either "onehot" or the name of
            a PSSM provided with this module.
        aas (dict): A dict of the possible aas and the
            blank or '-' character mapping each to an integer.
            Used to determine which aa maps to which
            encoding vector.
        reps (np.ndarray): A matrix where each row is a
            vector representation of the corresponding
            amino acid.
    """

    def __init__(self, encoding_type):
        if encoding_type != "onehot":
            encoder = _str_to_class(encoding_type)
            aa_list = encoder.get_aas()
            self.reps = encoder.get_mat().astype(np.float32)
        else:
            aa_list = ["A", "C", "D", "E", "F", "G", "H", "I",
                    "K", "L", "M", "N", "P", "Q", "R", "S", "T",
                    "V", "W", "Y", "-"]
            mat = []
            for i in range(21):
                mat.append(np.zeros((21)))
                mat[-1][i] = 1.0
            self.reps = np.stack(mat).astype(np.uint8)
        self.aas = {}
        for i, amino_acid in enumerate(aa_list):
            self.aas[amino_acid] = i

        self.encoding_type = encoding_type

    def run_input_checks(self, blocksize, output_dir):
        """Checks the input to ensure it is valid and that
        the target directory provided is valid."""
        if blocksize is not None:
            if blocksize < 100:
                raise ValueError("You have chosen a very small value for "
                        "blocksize. Please set blocksize to None or an "
                        "integer >= 100.")
        try:
            os.chdir(output_dir)
        except:
            raise ValueError("Invalid output directory supplied!")


    def encode_sequence_list(self, raw_seqs, yvalues,
            output_dir, blocksize = None, verbose=True, mode = "flat",
            fixed_len = None):
        """Encodes a list of sequences and an associated list
        of y-values, storing the results as numpy arrays in .npy
        files, each to contain 'blocksize' sequences.

        Args:
            raw_seqs (list): A list of strings. All must be the same
                length UNLESS fixed_len is not None.
            yvalues (list): A list of y-values, one per sequence. Must
                be the same length as raw_seqs.
            output_dir (str): A valid filepath where the output will
                be saved.
            blocksize: Either None or an int. If None, all sequences
                will be saved to a single npy file -- only recommended
                for very small datasets. For anything larger, a blocksize
                should be set. 2000 is a good default.
            verbose (bool): If True, regular updates are printed.
            mode (str): Either 'flat' or '3d'. If 'flat', encoded
                sequences are flattened to a 1d array -- this is used
                if the target kernel is a 'basic kernel', e.g. RBF,
                ERF-NN, Matern. If '3d', the sequences are kept as
                2d arrays of shape (L, A) for L amino acids and A
                features per amino acid. This is used for convolution
                kernels and convolution-based feature extraction.
            fixed_len: Either None or an integer. If None, all sequences
                must be the same length (i.e. from a multiple sequence
                alignment). If an int, sequences can be different lengths
                but must all be <= fixed_len. All will be zero padded
                to equal fixed_len. This is useful if doing convolution
                on sequences that cannot be aligned.
        """
        self.run_input_checks(blocksize, output_dir)
        blocknum, ycounter = 0, 0
        encoded_data = []
        seqlen = len(raw_seqs[0])
        for seqstring in raw_seqs:
            if len(seqstring) != seqlen:
                if fixed_len is None:
                    raise ValueError("Not all of the sequences in the specified "
                            "fasta file are of the same length. The MSA encoder can "
                            "only accept an MSA as input "
                            "(sequences aligned to all be same length).")
            encoded_seq = self.encode_seq(seqstring, mode, fixed_len)
            encoded_data.append(encoded_seq)
            if blocksize is not None:
                if len(encoded_data) >= blocksize:
                    self.save_data(encoded_data, yvalues,
                            ycounter, blocksize, blocknum)
                    encoded_data = []
                    blocknum += 1
                    ycounter += blocksize
                    if verbose:
                        print(f"Blocknum {blocknum} complete.")

        self.save_data(encoded_data, yvalues,
                        ycounter, blocksize, blocknum)
        if verbose:
            print("All files generated.")


    def encode_fasta_file(self, filepath, yvalues, output_dir,
                        blocksize = None, verbose = True, mode = "flat",
                        fixed_len = None):
        """Encodes a fasta file of protein seqs and an associated list
        of y-values, storing the results as numpy arrays in .npy
        files, each to contain 'blocksize' sequences.

        Args:
            filepath (str): A filepath to a fasta file.
            yvalues (list): A list of y-values, one per sequence. Must
                be the same length as the number of sequences in
                the fasta file.
            output_dir (str): A valid filepath where the output will
                be saved.
            blocksize: Either None or an int. If None, all sequences
                will be saved to a single npy file -- only recommended
                for very small datasets. For anything larger, a blocksize
                should be set. 2000 is a good default.
            verbose (bool): If True, regular updates are printed.
            mode (str): Either 'flat' or '3d'. If 'flat', encoded
                sequences are flattened to a 1d array -- this is used
                if the target kernel is a 'basic kernel', e.g. RBF,
                ERF-NN, Matern. If '3d', the sequences are kept as
                2d arrays of shape (L, A) for L amino acids and A
                features per amino acid. This is used for convolution
                kernels and convolution-based feature extraction.
            fixed_len: Either None or an integer. If None, all sequences
                must be the same length (i.e. from a multiple sequence
                alignment). If an int, sequences can be different lengths
                but must all be <= fixed_len. All will be zero padded
                to equal fixed_len. This is useful if doing convolution
                on sequences that cannot be aligned.
        """
        self.run_input_checks(blocksize, output_dir)
        try:
            filehandle = open(filepath, "r")
            for seqrec in SeqIO.parse(filehandle, "fasta"):
                seqlen = len(str(seqrec.seq))
                break
            filehandle.close()
        except:
            raise ValueError("Invalid filepath supplied "
                    "for the fasta file!")
        self.run_input_checks(blocksize, output_dir)
        blocknum, ycounter = 0, 0
        encoded_data = []
        with open(filepath, "r") as filehandle:
            for seqrec in SeqIO.parse(filehandle, "fasta"):
                seqstring = str(seqrec.seq)
                if len(seqstring) != seqlen:
                    if fixed_len is None:
                        raise ValueError("Not all of the sequences in the specified "
                            "fasta file are of the same length. The MSA encoder can "
                            "only accept an MSA as input "
                            "(sequences aligned to all be same length).")
                encoded_seq = self.encode_seq(seqstring, mode, fixed_len)
                encoded_data.append(encoded_seq)
                if blocksize is not None:
                    if len(encoded_data) >= blocksize:
                        self.save_data(encoded_data, yvalues,
                            ycounter, blocksize, blocknum)
                        encoded_data = []
                        blocknum += 1
                        ycounter += blocksize
                        if verbose:
                            print(f"Blocknum {blocknum} complete.")

            self.save_data(encoded_data, yvalues,
                        ycounter, blocksize, blocknum)
        if verbose:
            print("All files generated.")



    def encode_fasta_file_with_msa(self, filepath, yvalues, output_dir, msa,
                        pos_weights, blocksize = None, verbose = True):
        """Encodes a fasta file of protein seqs and an associated list
        of y-values, storing the results as numpy arrays in .npy
        files, each to contain 'blocksize' sequences, using information
        from an MSA. Since an MSA is used, the sequences must all be
        of the same length.

        Args:
            filepath (str): A filepath to a fasta file.
            yvalues (list): A list of y-values, one per sequence. Must
                be the same length as the number of sequences in
                the fasta file.
            output_dir (str): A valid filepath where the output will
                be saved.
            msa (np.ndarray): A (num_positions, num_possible_aas) numpy
                array storing the probability of each aa at each position.
            max_entropy (float): The maximum entropy expected. The default
                value is for 21 possible amino acids, change this if
                there is a different number.
            blocksize: Either None or an int. If None, all sequences
                will be saved to a single npy file -- only recommended
                for very small datasets. For anything larger, a blocksize
                should be set. 2000 is a good default.
            verbose (bool): If True, regular updates are printed.
        """
        self.run_input_checks(blocksize, output_dir)
        try:
            filehandle = open(filepath, "r")
            filehandle.close()
        except:
            raise ValueError("Invalid filepath supplied "
                    "for the fasta file!")
        self.run_input_checks(blocksize, output_dir)
        blocknum, ycounter = 0, 0
        encoded_data = []
        print(msa.shape)

        with open(filepath, "r") as filehandle:
            for seqrec in SeqIO.parse(filehandle, "fasta"):
                seqstring = str(seqrec.seq)
                if len(seqstring) != msa.shape[0]:
                    import pdb
                    pdb.set_trace()
                    raise ValueError("Not all of the sequences in the specified "
                            "fasta file are of the same length. The MSA encoder can "
                            "only accept an MSA as input "
                            "(sequences aligned to all be same length).")
                encoded_seq = self.encode_seq_with_msa(seqstring, msa, pos_weights)
                encoded_data.append(encoded_seq)
                if blocksize is not None:
                    if len(encoded_data) >= blocksize:
                        self.save_data(encoded_data, yvalues,
                            ycounter, blocksize, blocknum)
                        encoded_data = []
                        blocknum += 1
                        ycounter += blocksize
                        if verbose:
                            print(f"Blocknum {blocknum} complete.")

            self.save_data(encoded_data, yvalues,
                        ycounter, blocksize, blocknum)
        if verbose:
            print("All files generated.")




    def save_data(self, encoded_data, yvalues, ycounter,
            blocksize, blocknum):
        """Saves a block of data.

        Args:
            encoded_data (np.ndarray): A numpy array of encoded
                sequences.
            yvalues (array-like): The yvalues for the dataset.
            ycounter (int): yvalues from ycounter: ycounter + blocksize
                will be saved in this block.
            blocksize (int): yvalues from ycounter: ycounter + blocksize
                will be saved in this block.
            blocksize (int): The size of the block to save. If blocksize
                is None, all of yvalues will be saved in a single block.
            blocknum (int): A number to assign to the filenames when
                saving.
        """
        encoded_data = np.vstack(encoded_data)
        if self.encoding_type == "onehot":
            np.save(f"{blocknum}_block_xvalues.npy",
                    encoded_data.astype(np.uint8), allow_pickle=True)
        else:
            np.save(f"{blocknum}_block_xvalues.npy",
                    encoded_data.astype(np.float32), allow_pickle=True)
        if blocksize is None:
            yblock = yvalues
        else:
            endpt = min([yvalues.shape[0], ycounter + blocksize])
            if len(yvalues.shape) > 1:
                yblock = yvalues[ycounter:endpt,:]
            else:
                yblock = yvalues[ycounter:endpt]
        np.save(f"{blocknum}_block_yvalues.npy", yblock, allow_pickle=True)
        if yblock.shape[0] != encoded_data.shape[0]:
            raise ValueError("Different numbers of sequences and y-values "
                    "were passed.")


    def encode_seq(self, seqstring, mode="flat", fixed_len = None):
        """Encodes an input sequence.

        Args:
            seqstring (str): A protein sequence as a string.
            mode (str): If 'flat', the encoded sequence is flattened
                to a 1d array. Used for fully connected networks
                and RBF kernels, Matern kernels. If '3d', the original
                2d shape is kept. Used for CNNs, convolutional kernels,
                feature extractors.
            fixed_len (int): If None, sequences must all be the same
                length. If an int, sequences will be zero-padded
                to be the same length. fixed_len must be > the longest
                sequence in the dataset.
        """
        if fixed_len is None:
            encoded_data = np.zeros((1, len(seqstring), self.reps.shape[1]))
        else:
            encoded_data = np.zeros((1, fixed_len, self.reps.shape[1]))
        try:
            for i, letter in enumerate(seqstring):
                idx = self.aas[letter]
                encoded_data[0,i,:] = self.reps[idx,:]
        except:
            raise ValueError(f"The sequence {seqstring} contains a nonstandard amino acid "
                    "character and could not be encoded properly.")
        if mode == "flat":
            encoded_data =encoded_data.reshape((1, encoded_data.shape[1] *
                encoded_data.shape[2]))
        return encoded_data


    def encode_seq_with_msa(self, seqstring, msa, pos_weights):
        """Encodes an input sequence using an msa supplied by caller.

        Args:
            seqstring (str): A protein sequence as a string.
            msa (np.ndarray): A (num_positions, num_possible_aas) numpy
                array storing the probability of each aa at each position.
            pos_weights (np.ndarray): A (num_positions) array indicating the relative
                weighting for each position.
        """
        encoded_data = np.zeros((1, len(seqstring) * self.reps.shape[1]))
        start_position = len(seqstring) * self.reps.shape[1]
        try:
            for i, letter in enumerate(seqstring):
                idx = self.aas[letter]
                #encoded_data[0,i*self.reps.shape[1]:(i+1)*self.reps.shape[1]] = \
                #    self.reps[idx,:] * pos_weights[i]
                encoded_data[0,i*self.reps.shape[1]:(i+1)*self.reps.shape[1]] = msa[i,idx,:]
        except:
            raise ValueError(f"The sequence {seqstring} contains a nonstandard amino acid "
                    "character and could not be encoded properly.")
        return encoded_data
