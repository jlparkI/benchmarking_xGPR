"""Quick throwaway script to generate feature vectors using
the PFASUM matrices."""
import os
import numpy as np



def generate_pfasum_dotprod_file(percent_homology, pfasum_path):
    """Writes Python code that makes available a loadable set
    of feature vectors for a specified PFASUM matrix."""
    if "pfasum_matrices.py" not in os.listdir():
        fhandle = open("pfasum_matrices.py", "w+")
        fhandle.write("import numpy as np\n\n\n")
        fhandle.close()
    else:
        with open("pfasum_matrices.py") as fhandle:
            for line in fhandle:
                if line.endswith(f"PFASUM{percent_homology}_standardized:"):
                    print(f"The PFASUM{percent_homology} file has already been generated.")
                    return
    try:
        start_dir = os.getcwd()
        os.chdir(pfasum_path)
        with open(f"PFASUM{percent_homology}.mat", "r") as fhandle:
            lines = [line for line in fhandle if not line.startswith("#")]
        os.chdir(start_dir)
    except:
        print(f"The PFASUM{percent_homology} file was not found.")
        return
    #Cut off the last four amino acids (e.g. B, Z) which are basically
    #"unclear aa"
    lines = lines[:-4]
    #The first line and first column contain the aas. Cut off the
    #unused symbols (B, Z etc) and add a gap.
    aas = lines[0].strip().split()[:-4] + ["-"]
    mat_rows = [[float(z) for z in line.strip().split()[1:-4]] +
               [-4.0] for line in lines[1:]]
    mat_rows += [[-4.0 for i in range(20)] + [1.0]]
    raw_mat = np.asarray(mat_rows)
    raw_mat = (raw_mat - np.min(raw_mat)) / (np.max(raw_mat) - np.min(raw_mat))
    final_mat = np.linalg.cholesky(raw_mat)
    with open("pfasum_matrices.py", "a") as outfile:
        outfile.write("class PFASUM%s_standardized:\n\n"%(percent_homology))
        outfile.write("    @staticmethod\n    def get_aas():\n")
        outfile.write("        return %s\n\n"%aas)
        outfile.write("    @staticmethod\n    def get_mat():\n")
        matstring = ",\n".join(["          %s"%final_mat[i,:].tolist() for i in range(21)])
        outfile.write("        return np.asarray([%s])\n\n\n"%matstring)

def generate_all_pssm_loadfiles():
    """Generate PSSM loadfiles for all the percent homologies of interest."""
    fpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(fpath)
    pfasum_path = "pfasum_matrices"
    for homology in [95,85,75,62,50]:
        generate_pfasum_dotprod_file(homology, pfasum_path)


if __name__ == "__main__":
    generate_all_pssm_loadfiles()
