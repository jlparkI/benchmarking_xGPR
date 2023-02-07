import numpy as np, os, sys



def generate_pfasum_dotprod_file(percent_homology, pfasum_path,
                    enctype = "cosine_dist"):
    if "pfasum_matrices.py" not in os.listdir():
        fhandle = open("pfasum_matrices.py", "w+")
        fhandle.write("import numpy as np\n\n\n")
        fhandle.close()
    else:
        with open("pfasum_matrices.py") as fhandle:
            for line in fhandle:
                if line.endswith("PFASUM%s_%s:"%(percent_homology, enctype)):
                    print("The PFASUM%s file has already been generated."%(percent_homology,
                                    enctype))
                    return
    try:
        start_dir = os.getcwd()
        os.chdir(pfasum_path)
        with open("PFASUM%s.mat"%percent_homology, "r") as fhandle:
            lines = [line for line in fhandle if not line.startswith("#")]
        os.chdir(start_dir)
    except:
        print("The PFASUM%s file was not found."%percent_homology)
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
    if enctype == "cosine_dist":
        sim_mat = np.zeros(raw_mat.shape)
        for i in range(raw_mat.shape[0]):
            for j in range(i, raw_mat.shape[0]):
                norm_i = np.linalg.norm(raw_mat[i,:])
                norm_j = np.linalg.norm(raw_mat[j,:])
                similarity = np.dot(raw_mat[i,:], 
                            raw_mat[j,:]) / (norm_i * norm_j)
                sim_mat[i,j] = similarity
                sim_mat[j,i] = similarity
        final_mat = np.linalg.cholesky(sim_mat)
    else:
        final_mat = (raw_mat - np.mean(raw_mat)) / np.std(raw_mat)
    with open("pfasum_matrices.py", "a") as outfile:
        outfile.write("class PFASUM%s_%s:\n\n"%(percent_homology, enctype))
        outfile.write("    @staticmethod\n    def get_aas():\n")
        outfile.write("        return %s\n\n"%aas)
        outfile.write("    @staticmethod\n    def get_mat():\n")
        matstring = ",\n".join(["          %s"%final_mat[i,:].tolist() for i in range(21)])
        outfile.write("        return np.asarray([%s])\n\n\n"%matstring)
    print("%s complete."%enctype)

def generate_all_pssm_loadfiles():
    fpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(fpath)
    pfasum_path = "pfasum_matrices"
    for homology in [95,85,75,62]:
        for enctype in ["cosine_dist", "standardized"]:
            generate_pfasum_dotprod_file(homology, pfasum_path,
                    enctype)


if __name__ == "__main__":
    generate_all_pssm_loadfiles()
