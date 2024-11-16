import os
from pathlib import Path
import sys
from itertools import permutations  # , combinations
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # add the path to the DiffusionNet src
import diffusion_net_vol  # noqa

class HeartDataset(Dataset):
    def __init__(self, root_dir, name="hearts", train=True, k_eig=64, n_fmap=30, use_cache=True):

        self.train = train  # bool
        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")

        #make sure we have the cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # store in memory
        self.verts_list = []
        self.tets_list = []
        self.vts_list = []
        self.names_list = []
        self.massvec_list = []
        self.evals_list = []
        self.evecs_list = []
        self.grad_list = []
        self.hks_list = []

        # set combinations
        n_total = 20
        if self.train:
            self.combinations = list(permutations(range(n_total), 2))
        else:
            self.combinations = list(permutations(range(n_total), 2))

        # Get all the files
        mesh_files = []
        cache_files = []

        # load faust data
        mesh_dirpath = (Path(self.root_dir)).resolve()
        for fname in mesh_dirpath.iterdir():
            if fname.suffix != ".belem":
                continue
            mesh_fullpath = os.path.join(mesh_dirpath, fname.stem)
            cache_file = os.path.join(self.cache_dir, fname.stem + "_cache.npz")
            mesh_files.append(mesh_fullpath)
            cache_files.append(cache_file)

        print("loading {} meshes".format(len(mesh_files)))

        mesh_files = sorted(mesh_files)
        cache_files = sorted(cache_files)

        # for saving numpy arrays
        dtype_np = np.float32
        device = torch.device("cpu")
        
        # Load the actual files
        for mfile, cfile in zip(mesh_files, cache_files):
            print("processing ", os.path.basename(mfile))
            if use_cache:
                print("using dataset cache path: " + str(cfile))
                if os.path.exists(cfile):
                    print("  --> loading dataset from cache")
                    npzfile = np.load(cfile, allow_pickle=True)
                    self.names_list.append(npzfile["name"].item())
                    cache_verts = torch.tensor(npzfile["verts"]).float()
                    self.verts_list.append(cache_verts)
                    self.tets_list.append(torch.tensor(npzfile["tets"]))
                    self.k_eig = npzfile["k_eig"].item()
                    self.massvec_list.append(torch.from_numpy(npzfile["mass"]))
                    self.hks_list.append(torch.from_numpy(npzfile["hks"]))
                    self.evals_list.append(torch.from_numpy(npzfile["evals"]))
                    self.evecs_list.append(torch.from_numpy(npzfile["evecs"]))
                    data = npzfile["grad_data"]
                    indices = npzfile["grad_indices"]
                    indptr = npzfile["grad_indptr"]
                    mat = scipy.sparse.csr_matrix((data, indices, indptr))
                    self.grad_list.append(diffusion_net_vol.utils.sparse_np_to_torch(mat).coalesce())
                    continue
                print("  --> dataset not in cache, repopulating")
                
            print("loading mesh " + str(mfile))

            verts, tets, _, _ = diffusion_net_vol.utils.read_carp_bin_mesh(mfile)
            
            # center and unit scale
            verts = np.ascontiguousarray(verts)
            tets  = np.ascontiguousarray(tets)
            
            verts = diffusion_net_vol.geometry.normalize_positions(verts)

            # normalize area
            verts = diffusion_net_vol.geometry.normalize_volume_scale(verts, tets)
            mass, evals, evecs, gradMat = diffusion_net_vol.geometry.compute_operators(verts, tets, k_eig=k_eig)

            # hks
            hks = diffusion_net_vol.geometry.compute_hks_autoscale(evals, evecs, 16)

            # save stuff
            gradMat32 = gradMat.astype(dtype_np)
            
            np.savez(
                cfile,
                name=os.path.basename(mfile),
                verts=verts.astype(dtype_np),
                tets=tets,
                k_eig=k_eig,
                mass=mass.astype(dtype_np),
                hks=hks.astype(dtype_np),
                evals=evals.astype(dtype_np),
                evecs=evecs.astype(dtype_np),
                grad_data=gradMat32.data,
                grad_indices=gradMat32.indices,
                grad_indptr=gradMat32.indptr
            )
            
            verts = torch.tensor(verts).float()  
            self.massvec_list.append(torch.from_numpy(mass).to(device=device, dtype=verts.dtype))
            self.evals_list.append(torch.from_numpy(evals).to(device=device, dtype=verts.dtype))
            self.evecs_list.append(torch.from_numpy(evecs).to(device=device, dtype=verts.dtype))
            self.grad_list.append(diffusion_net_vol.utils.sparse_np_to_torch(gradMat).coalesce().to(device=device, dtype=verts.dtype))
            self.hks_list.append(torch.from_numpy(hks).to(device=device, dtype=verts.dtype))
            self.verts_list.append(verts)
            self.tets_list.append(torch.tensor(tets))
            self.names_list.append(os.path.basename(mfile))
            


    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = {
            "xyz": self.verts_list[idx1],
            "tets": self.tets_list[idx1],
            "mass": self.massvec_list[idx1],
            "evals": self.evals_list[idx1],
            "evecs": self.evecs_list[idx1],
            "grad": self.grad_list[idx1],
            "name": self.names_list[idx1]
        }

        shape2 = {
            "xyz": self.verts_list[idx2],
            "tets": self.tets_list[idx2],
            "mass": self.massvec_list[idx2],
            "evals": self.evals_list[idx2],
            "evecs": self.evecs_list[idx2],
            "grad": self.grad_list[idx2],
            "name": self.names_list[idx2]
        }

        return {"shape1": shape1, "shape2": shape2}


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "tets", "mass", "evals", "evecs", "grad"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                v[name] = v[name].to(device)
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape