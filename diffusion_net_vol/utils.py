import os
import hashlib
import numpy as np
import scipy
import torch
import struct


# Default settings and config
def get_default_opts():
    class OptsObject(object):
        pass

    opts = OptsObject()
    opts.eigensystem_cache_dir = None
    return opts


def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device("cpu")).numpy()


def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=1).mean()
    return loss


# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU.
def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen)
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R)

def orthogonalize (vecs, mat):
    mv = []
    for i in range(len(vecs)):
        for j in range(i):
            vecs[i] -= np.dot(vecs[i], mv[j]) * vecs[j]           
        hv = mat @ vecs[i]
        norm = np.sqrt(np.dot(vecs[i], hv))
        vecs[i] *= 1/norm
        hv *= 1/norm
        mv.append (hv)

# Numpy things

# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsr()

    return mat


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()


def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randgen is None:
        randgen = np.random.RandomState()

    theta, phi, z = tuple(randgen.rand(3).tolist())

    theta = theta * 2.0 * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def read_carp_bin_mesh(basename):

    elemfile = basename + '.belem'
    ptsfile = basename + '.bpts'
    lonfile = basename + '.blon'

    # start with the pts file
    with open(ptsfile, 'rb') as file:
        header = file.read(1024)
        # Decode the header as a string and split it
        header_str = header.decode('ascii').strip('\x00')
        numpts, endianness, checksum = map(int, header_str.split())
        print(f"Number of Points: {numpts}")
        #print(f"Endianness: {'little' if endianness == 0 else 'big'}")
        #print(f"Checksum: {checksum}")
        # Determine the byte order
        byte_order = '<' if endianness == 0 else '>'
        # Read the rest of the file
        data = file.read()
        # Assuming the data after the header is a series of integers
        # Adjust the format string if it's a different data type
        format_string = f"{byte_order}{3*numpts}f"
        # Unpack the data
        unpacked_data = struct.unpack(format_string, data)
        xyz = np.array(unpacked_data).reshape(numpts, -1)

    # now the element file
    with open(elemfile, 'rb') as file:
        header = file.read(1024)
        # Decode the header as a string and split it
        header_str = header.decode('ascii').strip('\x00')
        numele, endianness, checksum = map(int, header_str.split())

        print(f"Number of elements: {numele}")
        # Determine the byte order
        byte_order = '<' if endianness == 0 else '>'
        # Read the rest of the file
        data = file.read()
        # Assuming the data after the header is a series of integers
        # Adjust the format string if it's a different data type
        format_string = f"{byte_order}{6*numele}i"

        # Unpack the data
        unpacked_data = struct.unpack(format_string, data)

    temp = np.array(unpacked_data).reshape(numele, -1)
    con = temp[:,1:5]
    con = con[:, [1,0,2,3]]
    tags = temp[:,-1]

    # finally the lon file
    with open(lonfile, 'rb') as file:
        header = file.read(1024)
        # Decode the header as a string and split it
        header_str = header.decode('ascii').strip('\x00')
        numfibers, numele, endianness, checksum = map(int, header_str.split())

        print(f"Number of Fibers: {numfibers}")
        # Determine the byte order
        byte_order = '<' if endianness == 0 else '>'

        # Read the rest of the file
        data = file.read()
        # Assuming the data after the header is a series of integers
        # Adjust the format string if it's a different data type
        format_string = f"{byte_order}{numfibers*3*numele}f"
        # Unpack the data
        unpacked_data = struct.unpack(format_string, data)
    lon = np.array(unpacked_data).reshape(numele, -1)
    return xyz, con.astype(np.int64), tags, lon

def PINVIT(A, M, P, k=10, maxiter=20):  
    rng = np.random.default_rng(seed=42)
    uvecs = np.zeros((k, A.shape[0]))
    vecs =  rng.random((k, A.shape[0]))
    vecs = np.concatenate((vecs, np.zeros((k, A.shape[0]))), 0)
    uvecs = (P @ vecs[:k].T).T
    lams = np.ones(k)
    
    asmall = np.zeros((2*k, 2*k))
    msmall = np.zeros((2*k, 2*k))
    for i in range(maxiter):
        vecs[0:k] = (A @ uvecs.T).T - (lams * (M @ uvecs.T)).T
        vecs[k:2*k] = (P @ vecs[0:k].T).T
        vecs[0:k] = uvecs
        orthogonalize(vecs, M)
        asmall = (vecs @ A @ vecs.T)
        msmall = (vecs @ M @ vecs.T)

        ev,evec = scipy.linalg.eigh(a=asmall, b=msmall)
    
        lams = ev[0:k]
        
        uvecs = evec[:,0:k].T @ vecs
    
    return lams, uvecs


def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
