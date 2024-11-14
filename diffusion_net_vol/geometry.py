import os.path
import random
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import pyamg

# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import scipy.spatial
import torch
import sklearn.neighbors

from .utils import toNP, sparse_np_to_torch, ensure_dir_exists, hash_arrays, sparse_torch_to_np, PINVIT


def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def norm2(x, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return dot(x, x)


def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if len(x.shape) == 1:
        raise ValueError(
            "called normalize() on single vector of dim "
            + str(x.shape)
            + " are you sure?"
        )
    if not highdim and x.shape[-1] > 4:
        raise ValueError(
            "called normalize() with large last dimension "
            + str(x.shape)
            + " are you sure?"
        )
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)


def tet_coords(verts, tets):
    coords = verts[tets]
    return coords


def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)

def volume(V, T):
    """
    Compute the volume of tetrahedra
    """
    v321 = V[T[:, 2]] - V[T[:, 3]]
    v421 = V[T[:, 3]] - V[T[:, 0]]
    v131 = V[T[:, 0]] - V[T[:, 1]]
    return np.abs(np.sum(np.cross(v321, v421) * v131, axis=1) / 6)

def doublearea_intrinsic_v2(l: np.ndarray, degenerate_replacement=None) -> np.ndarray:
    """
    Compute double area of triangle from edge lengths using Heron's formula
    """
    s = np.sum(l, axis=1) * 0.5
    ret = 2 * np.sqrt(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]))
    if degenerate_replacement is not None:
        ret[ret < 0] = degenerate_replacement
    return ret

def dihedral_angles(V, T, **kwargs):
    """
    Compute dihedral angles for all tets of a given tet mesh (V,T)

    Parameters:
    V : numpy.ndarray
        #V by dim list of vertex positions
    T : numpy.ndarray
        #V by 4 list of tet indices
    **kwargs : dict
        Optional parameters:
        'SideLengths': #T by 6 list of tet edges lengths: 41 42 43 23 31 12
        'FaceAreas': #T by 4 list of tet face areas

    Returns:
    theta : numpy.ndarray
        #T by 6 list of dihedral angles (in radians)
    cos_theta : numpy.ndarray
        #T by 6 list of cosine of dihedral angles (in radians)
    """
    l = kwargs.get('SideLengths', None)
    s = kwargs.get('FaceAreas', None)

    if l is None:
        # lengths of edges opposite *face* pairs: 23 31 12 41 42 43
        l = np.hstack([
            np.sqrt(np.sum((V[T[:, 3]] - V[T[:, i]])**2, axis=1))[:, np.newaxis]
            for i in range(3)
        ] + [
            np.sqrt(np.sum((V[T[:, i]] - V[T[:, j]])**2, axis=1))[:, np.newaxis]
            for i, j in [(1, 2), (2, 0), (0, 1)]
        ])

    # (unsigned) face Areas (opposite vertices: 1 2 3 4)
    if s is None:
        s = 0.5 * np.column_stack([
            doublearea_intrinsic_v2(l[:, [1, 2, 3]]),
            doublearea_intrinsic_v2(l[:, [0, 2, 4]]),
            doublearea_intrinsic_v2(l[:, [0, 1, 5]]),
            doublearea_intrinsic_v2(l[:, [3, 4, 5]])
        ])

    # Law of cosines
    H_sqr = (1/16) * (
        4 * l[:, [3, 4, 5, 0, 1, 2]]**2 * l[:, [0, 1, 2, 3, 4, 5]]**2 -
        ((l[:, [1, 2, 3, 4, 5, 0]]**2 + l[:, [4, 5, 0, 1, 2, 3]]**2) -
         (l[:, [2, 3, 4, 5, 0, 1]]**2 + l[:, [5, 0, 1, 2, 3, 4]]**2))**2
    )

    cos_theta = (H_sqr - s[:, [1, 2, 0, 3, 3, 3]]**2 - s[:, [2, 0, 1, 0, 1, 2]]**2) / (
        -2 * s[:, [1, 2, 0, 3, 3, 3]] * s[:, [2, 0, 1, 0, 1, 2]]
    )

    theta = np.arccos(cos_theta)

    return theta, cos_theta

def volume_intrinsic(l):
    """
    Compute volumes of tets defined intrinsically by edge lengths l

    Parameters:
    l : numpy.ndarray
        #T by 6 list of tetrahedra side lengths of edges opposite *face* pairs
        [23 31 12 41 42 43]

    Returns:
    vol : numpy.ndarray
        #T list of tet volumes (always positive)

    Note:
    This function is based on the Heron-type formula for the volume of a tetrahedron.
    http://en.wikipedia.org/wiki/Heron%27s_formula#Heron-type_formula_for_the_volume_of_a_tetrahedron
    U, V, W, u, v, w are lengths of edges of the tetrahedron (first three form
    a triangle; u opposite to U and so on)
    """
    u, v, w, U, V, W = l[:, 0], l[:, 1], l[:, 2], l[:, 3], l[:, 4], l[:, 5]

    X = (w - U + v) * (U + v + w)
    x = (U - v + w) * (v - w + U)
    Y = (u - V + w) * (V + w + u)
    y = (V - w + u) * (w - u + V)
    Z = (v - W + u) * (W + u + v)
    z = (W - u + v) * (u - v + W)

    a = np.sqrt(x * Y * Z)
    b = np.sqrt(y * Z * X)
    c = np.sqrt(z * X * Y)
    d = np.sqrt(x * y * z)

    vol = np.sqrt(
        (-a + b + c + d) *
        ( a - b + c + d) *
        ( a + b - c + d) *
        ( a + b + c - d)
    ) / (192 * u * v * w)

    return vol


def cotangent(V: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the cotangents of each angle in mesh (V,F).

    Parameters:
    V : np.ndarray
        #V by dim list of rest domain positions
    F : np.ndarray
        #F by {3|4} list of {triangle|tetrahedra} indices into V
    **kwargs : dict
        Optional parameters:
        'SideLengths': #F by 3 list of edge lengths for triangles (23, 31, 12)
                       or #T by 6 list of tet edge lengths (41, 42, 43, 23, 31, 12)
        'FaceAreas': #T by 4 list of tet face areas (for tetrahedra only)

    Returns:
    C : np.ndarray
        #F by {3|6} list of cotangents corresponding to:
        - angles for triangles, columns correspond to edges 23,31,12
        - dihedral angles *times opposite edge length* over 6 for tets,
          columns correspond to *faces* 23,31,12,41,42,43

    Note:
    Known bugs:
    - This seems to return 0.5*C and for tets already multiplies by edge-lengths
    """

    if F.shape[1] == 3:
        return _cotangent_triangle(V, F, **kwargs)
    elif F.shape[1] == 4:
        return _cotangent_tetrahedra(V, F, **kwargs)
    else:
        raise ValueError('Unsupported simplex type')

def _cotangent_triangle(V: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
    l = kwargs.get('SideLengths', None)

    if l is None:
        l = np.sqrt(np.sum((V[F[:, [1,2,0]]] - V[F[:, [2,0,1]]])**2, axis=2))

    s = np.sum(l, axis=1) * 0.5
    dblA = 2 * np.sqrt(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]))

    C = (l**2 + np.roll(l, -1, axis=1)**2 - np.roll(l, -2, axis=1)**2) / (dblA[:, np.newaxis] * 4)

    return C

def _cotangent_tetrahedra(V: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
    l = kwargs.get('SideLengths', None)
    s = kwargs.get('FaceAreas', None)

    if l is None:
        l = np.sqrt(np.sum((V[F[:, [3,3,3,1,2,0]]] - V[F[:, [0,1,2,2,0,1]]])**2, axis=2))

    if s is None:
        s = 0.5 * np.column_stack([
            doublearea_intrinsic_v2(l[:, [1,2,3]]),
            doublearea_intrinsic_v2(l[:, [0,2,4]]),
            doublearea_intrinsic_v2(l[:, [0,1,5]]),
            doublearea_intrinsic_v2(l[:, [3,4,5]])
        ])

    _, cos_theta = dihedral_angles(None, None, SideLengths=l, FaceAreas=s)
    vol = volume_intrinsic(l)

    sin_theta = vol[:, np.newaxis] / ((2 / (3 * l)) * s[:, [1,2,0,3,3,3]] * s[:, [2,0,1,0,1,2]])

    C = 1/6 * l * cos_theta / sin_theta

    return C



def cotmatrix(V: np.ndarray, F: np.ndarray) -> sparse.csr_matrix:
    """
    Compute cotangent matrix (Laplacian mesh operator)

    Parameters:
    V : np.ndarray
        #V x dim matrix of vertex coordinates
    F : np.ndarray
        #F x simplex-size matrix of indices of triangle or tetrahedron corners

    Returns:
    L : scipy.sparse.csr_matrix
        sparse #V x #V matrix of cot weights

    Note:
    For size(F,2)==4, this is distinctly NOT following the definition that appears
    in the appendix of: "Interactive Topology-aware Surface Reconstruction,"
    by Sharf, A. et al.
    Instead, it is a purely geometric construction.
    """

    ss = F.shape[1]
    if ss == 3:
        return _cotmatrix_triangle(V, F)
    elif ss == 4:
        return _cotmatrix_tetrahedra(V, F)
    else:
        raise ValueError('Unsupported simplex type')

def _cotmatrix_triangle(V: np.ndarray, F: np.ndarray) -> sparse.csr_matrix:
    if F.shape[0] == 3 and F.shape[1] != 3:
        print('Warning: F seems to be 3 by #F, it should be #F by 3')
        F = F.T

    i1, i2, i3 = F[:, 0], F[:, 1], F[:, 2]
    v1 = V[i3] - V[i2]
    v2 = V[i1] - V[i3]
    v3 = V[i2] - V[i1]

    if V.shape[1] == 2:
        dblA = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    elif V.shape[1] == 3:
        n = np.cross(v1, v2)
        dblA = np.sqrt(np.sum(n**2, axis=1))
    else:
        raise ValueError(f'Unsupported vertex dimension {V.shape[1]}')

    cot12 = -np.sum(v1 * v2, axis=1) / dblA / 2
    cot23 = -np.sum(v2 * v3, axis=1) / dblA / 2
    cot31 = -np.sum(v3 * v1, axis=1) / dblA / 2
    diag1 = -cot12 - cot31
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23

    i = np.concatenate([i1, i2, i2, i3, i3, i1, i1, i2, i3])
    j = np.concatenate([i2, i1, i3, i2, i1, i3, i1, i2, i3])
    v = np.concatenate([cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3])

    return sparse.csr_matrix((v, (i, j)), shape=(V.shape[0], V.shape[0]))

def _cotmatrix_tetrahedra(V: np.ndarray, F: np.ndarray) -> sparse.csr_matrix:
    if F.shape[0] == 4 and F.shape[1] != 4:
        print('Warning: F seems to be 4 by #F, it should be #F by 4')

    n = V.shape[0]
    C = cotangent(V, F)

    L = sparse.csr_matrix((C[:, 0], (F[:, 1], F[:, 2])), shape=(n, n)) + \
        sparse.csr_matrix((C[:, 1], (F[:, 2], F[:, 0])), shape=(n, n)) + \
        sparse.csr_matrix((C[:, 2], (F[:, 0], F[:, 1])), shape=(n, n)) + \
        sparse.csr_matrix((C[:, 3], (F[:, 3], F[:, 0])), shape=(n, n)) + \
        sparse.csr_matrix((C[:, 4], (F[:, 3], F[:, 1])), shape=(n, n)) + \
        sparse.csr_matrix((C[:, 5], (F[:, 3], F[:, 2])), shape=(n, n))

    L = L + L.T
    L = L - sparse.diags(L.sum(axis=1).A1)

    if np.all(L.diagonal() > 0):
        print('Warning: Flipping sign of cotmatrix3, so that diag is negative')
        L = -L

    return L

def massmatrix(V, F, type='barycentric'):
    """
    Compute mass matrix for the mesh given by V and F

    Parameters:
    V : np.array
        #V x 3 matrix of vertex coordinates
    F : np.array
        #F x simplex-size matrix of indices of simplex corners
    type : str, optional
        Type of mass matrix to compute (default is 'voronoi')
        - 'full': full mass matrix for p.w. linear fem
        - 'barycentric': diagonal lumped mass matrix obtained by summing 1/3
        - 'voronoi': true voronoi area, except in cases where triangle is obtuse
          then uses 1/2, 1/4, 1/4 {simplex size 3 only}

    Returns:
    M : scipy.sparse.csr_matrix
        #V by #V sparse mass matrix
    """

    ss = F.shape[1]  # simplex size

    if ss == 2:
        l = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
        if type in ['voronoi', 'barycentric']:
            M = sparse.csr_matrix((np.concatenate([l, l]),
                                   (np.concatenate([F[:, 0], F[:, 1]]),
                                    np.concatenate([F[:, 0], F[:, 1]]))),
                                  shape=(V.shape[0], V.shape[0]))
        elif type == 'full':
            i1, i2 = F[:, 0], F[:, 1]
            i = np.concatenate([i1, i2, i1, i2])
            j = np.concatenate([i2, i1, i1, i2])
            v = np.concatenate([l/4, l/4, l/2, l/2])
            M = sparse.csr_matrix((v, (i, j)), shape=(V.shape[0], V.shape[0]))

    elif ss == 3:
        i1, i2, i3 = F[:, 0], F[:, 1], F[:, 2]
        v1 = V[i3] - V[i2]
        v2 = V[i1] - V[i3]
        v3 = V[i2] - V[i1]

        if V.shape[1] == 2:
            dblA = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        elif V.shape[1] == 3:
            n = np.cross(v1, v2)
            dblA = np.linalg.norm(n, axis=1)
        else:
            raise ValueError(f'Unsupported vertex dimension {V.shape[1]}')

        if type == 'full':
            i = np.concatenate([i1, i2, i2, i3, i3, i1, i1, i2, i3])
            j = np.concatenate([i2, i1, i3, i2, i1, i3, i1, i2, i3])
            offd_v = dblA / 24
            diag_v = dblA / 12
            v = np.concatenate([offd_v, offd_v, offd_v, offd_v, offd_v, offd_v, diag_v, diag_v, diag_v])
            M = sparse.csr_matrix((v, (i, j)), shape=(V.shape[0], V.shape[0]))

        elif type == 'barycentric':
            i = np.concatenate([i1, i2, i3])
            j = np.concatenate([i1, i2, i3])
            diag_v = dblA / 6
            v = np.concatenate([diag_v, diag_v, diag_v])
            M = sparse.csr_matrix((v, (i, j)), shape=(V.shape[0], V.shape[0]))

        elif type == 'voronoi':
            # This part needs the implementation of massmatrix_intrinsic function
            # which is not provided in the original MATLAB code
            raise NotImplementedError("Voronoi mass matrix for triangles is not implemented yet")

    elif ss == 4:
        assert V.shape[1] == 3, "Vertices must be defined in 3D for tetrahedral meshes"

        if type == 'full':
            vol = np.abs(volume(V, F))

            # Construct indices for all 16 combinations
            i = F[:, np.array([1,2,3,0,2,3,0,1,3,0,1,2,0,1,2,3])]
            j = F[:, np.array([0,0,0,1,1,1,2,2,2,3,3,3,0,1,2,3])]
            i = i.flatten()
            j = j.flatten()

            # Construct values
            v = np.concatenate([
                np.repeat(vol/20, 12),  # Off-diagonal elements
                np.repeat(vol/10, 4)    # Diagonal elements
            ])

            M = sparse.csr_matrix((v, (i, j)), shape=(V.shape[0], V.shape[0]))

        elif type == 'barycentric':
            vol = np.abs(volume(V, F))
            v = np.repeat(vol, 4) / 4
            M = sparse.csr_matrix((v, (F.flatten(), F.flatten())), shape=(V.shape[0], V.shape[0]))
            M = sparse.diags(M.diagonal())

        elif type == 'voronoi':
            # Implement Voronoi mass matrix for tetrahedra
            raise NotImplementedError("Voronoi mass matrix for tetrahedra is not implemented yet")

    else:
        raise ValueError(f'Unsupported simplex size: {ss}')

    if np.any(M.sum(axis=1) == 0):
        print('Warning: Some rows have all zeros... probably unreferenced vertices.')

    return M

def get_edges(F):
    """
    Compute the unique undirected edges of a simplicial complex

    Parameters:
    -----------
    F : ndarray
        #F x simplex-size matrix of indices of simplex corners

    Returns:
    --------
    E : ndarray
        Edges in sorted order, direction of each is also sorted

    Example:
    --------
    # get unique undirected edges
    E = edges(F)
    # get unique directed edges
    E_directed = np.vstack((E, E[:, [1,0]]))
    """
    # Get all combinations of edges
    n = F.shape[1]
    # Generate all possible pairs of column indices
    e = np.array([(i,j) for i in range(n) for j in range(i+1,n)])

    # Extract edges from F using the combinations
    edge_start = F[:, e[:, 0]].flatten()
    edge_end = F[:, e[:, 1]].flatten()

    # Create sparse matrix
    max_val = np.max(F)
    A = sparse.csr_matrix((np.ones(len(edge_start)),
                   (edge_start, edge_end)),
                   shape=(max_val+1, max_val+1))

    # Get lower triangular part of symmetric matrix
    A = A + A.T
    A = sparse.tril(A)

    # Find non-zero elements
    rows, cols = A.nonzero()

    # Return edges with sorted columns
    E = np.column_stack((cols, rows)).astype(np.int64)
    return E

def edge_vectors(verts, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    return torch.tensor(edge_vecs)


def build_grad(verts, edges, edge_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient.
    All values pointwise.
    - edges: (2, E)
    """
  
    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    spacedim = verts.shape[1]

    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges.shape[1]):
        tail_ind = edges[0, iE]
        tip_ind = edges[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals_X = []
    data_vals_Y = []
    data_vals_Z = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, spacedim))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_vectors[iE][:]
            w_e = 1.0

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(spacedim)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        solX = sol_mat[0, :]
        solY = sol_mat[1, :]
        solZ = sol_mat[2, :]

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals_X.append(solX[i_neigh])
            data_vals_Y.append(solY[i_neigh])
            data_vals_Z.append(solZ[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals_X = np.array(data_vals_X)
    data_vals_Y = np.array(data_vals_Y)
    data_vals_Z = np.array(data_vals_Z)
    mat = [ sparse.coo_matrix((data_vals_X, (row_inds, col_inds)), shape=(N, N)).tocsr(),
            sparse.coo_matrix((data_vals_Y, (row_inds, col_inds)), shape=(N, N)).tocsr(),
            sparse.coo_matrix((data_vals_Z, (row_inds, col_inds)), shape=(N, N)).tocsr()]
    return mat


def compute_operators(verts, tets, k_eig=32, eig_solver='PINVIT'):
    """
    Builds spectral operators for a mesh. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.
    See get_operators() for a similar routine that wraps this one with a layer of caching.
    Torch in / torch out.
    Arguments:
      - vertices: (V,3) vertex positions
      - faces: (F,4) list of tets.
      - k_eig: number of eigenvectors to use
      - eig_solver: solver used for estimating eigenvalues and eigenvectors
    Returns:
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient
      - gradZ: same as gradX but for Z-component of gradient
    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix,
    like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """
    dtype = verts.dtype
    eps = 1e-8

    # Build the scalar Laplacian
    # L, M = robust_laplacian.mesh_laplacian(verts_np, faces_np)
    # massvec_np = M.diagonal()
    L = (-1.0) * cotmatrix(verts, tets)
    massvec = massmatrix(verts, tets).diagonal()
    massvec += eps * np.mean(massvec)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec_np).any():
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # === Compute the eigenbasis
    if k_eig > 0:

        # Prepare matrices
        L_eigsh = (L + sparse.identity(L.shape[0]) * eps)
        massvec_eigsh = massvec
        Mmat = sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise,
                # but for some reason those parameters seem to have no effect

                # build an AMG prec for LOBPCG solver
                B = np.ones((L_eigsh.shape[0], 1))
                ml = pyamg.smoothed_aggregation_solver(L_eigsh , B)
                Mp = ml.aspreconditioner()
                
                if eig_solver == 'PINVIT':
                    evals, evecs = PINVIT(L_eigsh, Mmat, Mp, k_eig)
                    evecs = evecs_np.T
                else:
                    # compute eigenvalues and eigenvectors with LOBPCG
                    # initial approximation to the K eigenvectors
                    rng = np.random.default_rng(seed=42)
                    X = rng.standard_normal((L_eigsh.shape[0], k_eig))
                    # preconditioner based on ml    
                    evals, evecs = sparse.linalg.lobpcg(L_eigsh, X, M=Mp, B=Mmat, largest=False, maxiter=50)
                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
                

                break
            except Exception as e:
                print(e)
                if failcount > 3:
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + sparse.identity(L.shape[0]) * (
                    eps * 10 ** failcount
                )

    else:  # k_eig == 0
        evals = np.zeros((0))
        evecs = np.zeros((verts.shape[0], 0))

    # == Build gradient matrices

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood
    edges = np.stack((inds_row, inds_col), axis=0)
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    grads = build_grad(verts, edges, edge_vecs)
    gradX = grads[0]
    gradY = grads[1]
    gradZ = grads[2]

    # === Convert back to torch
    return evals, evecs, gradX, gradY, gradZ


def get_all_operators(verts_list, tets_list, k_eig, op_cache_dir=None):
    N = len(verts_list)
    massvec = [None] * N
    L = [None] * N
    evals = [None] * N
    evecs = [None] * N
    gradX = [None] * N
    gradY = [None] * N
    gradZ = [None] * N

    # process in random order
    inds = [i for i in range(N)]
    random.shuffle(inds)

    for num, i in enumerate(inds):
        print(
            "get_all_operators() processing {} / {} {:.3f}%".format(
                num, N, num / N * 100
            )
        )
        outputs = get_operators(verts_list[i], tets_list[i], k_eig, op_cache_dir)
        massvec[i] = outputs[0]
        L[i] = outputs[1]
        evals[i] = outputs[2]
        evecs[i] = outputs[3]
        gradX[i] = outputs[4]
        gradY[i] = outputs[5]
        gradZ[i] = outputs[6]

    return massvec, L, evals, evecs, gradX, gradY, gradZ


def get_operators(
    verts,
    tets,
    k_eig=32,
    op_cache_dir=None,
    overwrite_cache=False,
    truncate_cache=False,
):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk,
    and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(tets)

    if np.isnan(verts_np).any():
        raise RuntimeError("tried to construct operators from NaN verts")

    # Check the cache directory
    # Note 1: Collisions here are exceptionally unlikely, so we could probably just use the hash...
    #         but for good measure we check values nonetheless.
    # Note 2: There is a small possibility for race conditions to lead to bucket gaps or duplicate
    #         entries in this cache. The good news is that that is totally fine, and at most slightly
    #         slows performance with rare extra cache misses.
    found = False
    if op_cache_dir is not None:
        ensure_dir_exists(op_cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir, hash_key_str + "_" + str(i_cache_search) + ".npz"
            )

            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_tets = npzfile["tets"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (
                    not np.array_equal(tets, cache_tets)
                ):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache:
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break

                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break

                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:, :k_eig]
                gradX = read_sp_mat("gradX")
                gradY = read_sp_mat("gradY")
                gradZ = read_sp_mat("gradZ")

                if truncate_cache and cache_k_eig > k_eig:
                    assert False, "Error, Case not covered"

                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = sparse_np_to_torch(gradY).to(device=device, dtype=dtype)
                gradZ = sparse_np_to_torch(gradZ).to(device=device, dtype=dtype)
                found = True

                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break

            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:

        # No matching entry found; recompute.
        mass, L, evals, evecs, gradX, gradY, gradZ = compute_operators(
            verts, tets, k_eig
        )

        dtype_np = np.float32

        # Store it in the cache
        if op_cache_dir is not None:

            L_np = sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = sparse_torch_to_np(gradY).astype(dtype_np)
            gradZ_np = sparse_torch_to_np(gradZ).astype(dtype_np)

            np.savez(
                search_path,
                verts=verts_np,
                tets=tets_np,
                k_eig=k_eig,
                mass=toNP(mass).astype(dtype_np),
                L_data=L_np.data,
                L_indices=L_np.indices,
                L_indptr=L_np.indptr,
                L_shape=L_np.shape,
                evals=toNP(evals).astype(dtype_np),
                evecs=toNP(evecs).astype(dtype_np),
                gradX_data=gradX_np.data,
                gradX_indices=gradX_np.indices,
                gradX_indptr=gradX_np.indptr,
                gradX_shape=gradX_np.shape,
                gradY_data=gradY_np.data,
                gradY_indices=gradY_np.indices,
                gradY_indptr=gradY_np.indptr,
                gradY_shape=gradY_np.shape,
                gradZ_data=gradZ_np.data,
                gradZ_indices=gradZ_np.indices,
                gradZ_indptr=gradZ_np.indptr,
                gradZ_shape=gradZ_np.shape,
            )

    return mass, L, evals, evecs, gradX, gradY, gradZ


def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        raise Exception("Complexis analysis not implemented!")
    else:
        return torch.matmul(basis, values)


def compute_hks(evals, evecs, scales): #numpy
    """
    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) values
      - scales: (S) times
    Outputs:
      - (V,S) hks values
    """

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = np.expand_dims(evals, axis=0)
        evecs = np.expand_dims(evecs, axis=0)
        scales = np.expand_dims(scales, axis=0)
    else:
        expand_batch = False

    # TODO could be a matmul
    power_coefs = np.exapnd_dims(np.exp(-np.expand_dims(evals, axis=1) * np.expand_dims(scales, axis=-1)), axis=1) # (B,1,S,K)
    terms = power_coefs * np.expand_dims((evecs * evecs), axis=2)  # (B,V,S,K)

    out = np.sum(terms, axis=-1)  # (B,V,S)

    if expand_batch:
        return np.squeeze(out, axis=0)
    else:
        return out


def compute_hks_autoscale(evals, evecs, count): # numpy
    # these scales roughly approximate those suggested in the hks paper
    scales = np.logspace(-2, 0.0, steps=count)
    return compute_hks(evals, evecs, scales)

#numpy version
def normalize_positions(pos, method="mean"):
    # center and unit-scale positions

    if method == "mean":
        # center using the average point position
        pos = pos - np.mean(pos, axis=-2, keepdims=True)
    elif method == "bbox":
        # center via the middle of the axis-aligned bounding box
        bbox_min = np.min(pos, dim=-2)
        bbox_max = np.max(pos, dim=-2)
        center = (bbox_max + bbox_min) / 2.0
        pos -= np.expand_dims(center, axis=-2)
    else:
        raise ValueError("unrecognized method")

    scale = np.expand_dims(np.max(np.linalg.norm(pos, axis=len(pos.shape)-1), dim=-1, keepdims=True), axis=-1)
    pos = pos / scale
    return pos


# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(
    points_source, points_target, k, largest=False, omit_diagonal=False, method="brute"
):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError(
            "omit_diagonal can only be used when source and target are same shape"
        )

    if method != "cpu_kd" and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = "cpu_kd"
        print("switching to cpu_kd knn")

    if method == "brute":

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(
            -1, points_target.shape[0], -1
        )
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(
            points_source.shape[0], -1, -1
        )

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float("inf")

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result

    elif method == "cpu_kd":

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k + 1 if omit_diagonal else k
        _, neighbors = kd_tree.query(points_source_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape(
                (neighbors.shape[0], neighbors.shape[1] - 1)
            )

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds

    else:
        raise ValueError("unrecognized method")


def farthest_point_sampling(points, n_sample):
    # Torch in, torch out. Returns a |V| mask with n_sample elements set to true.

    N = points.shape[0]
    if n_sample > N:
        raise ValueError("not enough points to sample")

    chosen_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_dists = torch.ones(N, dtype=points.dtype, device=points.device) * float("inf")

    # pick the centermost first point
    points = normalize_positions(points)
    i = torch.min(norm2(points), dim=0).indices
    chosen_mask[i] = True

    for _ in range(n_sample - 1):

        # update distance
        dists = norm2(points[i, :].unsqueeze(0) - points)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        i = torch.max(min_dists, dim=0).indices.item()
        chosen_mask[i] = True

    return chosen_mask

#numpy
def normalize_volume_scale(verts, tets):
    """
    Normalizes a mesh by applying a uniform scaling such that it has volume 1.
    Returns only the new vertices, faces are unchagned.
    """

    # compute total surface area
    coords = tet_coords(verts, tets)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    vec_C = coords[:, 3, :] - coords[:, 0, :]
    tet_vols = np.abs(np.einsum('ij,ij->i', cross(vec_A, vec_B, axis=-1), vec_C)) / 6.
    total_vol = np.sum(tet_vols)
    # scale
    scale = total_vol ** (-1. / 3.)
    verts = verts * scale

    return verts
