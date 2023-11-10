import numpy as np


# Euler angles point group O. Convention ZYZ
euler_angles_O = np.array([
    # I
    np.array([0.0, 0.0, 0.0]),
    # 8C_3
    np.array([-np.pi, -np.pi, 0.0]) / 2.0,
    np.array([0.0, np.pi, np.pi]) / 2.0,
    np.array([0.0, -np.pi, -np.pi]) / 2.0,
    np.array([np.pi, np.pi, 0.0]) / 2.0,
    np.array([-np.pi, np.pi, 0.0]) / 2.0,
    np.array([0.0, -np.pi, np.pi]) / 2.0,
    np.array([0.0, np.pi, -np.pi]) / 2.0,
    np.array([np.pi, -np.pi, 0.0]) / 2.0,
    # 6C_4
    np.array([-np.pi, -np.pi, np.pi]) / 2.0,
    np.array([np.pi, -np.pi, -np.pi]) / 2.0,
    np.array([0.0, -np.pi, 0.0]) / 2.0,
    np.array([0.0, np.pi, 0.0]) / 2.0,
    np.array([-np.pi, 0.0, 0.0]) / 2.0,
    np.array([np.pi, 0.0, 0.0]) / 2.0,
    # 6C'_2
    np.array([-np.pi, -np.pi, -np.pi]) / 2.0,
    np.array([-np.pi, np.pi, -np.pi]) / 2.0,
    np.array([-np.pi / 2.0, -np.pi, 0.0]),
    np.array([0.0, np.pi, -np.pi / 2.0]),
    np.array([0.0, np.pi / 2.0, -np.pi]),
    np.array([0.0, -np.pi / 2.0, -np.pi]), 
    # 3C_2
    np.array([np.pi, np.pi, 0.0]),
    np.array([0.0, -np.pi, 0.0]),
    np.array([0.0, 0.0, -np.pi])
])


def rotation_matrix(ang):
    s0 = np.sin(ang[0])
    c0 = np.cos(ang[0])
    s1 = np.sin(ang[1])
    c1 = np.cos(ang[1])
    s2 = np.sin(ang[2])
    c2 = np.cos(ang[2])
    return np.array([
        [c0 * c1 * c2 - s0 * s2, -c2 * s0 - c0 * c1 * s2, -c0 * s1],
        [c0 * s2 + c1 * c2 * s0, c0 * c2 - c1 * s0 * s2, -s0 * s1],
        [c2 * s1, -s1 * s2, c1]
    ], dtype=np.int32)


def get_point_group(schoenflies_symbol):
    if schoenflies_symbol == "O":
        g = [rotation_matrix(ea) for ea in euler_angles_O]
        return np.array(g)
    elif schoenflies_symbol == "Oh":
        g = [rotation_matrix(ea) for ea in euler_angles_O]
        g += [-rotation_matrix(ea) for ea in euler_angles_O]
        return np.array(g)
    else:
        symbols = ["O", "Oh"]
        raise ValueError(f"Unrecognized Schoenflies symbol '{schoenflies_symbol}'."
                         f" Available symbols: {symbols}")


def cayley_table(g):
    n = len(g)
    table = np.zeros((n, n), dtype=int)
    for i, gi in enumerate(g):
        for j, gj in enumerate(g):
            h = gi @ gj
            for k, gk in enumerate(g):
                if np.allclose(h, gk):
                    table[i, j] = k
                    break
    return table


def dixon_matrix(rep):
    n = len(rep[0])
    for r in range(n):
        for s in range(n):
            H_rs = np.zeros((n, n), dtype=np.complex128)
            if r == s:
                H_rs[r, s] = 1.0
            elif r > s:
                H_rs[r, s] = 1.0
                H_rs[s, r] = 1.0
            else: # r < s
                H_rs[r, s] = 1.0j
                H_rs[s, r] = -1.0j
            H = sum([np.transpose(np.conj(r)) @ H_rs @ r for r in rep]) / n
            
            if not np.allclose(H[0, 0] * np.eye(n), H):
                return H # rep is reducible
    return np.eye(n, dtype=np.complex128) # rep is irrep


def decompose(rep, H):
    eigvals, eigvecs = np.linalg.eigh(H)
    eigspaces = []
    for eigval, eigvec in zip(eigvals, eigvecs.T):
        new_space = True
        for i, (ei, _) in enumerate(eigspaces):
            if np.isclose(eigval, ei):
                eigspaces[i][1].append(eigvec)
                new_space = False
                break
        if new_space:
            eigspaces += [(eigval, [eigvec])]

    reps = []
    for _, list_eigvecs in eigspaces:
        transformation = np.linalg.qr(np.transpose(list_eigvecs))[0]
        r = np.einsum(
            "li,klm,mj->kij", np.conj(transformation), rep, transformation, optimize="greedy"
        )
        reps += [r]
    return reps


def inner_product(rep0, rep1):
    assert len(rep0) == len(rep1)
    return sum(
        [np.conj(np.trace(r0)) * np.trace(r1) for r0, r1 in zip(rep0, rep1)]
    ) / len(rep0)


def is_equivalent(rep0, rep1):
    if np.isclose(inner_product(rep0, rep1), 1.0):
        return True
    return False


def irr_decompose(rep):
    if np.isclose(inner_product(rep, rep), 1.0):
        return [rep]
    else:
        irreps = []
        H = dixon_matrix(rep)
        reps = decompose(rep, H)
        for r in reps:
            rr = irr_decompose(r)
            for r0 in rr:
                is_unique = True
                for r1 in irreps:
                    if is_equivalent(r0, r1):
                        is_unique = False
                        break
                if is_unique:
                    irreps += [r0]
        idx = np.argsort([len(irrep[0]) for irrep in irreps])
        return [irreps[i] for i in idx]

    
def get_multiplicity(rep, irrep):
    m = inner_product(rep, irrep)
    assert np.isclose(m.real, np.around(m.real))
    assert np.isclose(m.imag, 0.0)
    return np.around(m.real).astype(int)


def project_to_irrep(rep, irrep):
    assert rep.shape[0] == irrep.shape[0]
    N = rep.shape[1]
    dtype = np.common_type(rep, irrep)
    p = np.zeros((N, N), dtype=dtype)
    m = get_multiplicity(rep, irrep)
    if m == 0:
        return p
    for ri, ii in zip(rep, irrep):
        p += np.conj(np.trace(ii)) * ri

    return p * irrep.shape[1] / irrep.shape[0]


class point_group:
    def __init__(self, schoenflies_symbol):
        self._fundamental = get_point_group(schoenflies_symbol)
        self._regular = None

        self._irreps = None

        self._table = None
        self.order = len(self._fundamental)

    def cayley_table(self):
        if self._table is None:
            self._table = cayley_table(self._fundamental)
        return self._table

    def invers_idx(self, idx):
        return np.where(self.cayley_table()[idx] == 0)[0][0]

    def trivial_representation(self):
        return np.ones((self.order, 1, 1), dtype=np.int32)
    
    def fundamental_representation(self):
        return self._fundamental
    
    def regular_representation(self):
        if self._regular is None:
            self._regular = np.zeros((self.order, self.order, self.order), dtype=np.int32)
            for i in range(self.order):
                for j in range(self.order):
                    self._regular[i, self.cayley_table()[i, j], j] = 1
        return self._regular

    def is_representation(self, rep):
        for i, ri in enumerate(rep):
            for j, rj in enumerate(rep):
                g = ri @ rj
                h = rep[self.cayley_table()[i, j]]
                if not np.allclose(g, h):
                    return False
        return True

    def is_irrep(self, rep):
        if self.is_representation(rep) and np.isclose(inner_product(rep, rep), 1.0):
            return True
        return False

    def irreps(self):
        if self._irreps is None:
            self._irreps = irr_decompose(self.regular_representation())
            for irrep in self._irreps:
                assert self.is_irrep(irrep)
        return self._irreps
