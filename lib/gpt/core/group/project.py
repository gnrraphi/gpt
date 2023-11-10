import gpt
import cgpt
import numpy as np
import itertools as it


def get_relative_shells(n, nd, stride, fdim, o):
    shells = []
    shells_mask = [-1 for i in range(n)]
    ns = 0
    for idx in range(n):
        if shells_mask[idx] == -1:
            c_rel = [0 for i in range(nd)]
            for i in range(nd-1, -1, -1):
                c = idx // stride[i]
                idx -= c * stride[i]
                fhalf = fdim[i] // 2
                c_rel[i] = (c + fdim[i] - o[i] + fhalf) % fdim[i] - fhalf
            pts_shell = []
            pts = list(set([p for p in it.permutations(c_rel)]))
            for sign in it.product([-1, 1], repeat=nd):
                for pt in pts:
                    pts_shell += [tuple([s * p for s, p in zip(sign, pt)])]
            for pt in pts_shell:
                sidx = 0
                for fi, si, oi, pi in zip(fdim, stride, o, pt):
                    sidx += si * ((pi - fi + oi) % fi)
                shells_mask[int(sidx)] = ns
            shells += [sorted(pts_shell)]
            ns += 1
    return shells, shells_mask


def get_shell_representation(shell, g, inverse_idx):
    rep = []
    n = len(shell)
    for gi in g:
        m = np.zeros((n, n), dtype=np.int32)
        for j, bj in enumerate(shell):
            bij = gi @ bj
            for k, bk in enumerate(shell):
                if not np.any(bij - bk):
                    m[k, j] = 1
                    break
        rep += [m]
    return np.array(rep)


class project_irrep:
    def __init__(self, grid, grp, irrep, dimensions=[0, 1, 2], origin=None):
        fdim = [grid.fdimensions[d] for d in dimensions]
        if origin is not None:
            o = origin
        else:
            o = [-0.5 * ((f - 1) % 2) for f in fdim]

        # check if representation is well defined
        assert len(dimensions) == len(o)
        assert all(f == fdim[0] for f in fdim)
        assert all([(2.0 * oi - fdim[0]) % 2.0 == 1.0 for oi in o]) 
        # assert is_checkerboarded == False
        assert all(oi == o[0] for oi in o) # TODO: fix or why?

        # check if rep is well defined
        assert len(dimensions) == len(grp._fundamental[0])

        inverse_idx = [grp.invers_idx(i) for i in range(grp.order)]

        t0 = gpt.time()
        nd = len(dimensions)
        stride = [1]
        for i in range(nd-1):
            stride += [stride[i] * fdim[i]]
        n = stride[nd-1] * fdim[nd-1]

        shells, shells_mask = get_relative_shells(n, nd, stride, fdim, o)
        shells_scale = [[] for i in range(n)]
        shells_idx = [[] for i in range(n)]
        shells_deg = [0 for i in range(n)]
        project = {}
        for shell in shells:
            ns = len(shell)
            if 0.0 in shell[0]:
                ns_idx *= -1 * ns
            else:
                ns_idx = ns
            if ns_idx not in project:
                rep = get_shell_representation(shell, grp._fundamental, inverse_idx)
                project[ns_idx] = gpt.core.group.project_to_irrep(rep, irrep)
            shell_idx = []
            for pt in shell:
                sidx = 0
                for fi, si, oi, pi in zip(fdim, stride, o, pt):
                    sidx += si * ((pi - fi + oi) % fi)
                shell_idx += [int(sidx)]
            for j, idx in enumerate(shell_idx):
                shells_idx[idx] = shell_idx
                shells_deg[idx] = ns
                shells_scale[idx] = project[ns_idx][j]
        t1 = gpt.time()
        gpt.message(f"time shells: {t1-t0} s")

        self.dims = np.array(dimensions, np.int32)
        self.sidx = np.concatenate(shells_idx, dtype=np.int32)
        self.scale = np.concatenate(shells_scale, dtype=np.complex128)
        self.deg = np.array(shells_deg, dtype=np.int32)

        self.mat = gpt.matrix_operator(mat=self.__call__)

        def reduce(dst, src):
            dst @= src - self.mat * src
        self.mat_reduce = gpt.matrix_operator(mat=reduce)

    def __call__(self, d, s):
        s = gpt.eval(s)
        assert len(d.otype.v_idx) == len(s.otype.v_idx)
        for i in d.otype.v_idx:
            cgpt.lattice_project_irrep(
                d.v_obj[i], s.v_obj[i],
                self.dims, self.sidx, self.scale, self.deg
            )


class project_trivial:
    def __init__(self, grid, dimensions=[0, 1, 2], origin=None):
        fdim = [grid.fdimensions[d] for d in dimensions]
        if origin is not None:
            o = origin
        else:
            o = [-0.5 * ((f - 1) % 2) for f in fdim]

        # check if representation is well defined
        assert len(dimensions) == len(o)
        assert all(f == fdim[0] for f in fdim)
        assert all([(2.0 * oi - fdim[0]) % 2.0 == 1.0 for oi in o]) 
        # assert is_checkerboarded == False

        nd = len(dimensions)
        stride = [1]
        for i in range(nd-1):
            stride += [stride[i] * fdim[i]]
        n = stride[nd-1] * fdim[nd-1]

        shells, shells_mask = get_relative_shells(n, nd, stride, fdim, o)

        self.dims = np.array(dimensions, np.int32)
        self.sidx = np.array(shells_mask, np.int32)
        self.scale = np.array([1.0 / len(shell) for shell in shells], np.complex128)
        self.Ns = len(shells)

        self.mat = gpt.matrix_operator(mat=self.__call__)

        def reduce(dst, src):
            dst @= src - self.mat * src
        self.mat_reduce = gpt.matrix_operator(mat=reduce)

    def __call__(self, d, s):
        s = gpt.eval(s)
        assert len(d.otype.v_idx) == len(s.otype.v_idx)
        for i in d.otype.v_idx:
            cgpt.lattice_project_trivial(
                d.v_obj[i], s.v_obj[i],
                self.dims, self.sidx, self.scale, self.Ns,
            )
