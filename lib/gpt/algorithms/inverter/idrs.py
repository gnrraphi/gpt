#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Raphael Lehner (raphael.lehner@physik.uni-regensburg.de)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#    Implementation of IDR(s) algorithm by Sonneveld and van Gijzen,
#    see https://doi.org/10.1137/070685804
#
import gpt as g
import numpy as np
from gpt.algorithms import base_iterative


class idrs(base_iterative):
    @g.params_convention(eps=1e-15, maxiter=1000, s=3, P=None, checkres=True)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.s = params["s"]
        self.P = params["P"]
        self.checkres = params["checkres"]

    def calc_res(self, mat, psi, mmp, src, r):
        mat(mmp, psi)
        return g.axpy_norm2(r, -1.0, mmp, src)

    def __call__(self, mat):

        vector_space = None
        if type(mat) == g.matrix_operator:
            vector_space = mat.vector_space
            mat = mat.mat

        @self.timed_function
        def inv(psi, src, t):
            assert src != psi

            # timing
            t("setup")

            # parameters
            s = self.s

            # tensors
            M = np.zeros((s, s), np.complex128)
            m = np.zeros((s, 1), np.complex128)

            # fields
            mmp, r, v, w = (
                g.copy(src), g.copy(src), g.copy(src), g.copy(src)
            )
            dR = [g.lattice(src) for i in range(s)]
            dX = [g.lattice(src) for i in range(s)]

            # initial residual
            mat(mmp, psi)
            r2 = g.axpy_norm2(r, -1.0, mmp, src)

            # source
            ssq = g.norm2(src)
            if ssq == 0.0:
                assert r2 != 0.0
                ssq = r2

            # target residual
            rsq = self.eps ** 2.0 * ssq

            # initialize shadow space
            t("shadow space")
            if self.P is None:
                P = [g.lattice(src) for i in range(s)]
                g.random("idrs").cnormal(P)
            else:
                P = self.P
                assert len(P) == s

            # produce start vectors in Sonneveld space G(0)
            t("start vectors")
            for k in range(s):
                mat(mmp, r)
                om = g.inner_product(mmp, r) / g.norm2(mmp)
                dX[k] @= g.eval(om * r)
                dR[k] @= g.eval(-om * mmp)
                psi += dX[k]
                r += dR[k]

            for k in range(self.maxiter):

                # index oldest vectors
                idx = (-k - 1) % s

                for l in range(s + 1):

                    # solve P^\dag * dR * c = P^\dag * r
                    t("solve")
                    for i, p in enumerate(P):
                        m[i] = g.inner_product(p, r)
                        for j, dr in enumerate(dR):
                            M[i, j] = g.inner_product(p, dr)
                    c = np.linalg.solve(M, m)

                    t("linear algebra")
                    g.linear_combination(mmp, dR, -c.T)
                    g.axpy(v, 1.0, mmp, r)
                    if l == 0:

                        # enter G(j+1)
                        t("mat")
                        mat(w, v)

                        t("linear algebra")
                        om = g.inner_product(w, v) / g.norm2(w)
                        g.axpy(dR[idx], -om, w, mmp)
                        g.linear_combination(mmp, dX, -c.T)
                        g.axpy(dX[idx], om, v, mmp)
                    else:

                        # subsequent vectors in G(j+1)
                        g.linear_combination(mmp, dX, -c.T)
                        g.axpy(dX[idx], om, v, mmp)

                        t("mat")
                        mat(dR[idx], g.eval(-dX[idx]))

                    t("linear algebra")
                    psi += dX[idx]
                    r += dR[idx]

                    t("residual")
                    r2 = g.norm2(r)

                    t("other")
                    self.log_convergence((k, l), r2, rsq)
                    if r2 <= rsq:
                        msg = f"converged at iteration {k}"
                        msg += f";  computed squared residual {r2:e} / {rsq:e}"
                        if self.checkres:
                            res = self.calc_res(mat, psi, mmp, src, r)
                            msg += f";  true squared residual {res:e} / {rsq:e}"
                        self.log(msg)
                        return

            msg = f"NOT converged in {k} iterations"
            msg += f";  computed squared residual {r2:e} / {rsq:e}"
            if self.checkres:
                res = self.calc_res(mat, psi, mmp, src, r)
                msg += f";  true squared residual {res:e} / {rsq:e}"
            self.log(msg)

        return g.matrix_operator(
            mat=inv, inv_mat=mat, accept_guess=(True, False), vector_space=vector_space
        )
