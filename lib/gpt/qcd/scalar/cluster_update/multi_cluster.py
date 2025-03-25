#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2025  Raphael Lehner
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
import gpt
import cgpt
import numpy as np


# multi-cluster update for non-linear sigma model
class multi_cluster:
    def __init__(self, rng, kappa, alpha):

        self.rng = rng
        self.k = kappa
        self.a = alpha

    def __call__(self, phi):
        verbose = gpt.default.is_verbose("multi_cluster")

        # Initialize random probe spin
        t0 = gpt.time()
        N = phi.otype.shape[0]
        r = np.array([self.rng.normal().real for i in range(N)])
        r /= np.linalg.norm(r)
        _probe = gpt.vreal(r, N)
        probe = gpt.lattice(phi)
        probe[:] = _probe

        # Projection of site along the probing spin
        grid = phi.grid
        point_probe = gpt.real(grid)
        point_probe @= gpt.trace(probe * gpt.adj(phi))

        # Compute probability of bonding
        nd = phi.grid.nd
        random = [gpt.real(grid) for i in range(nd)]
        self.rng.uniform_real(random, min=0, max=1)

        # Check if probability of bonding is accepted
        prob = gpt.real(grid)
        one = gpt.real(grid)
        one[:] = 1
        bond = [gpt.real(grid) for i in range(nd)]
        for mu in range(nd):
            prob @= one - gpt.component.exp(
                -4.0 * self.k * point_probe * gpt.cshift(point_probe, mu, 1)
            )
            bond[mu] @= random[mu] < prob
        t1 = gpt.time()

        if verbose:
            gpt.message(
                f"multi_cluster: bonds took {t1-t0} s;  acceptance rate = {(sum(gpt.norm2(bond)) / (nd * grid.fsites))}"
            )

        t0 = gpt.time()
        p = {}
        p["alpha"] = self.a
        p["probe"] = r
        p["bond"] = bond
        nc, nf = cgpt.multi_cluster(self.rng.obj, phi.v_obj[0], point_probe.v_obj[0], p)
        t1 = gpt.time()

        if verbose:
            msg = f"multi_cluster: cluster update took {t1-t0} s;  "
            msg += f"acceptance rate = {(nf / nc)};  average size = {(grid.fsites / nc):e}"
            gpt.message(msg)
