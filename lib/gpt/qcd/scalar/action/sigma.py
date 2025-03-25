#
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
import gpt as g
from gpt.core.group import differentiable_functional
import numpy

# linear sigma model

# S[phi] = -2 * kappa * sum_x,n,mu phi_n(x)^dag * phi_n(x+mu)
#          + \sum_x phi2(x) + lambda / 4! * sum_x (phi2(x) - 1)^2
#          - alpha * \sum_x phi_0(x)
# phi2(x) = sum_n |phi_n(x)|^2
# sum_x (phi2(x) - 1)^2 = sum_x |phi2(x)|^2 - 2 sum_x phi2(x) + vol
class linear_sigma_model(differentiable_functional):
    def __init__(self, kappa, l, alpha):
        self.kappa = kappa
        self.l = l
        self.alpha = alpha

        self.__name__ = f"linear_sigma_model({self.kappa},{self.l},{self.alpha})"

    def kappa_to_mass2(self, k, l, D):
        return (1.0 - 2.0 * l / 24.0) / k - 2.0 * D

    def kappa_to_lambda(self, k, l):
        return l / (2.0 * k) ** 2.0

    def kappa_to_alpha(self, k, a):
        return a / numpy.sqrt(2.0 * k)

    def __call__(self, phi):
        J = g.lattice(phi)
        J[:] = 0.0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)

        act = -2.0 * self.kappa * g.inner_product(J, phi).real

        p2 = g.norm2(phi)
        act += p2

        if self.l != 0.0:
            phi2 = g.real(phi.grid)
            # TO DO: replace with g.adj(phi) * phi
            phi2 @= g.trace(phi * g.adj(phi))

            p4 = g.norm2(phi2)
            act += self.l / 24.0 * (p4 - 2.0 * p2 + phi.grid.fsites)

        if self.alpha != 0.0:
            N = J.otype.shape[0]
            J_comp = g.vreal([self.alpha] + [0] * (N - 1), N)
            J[:] = J_comp
            act -= g.inner_product(J, phi).real

        return act

    @differentiable_functional.single_field_gradient
    def gradient(self, phi):
        J = g.lattice(phi)
        J[:] = 0.0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)
            J += g.cshift(phi, mu, -1)

        frc = g.lattice(phi)
        frc @= -2.0 * self.kappa * J

        frc += 2.0 * phi

        if self.l != 0.0:
            phi2 = g.real(phi.grid)
            # TO DO: replace with g.adj(phi) * phi
            phi2 @= g.trace(phi * g.adj(phi))

            frc += self.l / 6.0 * phi2 * phi
            frc -= self.l / 6.0 * phi

        if self.alpha != 0.0:
            N = J.otype.shape[0]
            J_comp = g.vcomplex([self.alpha] + [0] * (N - 1), N)
            J[:] = J_comp
            frc -= J

        return frc


# non-linear sigma model

# S[phi] = -2.0 * kappa * sum_x,n,mu phi_n(x)^dag * phi_n(x+mu)
#          + vol - alpha * \sum_x phi_0(x)
#          - 2.0 * kappa * omega * sum_x,n,mu phi_n(x)^dag * phi_n(x+2*mu)
# with sum_n |phi_n(x)|^2 = 1
#
class non_linear_sigma_model(differentiable_functional):
    def __init__(self, kappa, alpha, omega=0.0):
        self.kappa = kappa
        self.alpha = alpha
        self.omega = omega

        self.__name__ = f"non_linear_sigma_model({self.kappa},{self.alpha})"

    def __call__(self, phi):
        J = g.lattice(phi)
        J[:] = 0.0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)

        act = -2.0 * self.kappa * g.inner_product(J, phi).real

        # omit constant contribution
        # act += phi.grid.fsites

        if self.omega != 0.0:
            J[:] = 0.0
            for mu in range(phi.grid.nd):
                J += g.cshift(phi, mu, +2)
            act -= 2.0 * self.kappa * self.omega * g.inner_product(J, phi).real

        if self.alpha != 0.0:
            N = J.otype.shape[0]
            J_comp = g.vreal([self.alpha] + [0] * (N - 1), N)
            J[:] = J_comp
            act -= g.inner_product(J, phi).real

        return act

    @differentiable_functional.single_field_gradient
    def gradient(self, phi):
        J = g.lattice(phi)
        J[:] = 0.0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)
            J += g.cshift(phi, mu, -1)

        frc = g.lattice(phi)
        frc @= -2.0 * self.kappa * J

        if self.omega != 0.0:
            J[:] = 0.0
            for mu in range(phi.grid.nd):
                J += g.cshift(phi, mu, +2)
                J += g.cshift(phi, mu, -2)
            frc -= 2.0 * self.kappa * self.omega * J

        if self.alpha != 0.0:
            N = J.otype.shape[0]
            J_comp = g.vcomplex([self.alpha] + [0] * (N - 1), N)
            J[:] = J_comp
            frc -= J

        frc -= g.trace(frc * g.adj(phi)) * phi

        return frc

    # https://arxiv.org/abs/1102.1852
    def constrained_leap_frog(self, eps, z, mom_z):
        # TO DO: replace with g.adj(v1) * v2
        def dot(v1, v2):
            return g.trace(v2 * g.adj(v1))

        n = g.real(z.grid)
        n @= g.component.sqrt(g.component.real(dot(mom_z, mom_z)))

        # z'      =  cos(alpha) z + (1/|pi|) sin(alpha) mom_z
        # mom_z'  = -|pi| sin(alpha) z + cos(alpha) mom_z
        # alpha = eps |pi|
        _z = g.lattice(z)
        _z @= z

        cos = g.real(z.grid)
        cos @= g.component.cos(eps * n)

        sin = g.real(z.grid)
        sin @= g.component.sin(eps * n)

        z @= cos * _z + g(g.component.inv(n) * sin) * mom_z
        mom_z @= -g(n * sin) * _z + cos * mom_z
        del _z, cos, sin, n

    # https://arxiv.org/abs/1102.1852
    def draw(self, field, rng, constraint=None):
        if constraint is None:
            z = field
            rng.element(z)
            n = g.component.real(g.trace(z * g.adj(z)))
            z @= z * g.component.inv(g.component.sqrt(n))
        else:
            mom_z = field
            z = constraint
            rng.normal_element(mom_z)
            mom_z @= mom_z - g(z * g.adj(z)) * mom_z
