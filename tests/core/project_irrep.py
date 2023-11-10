import gpt as g
import numpy as np
import sys


rng = g.random("test")
grid = g.grid([32, 32, 32, 8], g.double)
a = rng.cnormal(g.vcolor(grid))

trivial_0 = g.group.project_trivial(grid)
trivial_1 = g.group.project_trivial(grid, origin=[1.5, 1.5, 1.5])
trivial_2 = g.group.project_trivial(grid, dimensions=[0, 2])

grp = g.core.group.point_group("O")
irreps = grp.irreps()

irrep_0 = g.group.project_irrep(grid, grp, irreps[0])
irrep_1 = g.group.project_irrep(grid, grp, irreps[3])
irrep_2 = g.group.project_irrep(grid, grp, irreps[0], origin=[1.5, 1.5, 1.5])


def test(p):
    mat = p.mat
    mat_reduce = p.mat_reduce

    b = g.lattice(a)
    c = g.lattice(a)

    # P * P = P
    b @= mat * a
    c @= mat * mat * a
    assert g.norm2(b-c) / g.norm2(b) < 1e-28

    # P_r * P_r = P_r
    b @= mat_reduce * a
    c @= mat_reduce * mat_reduce * a
    assert g.norm2(b-c) / g.norm2(b) < 1e-28

    # P * P_r = 0
    b @= mat * mat_reduce * a
    assert g.norm2(b) / g.norm2(a) < 1e-28

    # P + P_r = 1
    b @= mat * a + mat_reduce * a
    assert g.norm2(b-a) / g.norm2(a) < 1e-28


for p in [trivial_0, trivial_1, trivial_2, irrep_0, irrep_1, irrep_2]:
    test(p)
