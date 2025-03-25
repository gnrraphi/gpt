/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020-25  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2025     Raphael Lehner

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
long find(const long idx, std::vector<long>& parent) {
    if (parent[idx] != idx)
        parent[idx] = find(parent[idx], parent); // Path compression
    return parent[idx];
}

class cgpt_random_engine_base {
public:
  virtual ~cgpt_random_engine_base() { };
  virtual PyObject* sample(PyObject* param) = 0;
  virtual void multi_cluster(cgpt_Lattice_base* s, cgpt_Lattice_base* pp, PyObject* _param, long & nc, long & nf) = 0;
  virtual double test_U01() = 0;
  virtual uint32_t test_bits() = 0;
};

template<typename cgpt_rng_engine>
class cgpt_random_engine : public cgpt_random_engine_base {
 public:
  std::string _seed_str;
  cgpt_rng_engine cgpt_srng;

  struct index_t {
    int i,o; // adjust once grid adjusts
  };
  
  struct prng_t {
    std::vector<cgpt_rng_engine*> rng;
    std::vector<long> hash;
    std::vector<uint64_t> seed;
    std::vector< std::vector< index_t > > samples;
    long block, sites;
    std::string grid_tag;
  };
  
  std::map<GridBase*,prng_t> cgpt_prng;

  std::vector<uint64_t> str_to_seed(const std::string & seed_str) {
    std::vector<uint64_t> r;
    for (auto x : seed_str)
      r.push_back(x);
    return r;
  }

  cgpt_random_engine(const std::string & seed_str) : _seed_str(seed_str), cgpt_srng(str_to_seed(seed_str)) {
  }
  
  virtual ~cgpt_random_engine() {
    for (auto & x : cgpt_prng)
      for (auto & y : x.second.rng)
	delete y;
  }

  virtual PyObject* sample(PyObject* _param) {

    double t0 = cgpt_time();
    
    ASSERT(PyDict_Check(_param));
    std::string dist = get_str(_param,"distribution");
    GridBase* grid = 0;
    std::vector<cgpt_Lattice_base*> lattices;
    long n_virtual;
    PyObject* _lattices = PyDict_GetItemString(_param,"lattices");
    if (_lattices) {
      n_virtual = cgpt_basis_fill(lattices,_lattices);
      ASSERT(lattices.size() > 0);
      grid = lattices[0]->get_grid();
      for (size_t i=1;i<lattices.size();i++)
	ASSERT(grid == lattices[i]->get_grid());
    }

    prng_t & prng = cgpt_prng[grid];
    if (prng.grid_tag.size() == 0) {
      prng.grid_tag = cgpt_grid_cache_tag[grid];
    }
    if (prng.grid_tag != cgpt_grid_cache_tag[grid]) {
      // Grid changed! clear cache
      prng = prng_t();
    }
    std::vector<uint64_t> & seed = prng.seed;
    if (seed.size() == 0) {
      seed = str_to_seed(_seed_str);
      if (grid) {
	for (auto x : grid->_fdimensions)
	  seed.push_back(x);
	for (auto x : grid->_gdimensions)
	  seed.push_back(x);
      }
    }

    double t1 = cgpt_time();
    //std::cout << GridLogMessage << "prep " << t1-t0 << std::endl;

    // always generate in double first regardless of type casting to ensure that numbers are the same up to rounding errors
    // (rng could use random bits to result in different next float/double sampling)
    if (dist == "normal") {
      cgpt_normal_distribution distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else if (dist == "cnormal") {
      cgpt_cnormal_distribution distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else if (dist == "uniform_real") {
      cgpt_uniform_real_distribution distribution(get_float(_param,"min"),get_float(_param,"max"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else if (dist == "uniform_int") {
      cgpt_uniform_int_distribution distribution(get_int(_param,"min"),get_int(_param,"max"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else if (dist == "zn") {
      cgpt_zn_distribution distribution(get_int(_param,"n"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else {
      ERR("Unknown distribution: %s", dist.c_str());
    }

  }

  virtual void multi_cluster(cgpt_Lattice_base* s, cgpt_Lattice_base* pp, PyObject* _param, long & nc, long & nf) {

    ASSERT(PyDict_Check(_param));

    RealD alpha = get_float(_param,"alpha");

    GridBase* grid = s->get_grid();
    int nd = grid->Nd();
    long sites = (long)grid->_osites * (long)grid->_isites;

    // Data layout source
    long Nsimd_s, word_s, simd_word_s;
    s->describe_data_layout(Nsimd_s, word_s, simd_word_s);
    long osite_stride_s = (long)word_s * Nsimd_s / simd_word_s;
    long obj_stride_s = (long)Nsimd_s;
    int isite_stride_s = 1;
    long n_per_site_s = (long)word_s / simd_word_s;

    // Data layout point_probe
    long Nsimd_pp, word_pp, simd_word_pp;
    pp->describe_data_layout(Nsimd_pp, word_pp, simd_word_pp);
    long osite_stride_pp = (long)word_pp * Nsimd_pp / simd_word_pp;
    long obj_stride_pp = (long)Nsimd_pp;
    int isite_stride_pp = 1;
    long n_per_site_pp = (long)word_pp / simd_word_pp;

    std::vector<cgpt_Lattice_base*> bond;
    long n_virtual;
    PyObject* _bond = PyDict_GetItemString(_param,"bond");
    n_virtual = cgpt_basis_fill(bond,_bond);
    int n_bond = bond.size();
    ASSERT(n_virtual == 1);
    ASSERT(n_bond == nd);

    PyObject* _probe = PyDict_GetItemString(_param,"probe");
    RealD* probe;
    int n_probe;
    cgpt_numpy_import_vector(_probe,probe,n_probe);
    ASSERT(n_probe == n_per_site_s);

    std::vector<PyObject*> view(nd);
    for (int i=0;i<nd;i++)
        view[i] = bond[i]->memory_view(mt_host);

    std::vector<ComplexD*> b(nd);
    for (int i=0;i<nd;i++)
        b[i] = (ComplexD*)PyMemoryView_GET_BUFFER(view[i])->buf; // TODO: Change ComplexD to complex_t

    PyObject* view_s = s->memory_view(mt_host);
    ComplexD* s_pointer = (ComplexD*)PyMemoryView_GET_BUFFER(view_s)->buf;

    PyObject* view_pp = pp->memory_view(mt_host);
    ComplexD* pp_pointer = (ComplexD*)PyMemoryView_GET_BUFFER(view_pp)->buf;

    // Initialize cluster-related data structures
    std::vector<long> parent(sites); // Representation of a element
    std::vector<long> rank(sites, 0); // Rank of a tree
    std::vector<RealD> cluster_pp(sites, 0.0); // Sum of point * probe of clusters
    std::vector<long> cluster_flip(sites, 0); // Will flip site if 1

    for (long idx=0; idx<sites; idx++) {
        parent[idx] = idx;
    }

    std::vector<long> pp_lattice_idx(sites);
    std::vector<long> s_lattice_idx(sites);

    for (long idx=0; idx<sites; idx++) {
        Coordinate coor(nd);
        Lexicographic::CoorFromIndex(coor,idx,grid->_fdimensions);
        auto ii = grid->iIndex(coor);
        auto oo = grid->oIndex(coor);
        long lattice_idx = oo * osite_stride_pp + ii * isite_stride_pp;
        pp_lattice_idx[idx] = lattice_idx;
        long lattice_idx_s = oo * osite_stride_s + ii * isite_stride_s;
        s_lattice_idx[idx] = lattice_idx_s;
        for (int mu=0; mu<nd; mu++) {
            if ((b[mu][lattice_idx]).real() == 1.0) {
                //  Get neighboring site coordinate
                Coordinate ncoor(nd);
                for (int nu=0; nu<nd; nu++) {
                    if (mu == nu) {
                        ncoor[nu] = (coor[nu] + 1) % grid->_fdimensions[nu];
                    } else {
                        ncoor[nu] = coor[nu];
                    }
                }
                int nsite_idx; // Neighboring site index
                Lexicographic::IndexFromCoor(ncoor, nsite_idx, grid->_fdimensions);
                long root = find(idx, parent);
                long nsite_root = find(nsite_idx, parent);
                if (root != nsite_root) {
                    // Union of small set with larger one
                    if (rank[root] > rank[nsite_root]) {
                        parent[nsite_root] = root;
                    } else if (rank[root] < rank[nsite_root]) {
                        parent[root] = nsite_root;
                    } else {
                        parent[nsite_root] = root;
                        rank[root]++;
                    }
                }
            }
        }
    }

    std::vector<long> cluster_idx(sites); // Root index of element
    std::vector<long> root_idx(sites); // List of all roots

    long n_cluster = 0; // Number of clusters
    for (long idx=0; idx<sites; idx++) {
        long root = find(idx, parent);
        cluster_idx[idx] = root;
        cluster_pp[root] += (pp_pointer[pp_lattice_idx[idx]]).real();
        if (idx == root) {
            root_idx[n_cluster] = root;
            n_cluster++;
        }
    }
    root_idx.resize(n_cluster);
    nc = n_cluster;

    cgpt_uniform_real_distribution ur_dis(0.0,1.0); // Random number generator for uniform distribution from 0 to 1
    long n_flip = 0; // Number of flipped clusters
    for (long idx=0; idx<n_cluster; idx++) {

        long root = root_idx[idx];
        RealD aux_flip = cluster_pp[root] * alpha * probe[0];
        RealD p_flip = 1.0 / (1.0 + ::exp(2.0 * aux_flip));
        RealD ur_flip = ur_dis(cgpt_srng);
        if (ur_flip < p_flip) {
            cluster_flip[root] = 1;
            n_flip++;
        }
    }
    nf = n_flip;

    for (long idx=0; idx<sites; idx++) {
        if (cluster_flip[cluster_idx[idx]] == 1) {
            for (long j=0; j<n_per_site_s; j++) {
                RealD aux_RealD = 2.0 * (pp_pointer[pp_lattice_idx[idx]]).real() * probe[j];
                s_pointer[s_lattice_idx[idx]+j*obj_stride_s] -= ComplexD(aux_RealD, 0.0);
             }
         }
    }

    for (int i=0;i<nd;i++)
        Py_DECREF(view[i]); // close views
    Py_DECREF(view_s);
    Py_DECREF(view_pp);

  }

  virtual double test_U01() {
    return cgpt_srng.get_double();
  }

  virtual uint32_t test_bits() {
    return cgpt_srng.get_uint32_t();
  }

};