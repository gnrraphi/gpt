/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)


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

template<typename T>
void cgpt_project_irrep(Lattice<T>& dst,Lattice<T>& src,int* dims,int* sidx,ComplexD* scale,int* deg, int nd) {

  GridBase* grid = dst.Grid();
  conformable(grid, src.Grid());

  dst.Checkerboard() = src.Checkerboard();

  autoView(dst_v, dst, AcceleratorWriteDiscard);
  autoView(src_v, src, AcceleratorRead);

  auto dst_p = &dst_v[0];
  auto src_p = &src_v[0];

  int osites = grid->oSites();
  int nsimd = grid->Nsimd();
  int ndim = grid->Nd();
  int ndo = ndim - nd;

  Vector<bool> dims_mask(ndim,false);
  for (int idx=0; idx < nd; idx++) {
      dims_mask[dims[idx]] = true;
  }

  Coordinate Dims;
  Coordinate Dimso;
  for (int idx=0; idx < ndim; idx++) {
      if (dims_mask[idx]) {
          Dims.push_back(idx);
      } else {
          Dimso.push_back(idx);
      }
  }

  Coordinate stride(nd);
  stride[0] = 1;
  for (int idx=0; idx < nd-1; idx++) {
      stride[idx+1] = stride[idx] * grid->_rdimensions[Dims[idx]];
   }

  Coordinate strideo(ndo);
  strideo[0] = 1;
  for (int idx=0; idx < ndo-1; idx++) {
      strideo[idx+1] = strideo[idx] * grid->_rdimensions[Dimso[idx]];
   }

  int size = stride[nd-1] * grid->_rdimensions[Dims[nd-1]];
    
  Vector<int> _Deg(size+1);
  int* Deg =&_Deg[0];
  Deg[0] = 0;
  for(int idx=0; idx < size; idx++) {
      Deg[idx+1] = Deg[idx] + deg[idx];
  }

  int ns = Deg[size];
  Vector<int> _Sidx(ns);
  int* Sidx = &_Sidx[0];
  for(int idx=0; idx < ns; idx++) {
      Sidx[idx] = sidx[idx];
  }
    
  Vector<ComplexD> _Scale(ns);
  ComplexD* Scale = &_Scale[0];
  thread_for(idx, ns, {
      Scale[idx] = scale[idx];
    });

  bool simd = false;
  for (int idx=0; idx < nd; idx++) {
      if (grid->_simd_layout[Dims[idx]] != 1)
          simd = true;
  }
  if (simd)
    ERR("Not implemented yet");
    
  typedef decltype(coalescedRead(src_v[0])) CalcElem;

  accelerator_for(idx,osites,nsimd,{
      Coordinate ocoor(ndim);  
      Lexicographic::CoorFromIndex(ocoor,idx,grid->_rdimensions);
      int idx_s = 0;
      for (int i=0; i < nd; i++) {
          idx_s += ocoor[Dims[i]] * stride[i];
      }
      int idx_so = 0;
      for (int i=0; i < ndo; i++) {
          idx_so += ocoor[Dimso[i]] * strideo[i];
      }
      CalcElem sElem = Zero();
      for (int i=Deg[idx_s]; i < Deg[idx_s+1]; i++) { //Deg[-1] = n, Deg[i] = deg[i-1] + deg[i]
          sElem += Scale[i] * coalescedRead(src_p[Sidx[i]+idx_so*size]);
      }
      coalescedWrite(dst_p[idx], sElem);
  });
  
}

template<typename T>
void cgpt_project_trivial(Lattice<T>& dst,Lattice<T>& src,int* dims,int* sidx,ComplexD* scale,int ns,int nd) {

  GridBase* grid = dst.Grid();
  conformable(grid, src.Grid());

  dst.Checkerboard() = src.Checkerboard();

  autoView(dst_v, dst, AcceleratorWriteDiscard);
  autoView(src_v, src, AcceleratorRead);

  auto dst_p = &dst_v[0];
  auto src_p = &src_v[0];

  int osites = grid->oSites();
  int nsimd = grid->Nsimd();
  int ndim = grid->Nd();
  int ndo = ndim - nd;

  Vector<bool> dims_mask(ndim,false);
  for (int idx=0; idx < nd; idx++) {
      dims_mask[dims[idx]] = true;
  }

  Coordinate Dims;
  Coordinate Dimso;
  for (int idx=0; idx < ndim; idx++) {
      if (dims_mask[idx]) {
          Dims.push_back(idx);
      } else {
          Dimso.push_back(idx);
      }
  }

  Coordinate stride(nd);
  stride[0] = 1;
  for (int idx=0; idx < nd-1; idx++) {
      stride[idx+1] = stride[idx] * grid->_rdimensions[Dims[idx]];
   }

  Coordinate strideo(ndo);
  strideo[0] = 1;
  for (int idx=0; idx < ndo-1; idx++) {
      strideo[idx+1] = strideo[idx] * grid->_rdimensions[Dimso[idx]];
   }

  int size = stride[nd-1] * grid->_rdimensions[Dims[nd-1]];
  int sizeo = strideo[ndo-1] * grid->_rdimensions[Dimso[ndo-1]];

  Vector<int> _Sidx(size);
  int* Sidx = &_Sidx[0];
  for(int idx=0; idx < size; idx++) {
      Sidx[idx] = sidx[idx];
  }
    
  Vector<ComplexD> _Scale(ns);
  ComplexD* Scale = &_Scale[0];
  thread_for(idx, ns, {
      Scale[idx] = scale[idx];
    });

  typedef decltype(coalescedRead(src_v[0])) CalcElem;
  Vector<CalcElem> _sElem(ns*sizeo,Zero());
  CalcElem* sElem = &_sElem[0];

  bool simd = false;
  for (int idx=0; idx < nd; idx++) {
      if (grid->_simd_layout[Dims[idx]] != 1)
          simd = true;
  }
  if (simd)
    ERR("Not implemented yet");

  for(int idx=0; idx < osites; idx++) {
      Coordinate ocoor(ndim);  
      Lexicographic::CoorFromIndex(ocoor,idx,grid->_rdimensions);
      int idx_s = 0;
      for (int i=0; i < nd; i++) {
          idx_s += ocoor[Dims[i]] * stride[i];
      }
      int idx_so = 0;
      for (int i=0; i < ndo; i++) {
          idx_so += ocoor[Dimso[i]] * strideo[i];
      }
      sElem[Sidx[idx_s]+idx_so*ns] += coalescedRead(src_p[idx]);
  }

  accelerator_for(idx,osites,nsimd,{
      Coordinate ocoor(ndim);  
      Lexicographic::CoorFromIndex(ocoor,idx,grid->_rdimensions);
      int idx_s = 0;
      for (int i=0; i < nd; i++) {
          idx_s += ocoor[Dims[i]] * stride[i];
      }
      int idx_so = 0;
      for (int i=0; i < ndo; i++) {
          idx_so += ocoor[Dimso[i]] * strideo[i];
      }
      coalescedWrite(dst_p[idx], sElem[Sidx[idx_s]+idx_so*ns] * Scale[Sidx[idx_s]]);
  });
  
}

template<typename T>
void cgpt_scale_per_coordinate(Lattice<T>& dst,Lattice<T>& src,ComplexD* s,int dim) {

  GridBase* grid = dst.Grid();
  conformable(grid, src.Grid());

  dst.Checkerboard() = src.Checkerboard();

  int L = grid->_gdimensions[dim];
    
  autoView(dst_v, dst, AcceleratorWriteDiscard);
  autoView(src_v, src, AcceleratorRead);

  auto dst_p = &dst_v[0];
  auto src_p = &src_v[0];

  Vector<ComplexD> _S(L);
  ComplexD* S = &_S[0];
  thread_for(idx, L, {
      S[idx] = s[idx];
    });

  if (dim == 0 && grid->_simd_layout[0] == 1) {
    accelerator_for(idx, grid->oSites(), T::Nsimd(), {
        int s_idx = idx % L;
        coalescedWrite(dst_p[idx], coalescedRead(src_p[idx]) * S[s_idx]);
      });
  } else {
    ERR("Not implemented yet");
  }
  
}

// sliceSum from Grid but with vector of lattices as input
template<class vobj>
inline void cgpt_rank_slice_sum(const PVector<Lattice<vobj>> &Data,
				std::vector<typename vobj::scalar_object> &result,
				int orthogdim)
{
  ///////////////////////////////////////////////////////
  // FIXME precision promoted summation
  // may be important for correlation functions
  // But easily avoided by using double precision fields
  ///////////////////////////////////////////////////////
  typedef typename vobj::scalar_object sobj;

  GridBase *grid = Data[0].Grid();
  assert(grid!=NULL);

  const int     Nd = grid->_ndimension;
  const int  Nsimd = grid->Nsimd();
  const int Nbasis = Data.size();

  assert(orthogdim >= 0);
  assert(orthogdim < Nd);

  int fd = grid->_fdimensions[orthogdim];
  int ld = grid->_ldimensions[orthogdim];
  int rd = grid->_rdimensions[orthogdim];

  Vector<vobj> lvSum(rd * Nbasis);         // will locally sum vectors first
  Vector<sobj> lsSum(ld * Nbasis, Zero()); // sum across these down to scalars
  result.resize(fd * Nbasis);              // And then global sum to return the same vector to every node

  int      e1 = grid->_slice_nblock[orthogdim];
  int      e2 = grid->_slice_block [orthogdim];
  int  stride = grid->_slice_stride[orthogdim];
  int ostride = grid->_ostride[orthogdim];

  // sum over reduced dimension planes, breaking out orthog dir
  // Parallel over orthog direction
  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  auto lvSum_p = &lvSum[0];
  typedef decltype(coalescedRead(Data_v[0][0])) CalcElem;

  accelerator_for(r, rd * Nbasis, grid->Nsimd(), {
    CalcElem elem = Zero();

    int n_base = r / rd;
    int so = (r % rd) * ostride; // base offset for start of plane
    for(int n = 0; n < e1; n++){
      for(int b = 0; b < e2; b++){
        int ss = so + n * stride + b;
        elem += coalescedRead(Data_v[n_base][ss]);
      }
    }
    coalescedWrite(lvSum_p[r], elem);
  });
  VECTOR_VIEW_CLOSE(Data_v);

  thread_for(n_base, Nbasis, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<sobj> extracted(Nsimd); // splitting the SIMD
    Coordinate icoor(Nd);

    for(int rt = 0; rt < rd; rt++){
      extract(lvSum[n_base * rd + rt], extracted);
      for(int idx = 0; idx < Nsimd; idx++){
        grid->iCoorFromIindex(icoor, idx);
        int ldx = rt + icoor[orthogdim] * rd;
        lsSum[n_base * ld + ldx] = lsSum[n_base * ld + ldx] + extracted[idx];
      }
    }

    for(int t = 0; t < fd; t++){
      int pt = t / ld; // processor plane
      int lt = t % ld;
      if ( pt == grid->_processor_coor[orthogdim] ) {
        result[n_base * fd + t] = lsSum[n_base * ld + lt];
      } else {
        result[n_base * fd + t] = Zero();
      }
    }
  });
}

template<class vobj>
inline void cgpt_rank_indexed_sum(const PVector<Lattice<vobj>> &Data,
				  const Lattice<iSinglet<typename vobj::vector_type>> & Index,
				  std::vector<typename vobj::scalar_object> &result)
{
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *grid = Data[0].Grid();
  ASSERT(grid == Index.Grid());
  
  const int Nbasis = Data.size();
  constexpr int n_elem = GridTypeMapper<vobj>::count;
  
  size_t len = result.size() / Nbasis;
  ASSERT(result.size() % Nbasis == 0);

  size_t index_osites_per_block = (grid->oSites() + len - 1) / len;

  Vector<sobj> lsSum(index_osites_per_block * len * Nbasis);
  auto lsSum_p = &lsSum[0];
  
  // first zero blocks
  accelerator_for(ss, lsSum.size(), 1, {
      lsSum_p[ss] = Zero();
    });

  int Nsimd = grid->Nsimd();
  int ndim = grid->Nd();

  long index_osites = grid->oSites();

  autoView(Index_v, Index, AcceleratorRead);
  auto Index_p = &Index_v[0];

  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  
  accelerator_for(ii,index_osites_per_block,1,{

      for (long jj=0;jj<len;jj++) {
	long oidx = jj*index_osites_per_block + ii;
	if (oidx < index_osites) {
      
	  for (int lane=0;lane<Nsimd;lane++) {

	    long index = (long)((scalar_type*)&Index_p[oidx])[lane].real();
			
	    for (int nb=0;nb<Nbasis;nb++) {
	      for (int i=0;i<n_elem;i++) {
		((scalar_type*)&lsSum_p[(nb * len + index)*index_osites_per_block + ii])[i] +=
		  ((scalar_type*)&Data_v[nb][oidx])[i * Nsimd + lane];
	      }
	    }
	  }
	}
      }	
  });

  VECTOR_VIEW_CLOSE(Data_v);

  thread_for(i, result.size(), {
      sobj x = Zero();
      for (size_t j=0;j<index_osites_per_block;j++)
	x = x + lsSum_p[i*index_osites_per_block + j];
      result[i] = x;
    });
}
