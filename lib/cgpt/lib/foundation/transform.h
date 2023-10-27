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
void cgpt_scale_per_coordinate(Lattice<T>& dst,Lattice<T>& src,ComplexD* s,int dim) {

  GridBase* grid = dst.Grid();
  conformable(grid, src.Grid());

  dst.Checkerboard() = src.Checkerboard();

  int ndim = grid->Nd();
  int nsimd = grid->Nsimd();
  int fdim = grid->_fdimensions[dim];
  int gdim = grid->_gdimensions[dim];
  int osites = grid->oSites();

  autoView(dst_v, dst, AcceleratorWriteDiscard);
  autoView(src_v, src, AcceleratorRead);

  auto dst_p = &dst_v[0];
  auto src_p = &src_v[0];

  Vector<ComplexD> _S(fdim);
  ComplexD* S = &_S[0];
  thread_for(idx, fdim, {
      S[idx] = s[idx];
    });

  if (fdim == gdim && grid->_simd_layout[dim] == 1) {
    accelerator_for(idx, osites, nsimd, {
        Coordinate ocoor(ndim);
        Lexicographic::CoorFromIndex(ocoor, idx, grid->_rdimensions);
        int s_idx = ocoor[dim];
        coalescedWrite(dst_p[idx], coalescedRead(src_p[idx]) * S[s_idx]);
      });
  } else {

    // Lexicographic coordinates_from_cartesian_view for dimension dim
    Coordinate top(ndim);
    Coordinate size(ndim);
    int points = 1;
    for (int idx=0; idx < ndim; idx++) {
        int cbf = grid->_fdimensions[idx] / grid->_gdimensions[idx];
        top[idx] = grid->_processor_coor[idx] * grid->_ldimensions[idx] * cbf;
        size[idx] = grid->_ldimensions[idx] * cbf;
        points *= size[idx];
    }
    int fstride = 1;
    for (int idx=0; idx < ndim; idx++) {
        if (grid->_checker_dim_mask[idx])
            break;
        fstride *= size[idx];
    }
    int cb = src.Checkerboard();
    std::vector<int32_t> _coor(osites * nsimd);
    int32_t* coor = &_coor[0];
    thread_for(idx, points, {
        Coordinate lcoor(ndim);
        Lexicographic::CoorFromIndex(lcoor,idx,size);
        long idx_cb = (idx % fstride) + ((idx / fstride)/2) * fstride;
        long site_cb = 0;
        for (int i=0; i < ndim; i++)
            if (grid->_checker_dim_mask[i])
                site_cb += top[i] + lcoor[i];
        if (site_cb % 2 == cb) {
            coor[idx_cb] = top[dim] + lcoor[dim];
        }
    });

    // Compute coordinates for simd
    Coordinate gstride(ndim);
    gstride[0] = 1;
    for (int idx=1; idx < ndim; idx++) {
      gstride[idx] = gstride[idx-1] * grid->_gdimensions[idx-1];
    }
    Vector<int32_t> _Coor(osites);
    int32_t* Coor = &_Coor[0];
    for(int idx=0; idx < osites; idx++) {
      Coordinate ocoor(ndim);
      Lexicographic::CoorFromIndex(ocoor, idx, grid->_rdimensions);
      for (int lane=0; lane < nsimd; lane++) {
          Coordinate icoor(ndim);
          grid->iCoorFromIindex(icoor, lane);
          int lidx=0;
          for (int nd=0; nd < ndim; nd++) {
              lidx += (ocoor[nd] + grid->_rdimensions[nd] * icoor[nd]) * gstride[nd];
          }
          if (lane == 0) {
              Coor[idx] = coor[lidx];
          }
          if (Coor[idx] != coor[lidx]) {
              // Need to update simd separately
              ERR("Not implemented yet");
          }
      }
    }

    // Scale dimension dim
    accelerator_for(idx, osites, nsimd, {
        int s_idx = Coor[idx];
        coalescedWrite(dst_p[idx], coalescedRead(src_p[idx]) * S[s_idx]);
      }); 
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
