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

template<typename rank_t>
global_transfer<rank_t>::global_transfer(rank_t _rank, Grid_MPI_Comm _comm) : rank(_rank), comm(_comm) {

#ifdef CGPT_USE_MPI
  MPI_Comm_size(comm,&mpi_ranks);
  MPI_Comm_rank(comm,&mpi_rank);
  mpi_rank_map.resize(mpi_ranks,0);
  ASSERT(rank < mpi_ranks);
  mpi_rank_map[rank] = mpi_rank;
  ASSERT(MPI_SUCCESS == MPI_Allreduce(MPI_IN_PLACE,&mpi_rank_map[0],mpi_ranks * sizeof(rank_t) / sizeof(int),MPI_INT,MPI_SUM,comm));
#else
  mpi_ranks=1;
  mpi_rank=0;
  mpi_rank_map.push_back(0);
  ASSERT(rank == 0);
#endif

}

template<typename rank_t>
void global_transfer<rank_t>::global_sum(uint64_t* pdata, size_t size) {
#ifdef CGPT_USE_MPI
  ASSERT(MPI_SUCCESS == MPI_Allreduce(MPI_IN_PLACE,pdata,size,MPI_UINT64_T,MPI_SUM,comm));
#endif
}

template<typename rank_t>
template<typename vec_t>
void global_transfer<rank_t>::root_to_all(const std::map<rank_t, vec_t > & all, vec_t & my) {

  // store mine
  if (rank == 0) {
    auto e = all.find(0);
    my.clear();
    if (e != all.end())
      my = e->second;
  }

#ifdef CGPT_USE_MPI

  std::vector<int> all_size(mpi_ranks,0);
  int my_size;
  for (rank_t i=0; i<mpi_ranks; i++) {
    auto e = all.find(i);
    all_size[i] = (e != all.end()) ? (int)e->second.size() : 0;
  }

  ASSERT(MPI_SUCCESS == MPI_Scatter(&all_size[0], 1, MPI_INT, &my_size, 1, MPI_INT, mpi_rank_map[0], comm));

  // root node now receives from every node the list of its partners (if it is non-vanishing)
  std::vector<MPI_Request> req;
  if (rank == 0) {
    // root node now
    for (rank_t i=1;i<mpi_ranks;i++) {

      int rank_size = all_size[mpi_rank_map[i]];

      if (rank_size != 0) {
        auto & data = all.at(i);

	isend(i,data);
      }

    }

  } else {

    if (my_size != 0) {
      my.resize(my_size);
      irecv(0,my);
    }
  }

  waitall();
#endif
}

template<typename rank_t>
long global_transfer<rank_t>::global_gcd(long n) {

#ifdef CGPT_USE_MPI
  std::vector<long> all_n(mpi_ranks,0);
  ASSERT(MPI_SUCCESS == MPI_Allgather(&n, 1, MPI_LONG, &all_n[0], 1, MPI_LONG, comm));
  return cgpt_reduce(all_n, cgpt_gcd, n);
#else
  return n;
#endif
  
}

template<typename rank_t>
template<typename vec_t>
void global_transfer<rank_t>::all_to_root(const vec_t& my, std::map<rank_t, vec_t > & all) {

  // store mine
  if (rank == 0)
    all[0] = my;

#ifdef CGPT_USE_MPI
  int my_size = (int)my.size();
  std::vector<int> all_size(mpi_ranks,0);
  ASSERT(MPI_SUCCESS == MPI_Gather(&my_size, 1, MPI_INT, &all_size[0], 1, MPI_INT, mpi_rank_map[0], comm));

  // root node now receives from every node the list of its partners (if it is non-vanishing)
  if (rank == 0) {
    // root node now
    for (rank_t i=1;i<mpi_ranks;i++) {

      int rank_size = all_size[mpi_rank_map[i]];

      if (rank_size != 0) {
	auto & data = all[i];
	data.resize(rank_size);

	irecv(i,data);
      }

    }

  } else {

    if (my_size != 0) {
      isend(0,my);
    }
  }

  waitall();
#endif
}

template<typename rank_t>
void global_transfer<rank_t>::waitall() {
#ifdef CGPT_USE_MPI
  //printf("WAIT %d\n",(int)requests.size());
  if (!requests.size())
    return;
  std::vector<MPI_Status> stat(requests.size());
  ASSERT(MPI_SUCCESS == MPI_Waitall((int)requests.size(), &requests[0], &stat[0]));
  requests.clear();

#ifndef ACCELERATOR_AWARE_MPI

  for (auto & b : host_bounce_buffer) {
    if (b.device) {
#ifdef GRID_CHECKSUM_COMMS
      if (b.device_mt == mt_accelerator) {
	acceleratorCopyToDevice(b.host, b.device, b.size - 8*3);
      } else {
	memcpy(b.device, b.host, b.size - 8*3);
      }
      uint64_t computed_cs = host_bounce_checksum((uint64_t*)b.device, b.size / 8 - 1*3, b.device_mt);// ^ (13 + *0 + 1000 * b.tag);
      uint64_t expected_cs = *(uint64_t*)(((char*)b.host) + b.size - 8*3);
      uint64_t tag = *(uint64_t*)(((char*)b.host) + b.size - 8*2);
      uint64_t inc = *(uint64_t*)(((char*)b.host) + b.size - 8*1);
      uint64_t inc_exp = host_checksum_increment(b.sender, rank);
      
      if (computed_cs != expected_cs || inc != inc_exp || tag != b.tag)
	ERR("Packet receive checksum mismatch: %ld != %ld, tag %ld != %ld, inc %ld != %ld, bsender,rank = %ld, %ld", (long)computed_cs, (long)expected_cs,
	    (long)tag, (long)b.tag, inc, inc_exp,(long)b.sender, (long)rank);
#else
      acceleratorCopyToDevice(b.host, b.device, b.size);
#endif
    }
  }
  host_bounce_reset();

#endif

#endif
}

template<typename rank_t>
void global_transfer<rank_t>::isend(rank_t other_rank, const void* pdata, size_t sz, memory_type type) {
  if (sz <= size_mpi_max) {
#ifdef CGPT_USE_MPI
    int tag = other_rank * 65100 + rank;
    //printf("Send from %d to %d, %d bytes from %p (%g double)\n",this->rank,other_rank,(int)sz,pdata,*(double*)pdata);

#ifndef ACCELERATOR_AWARE_MPI

    if (type == mt_accelerator
#ifdef GRID_CHECKSUM_COMMS
	|| true
#endif
	) {

#ifdef GRID_CHECKSUM_COMMS
      void* host = host_bounce_allocate(sz + 8*3, 0, type, tag, rank);
      if (type == mt_accelerator) {
	acceleratorCopyFromDevice(pdata, host, sz);
      } else {
	memcpy(host, pdata, sz);
      }

      ASSERT(sz % 8 == 0);
      *(uint64_t*)(((char*)host) + sz) = host_bounce_checksum((uint64_t*)pdata, sz / 8, type);// ^ (13 + host_checksum_increment(rank, other_rank)*0 + 1000 * tag);
      *(uint64_t*)(((char*)host) + sz + 8) = tag;
      *(uint64_t*)(((char*)host) + sz + 16) = host_checksum_increment(rank, other_rank);
      pdata = host;
      sz += 8*3;
#else
      void* host = host_bounce_allocate(sz, 0, type, tag, rank);
      acceleratorCopyFromDevice(pdata, host, sz);
      pdata = host;
#endif

    }
    
#endif
    
    MPI_Request r;
    ASSERT(MPI_SUCCESS == MPI_Isend(pdata,sz,MPI_CHAR,mpi_rank_map[other_rank],tag,comm,&r));
    requests.push_back(r);
#endif
    
  } else {
    while (sz) {
      size_t sz_block = std::min(sz,size_mpi_max);
      isend(other_rank,pdata,sz_block,type);
      sz -= sz_block;
      pdata = (void*)((char*)pdata + sz_block);
    }
  }
}

template<typename rank_t>
void global_transfer<rank_t>::irecv(rank_t other_rank, void* pdata, size_t sz, memory_type type) {
  if (sz <= size_mpi_max) {
#ifdef CGPT_USE_MPI
    int tag = rank * 65100 + other_rank;
    
#ifndef ACCELERATOR_AWARE_MPI
  
    if (type == mt_accelerator
#ifdef GRID_CHECKSUM_COMMS
	|| true
#endif
	) {

#ifdef GRID_CHECKSUM_COMMS
      sz += 8*3;
#endif

      void* host = host_bounce_allocate(sz, pdata, type, tag, other_rank);
      pdata = host;
    }
    
#endif
    
    //printf("Recv from %d to %d, %d bytes to %p\n",other_rank,this->rank,(int)sz,pdata);
    MPI_Request r;
    ASSERT(MPI_SUCCESS == MPI_Irecv(pdata,sz,MPI_CHAR,mpi_rank_map[other_rank],tag,comm,&r));
    requests.push_back(r);
#endif
  } else {
    while (sz) {
      size_t sz_block = std::min(sz,size_mpi_max);
      irecv(other_rank,pdata,sz_block,type);
      sz -= sz_block;
      pdata = (void*)((char*)pdata + sz_block);
    }
  }
}

template<typename rank_t>
void global_transfer<rank_t>::provide_my_receivers_get_my_senders(const std::map<rank_t, size_t>& receivers,
								  std::map<rank_t, size_t>& senders) {

  struct rank_size_t {
    rank_t rank;
    size_t size;
  };

  // root node collects
  std::map<rank_t, std::vector<rank_size_t> > ranks_that_will_receive_data_from_rank;
  std::vector<rank_size_t> ranks_that_will_receive_my_data;
  for (auto & r : receivers) {
    ranks_that_will_receive_my_data.push_back({r.first, r.second});
  }

  all_to_root(ranks_that_will_receive_my_data, ranks_that_will_receive_data_from_rank);

  // create communication matrix
  std::map<rank_t, std::vector<rank_size_t> > ranks_from_which_rank_will_receive_data;
  if (this->rank == 0) {
    // for each rank create list of ranks that needs to talk to them
    std::map<rank_t, std::map<rank_t,size_t> > ranks_set_from_which_rank_will_receive_data;

    rank_t l;
    for (l=0;l<this->mpi_ranks;l++) {
      for (rank_size_t j : ranks_that_will_receive_data_from_rank[l]) {
	auto & x = ranks_set_from_which_rank_will_receive_data[j.rank];
	auto y = x.find(l);
	ASSERT(y == x.end());
	x[l] = j.size;
      }
    }

    for (l=0;l<this->mpi_ranks;l++) {
      for (auto & x : ranks_set_from_which_rank_will_receive_data[l]) {
	ranks_from_which_rank_will_receive_data[l].push_back({x.first,x.second});
      }
    }
  }

  // scatter packet number to be received by each rank
  std::vector<rank_size_t> ranks_from_which_I_will_receive_data;
  root_to_all(ranks_from_which_rank_will_receive_data, ranks_from_which_I_will_receive_data);

  // convert
  for (auto & r : ranks_from_which_I_will_receive_data) {
    //std::cout << "Rank " << this->rank << " here, will receive " << r.size << " bytes from rank " << r.rank << std::endl;
    senders[r.rank] = r.size;
  }
}

template<typename rank_t>
void global_transfer<rank_t>::multi_send_recv(const std::map<rank_t, comm_message>& send,
					      std::map<rank_t, comm_message>& recv) {

  // prepare list of all ranks for which I have data
  std::map<rank_t, size_t> my_receivers;
  std::map<rank_t, size_t> my_senders;

  for (auto & s : send) {
    ASSERT(s.first != rank); // should not send to myself
    my_receivers[s.first] = s.second.data.size();
  }

  provide_my_receivers_get_my_senders(my_receivers,my_senders);

  // allocate receive buffers
  for (auto & s : my_senders) {
    recv[s.first].resize(s.second);
  }

  // initiate communication
  for (auto & s : send) {
    isend(s.first, s.second.data);
  }

  for (auto & r : recv) {
    irecv(r.first, r.second.data);
  }

  // wait
  waitall();
}
