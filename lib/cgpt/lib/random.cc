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
#include "lib.h"
#include "random/ranlux.h"
#include "random/vector.h"
#include "random/distribution.h"
#include "random/parallel.h"
#include "random/engine.h"

/*
  rng[hash[pos]]

  hash[pos] : divide lattice into 2^nd blocks/hashes
              need to make sure that no two ranks have same hash

	      for 32^4 local volume, this means 65536 local hashes

	      when do we send all hashes do root to check for duplicates
*/

EXPORT(create_random,{

#if 0
    {
      std::vector<long> seed = { 1,2,3 };
      cgpt_vrng_ranlux24_794_256 vtest(seed);
      cgpt_rng_ranlux24_794_256 stest(seed);
      
      for (int i=0;i<1024*100;i++) {
	long a = vtest();
	long b = stest();
	assert(a == b);
      }
      
      double t0 = cgpt_time();
      for (int i=0;i<1024*100;i++) {
	long a = vtest();
      }
      double t1 = cgpt_time();
      for (int i=0;i<1024*100;i++) {
	long a = stest();
      }
      double t2 = cgpt_time();
      std::cout << GridLogMessage << "Timing: " << (t1-t0) << " and " << (t2-t1) << std::endl;
      
      cgpt_random_vectorized_ranlux24_794_256 rnd(seed);
      std::cout << GridLogMessage << rnd.get_normal() << std::endl;
    }
#endif



    PyObject* _type,* _seed;
    std::string type, seed;
    if (!PyArg_ParseTuple(args, "OO", &_type,&_seed)) {
      return NULL;
    }
    cgpt_convert(_type,type);
    cgpt_convert(_seed,seed);

    if (type == "vectorized_ranlux24_794_256") {
      //std::cout << "Before: " << seed << std::endl;
      return PyLong_FromVoidPtr(new cgpt_random_engine< cgpt_random_vectorized_ranlux24_794_256 >(seed));
    } else {
      ERR("Unknown rng engine type %s",type.c_str());
    }

    return PyLong_FromLong(0);
    
  });

EXPORT(delete_random,{

    void* _p;
    if (!PyArg_ParseTuple(args,"l", &_p)) {
      return NULL;
    }

    cgpt_random_engine_base* p = (cgpt_random_engine_base*)_p;
    delete p;

    return PyLong_FromLong(0);
  });

EXPORT(random_sample,{

    PyObject* _target, *_param;
    void* _p;
    if (!PyArg_ParseTuple(args, "lOO", &_p,&_target,&_param)) {
      return NULL;
    }

    cgpt_random_engine_base* p = (cgpt_random_engine_base*)_p;
    return p->sample(_target,_param);

  });