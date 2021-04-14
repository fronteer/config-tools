/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef __BWD_NCHW_CONFIG_HPP__
#define __BWD_NCHW_CONFIG_HPP__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <algorithm>
#include <map> 
#include <memory>
#include <fstream>
#include <iostream>
#include <string>

#include "config_parser.hpp"
#include "igemm_gtc_base.hpp"
#include "config_comm.hpp"

class bwd_nchw_config : public basic_igemm_config
{
public:
    bwd_nchw_config() = default;
    ~bwd_nchw_config() = default;

    bwd_nchw_config(const bwd_nchw_config&) = delete;
    bwd_nchw_config& operator=(bwd_nchw_config&) = delete;

    void generate_configs(const char *precision, const char *config_file);
private:
    std::vector<igemm_gtc_tunable_t> configs;

    int get_num_soffset_sgprs(int d0_length, int d1_length, int max_vector_size);
    int get_available_sgprs_for_soffset(bool is_zero_nxe); 
}; 

// try to adjust this if the generator codes improved the usage of sgprs

// d0_length is the length of d0-dimension 
// d1_length is the length os d1-dimension
int bwd_nchw_config::get_num_soffset_sgprs(int d0_length, int d1_length, int max_vector_size)
{
    assert(d0_length > 0 && d1_length > 0 && max_vector_size > 0); 

    int d1_num_vectors = d1_length / std::min<int>(d1_length, max_vector_size); 

    if ( d0_length == 1 )
	 if ( d1_num_vectors == 1 ) 
	      return(0); 
         else 
	      return( std::max<int>(d1_num_vectors-2, 0) ); 
    if ( d1_num_vectors == 1 )
	 if ( d0_length == 1 )
	      return(0); 
         else 
	      return( std::max<int>(d0_length-2, 0) ); 

    return( std::max<int>(d0_length*d1_num_vectors-3, 0) ); 
}; 

// This function heavily depends on the implementation of the generator for bwd-fp16
int bwd_nchw_config::get_available_sgprs_for_soffset(bool is_zero_nxe)
{
    if ( is_zero_nxe ) 
	 return(103-1-6-45); // "-1" is considering for "s_tmp" aligned allocation
    else 
	 return(103-1-6-63); // "-1" is considering for "s_tmp" aligned allocation
}; 

void bwd_nchw_config::generate_configs(const char *precision, const char *config_file)
{
    std::ofstream ofs(config_file, std::ofstream::out);

    int num_mappings = (std::string(precision) == "fp32")? NUM_XDLOPS_MAPPING_FP32 : NUM_XDLOPS_MAPPING_FP16; 

    for (int i=0; i < num_mappings; i++) {
         auto xm = (std::string(precision) == "fp32")? xdlops_mappings_fp32[i] : xdlops_mappings_fp16[i];

         igemm_gtc_tunable_t cfg;

         cfg.gemm_m_per_block = xm.macro_tile_m;
         cfg.gemm_n_per_block = xm.macro_tile_n;
         cfg.wave_tile_m = xm.wave_tile_m;
         cfg.wave_tile_n = xm.wave_tile_n;
         cfg.wave_tile_k = xm.wave_tile_k;
         cfg.wave_repeat_m = xm.wave_repeat_m;
         cfg.wave_repeat_n = xm.wave_repeat_n;
         cfg.wave_step_m = xm.wave_step_m;
         cfg.wave_step_n = xm.wave_step_n;

         cfg.tensor_a_thread_lengths.resize(4);
         cfg.tensor_a_cluster_lengths.resize(4);
         cfg.tensor_b_thread_lengths.resize(4);
         cfg.tensor_b_cluster_lengths.resize(4);

         cfg.tensor_layout = "nchw"; 
         cfg.direction = "bwd"; 
	 cfg.precision = precision; 

         int blockSize = waveSize * xm.waves;
         int tensor_a_soffset_sgprs; 
         int tensor_b_soffset_sgprs; 

         int max_vector_size = (std::string(precision) == "fp16") ? 8 : 4; 

         for (int nxe=0; nxe < 2; nxe += 1)  {
              cfg.nxe = nxe;
              for (int nxb=1; nxb < 5; nxb *= 4) { // no use for nxb bigger than 1
                   cfg.nxb = nxb;

                   int unmerge_sub_n = cfg.gemm_n_per_block / cfg.nxb;    // assuming gemm_n_unmerge_cluster == 0 is used for generated configs 

                   // consider the gemm_k_per_block sizes to be 2x, 4x, 8x that of the k_per_inst of the specific xlops instruction
		   
                   // for fp32, gemm_k_per_block must be at lest 4-times multiplier of k_per_inst
                   int lower_k_shifts = std::string(precision) == "fp16"? 1 : 2; 
                   int upper_k_shifts = std::string(precision) == "fp16"? 3 : 4;

                   for (int k=lower_k_shifts; k < upper_k_shifts; k++) { 
                        cfg.gemm_k_per_block = xm.wave_tile_k << k;

                        if ( cfg.gemm_k_per_block / 8 > blockSize )  // this should not occurr easily
                             continue;

                        // blockSize/(cfg.gemm_k_per_block/1) indicates the least required cluster size in gemm_m dimension
                        if ( blockSize / (cfg.gemm_k_per_block/1) > cfg.gemm_m_per_block ) 
                             continue;

                        // blockSize/std::min(blockSize,cfg.gemm_n_per_block) indicates the least required cluster size in gemm_k dimension 
                        if ( blockSize / std::min(blockSize, cfg.gemm_n_per_block) > cfg.gemm_k_per_block )
                             continue;

                        // We have the following assumption to generate configs:
			// 1) tensor_a and tensor_b tries to be same in gemm_k dimensions (k0, k1e) 
			// 2) cluster dimension is always the lower dimension of gemm_k/gemm_m/gemm_n (k1e/c1/n1b)
			// 3) For fp16, since gemm_k_pack is used, the per-block size on k1e should not be less than gemm_k_pack size 4 
		
                        cfg.tensor_a_cluster_lengths[0] = 1; 
			cfg.tensor_b_cluster_lengths[0] = 1; 
			cfg.tensor_a_cluster_lengths[2] = 1; 
			cfg.tensor_b_cluster_lengths[2] = 1; 

                        if ( cfg.nxe == 0 ) {
                            if ( cfg.gemm_n_per_block % nxb != 0 )   // only check this for nxe == 0 since for nxe == 1,  nhw is padded according to nxb
                                 continue;
                            
#if GENERATE_REDUCED_CONFIGS == 0 			    
                            // use dimension k0 for thread slice for gemm_k of tensor_a/tensor_b
                            for(int sliceSize=1; sliceSize <= cfg.gemm_k_per_block; sliceSize *= 2) {
                                cfg.tensor_a_thread_lengths[0] = sliceSize;
                                cfg.tensor_a_thread_lengths[1] = 1;
                                cfg.tensor_a_cluster_lengths[1] = cfg.gemm_k_per_block / sliceSize;

                                int n_k1e = cfg.tensor_a_thread_lengths[1] * cfg.tensor_a_cluster_lengths[1];

                                // this is a situation difficult to handle, so just give it up
                                if ( std::string(precision) == "fp16" && n_k1e < 4 )
                                     continue;

                                cfg.tensor_a_cluster_lengths[3] = blockSize / cfg.tensor_a_cluster_lengths[1];

                                cfg.tensor_b_cluster_lengths[1] = cfg.tensor_a_cluster_lengths[1];
                                cfg.tensor_b_cluster_lengths[3] = cfg.tensor_a_cluster_lengths[3];
                                cfg.tensor_b_thread_lengths[0] = cfg.tensor_a_thread_lengths[0];
                                cfg.tensor_b_thread_lengths[1] = cfg.tensor_a_thread_lengths[1];

                                if ( cfg.tensor_a_cluster_lengths[3] > cfg.gemm_m_per_block  || cfg.tensor_b_cluster_lengths[3] > cfg.gemm_n_per_block )
                                     continue;
				
                                bool last_cfg_selected = false; 

                                // use c0/n0 for thread slice for gemm_m/gemm_n
			        do {	
                                    cfg.tensor_a_thread_lengths[2] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3];
                                    cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3];
                                    cfg.tensor_a_thread_lengths[3] = 1;
                                    cfg.tensor_b_thread_lengths[3] = 1;

                                    tensor_a_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_a_thread_lengths[0], cfg.tensor_a_thread_lengths[2], 1); 
                                    tensor_b_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_b_thread_lengths[0], cfg.tensor_b_thread_lengths[2], 1); 

                                    // Limitation due to large sgpr consumption in precache soffset
                                    if ( tensor_a_soffset_sgprs + tensor_b_soffset_sgprs > get_available_sgprs_for_soffset(true) ) 
                                         break;

                                    if ( unmerge_sub_n % cfg.tensor_b_thread_lengths[2] != 0) 
					 break; 

                                    this->configs.push_back(cfg);
                                    last_cfg_selected = true; 
                                } while(0);

                                if ( last_cfg_selected && cfg.tensor_a_thread_lengths[2] == 1 && cfg.tensor_b_thread_lengths[2] == 1 )
				     continue; 

                                // use c1/n1b for thread slice for gemm_m/gemm_n
                                do {
                                    cfg.tensor_a_thread_lengths[3] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3];
                                    cfg.tensor_b_thread_lengths[3] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3];
                                    cfg.tensor_a_thread_lengths[2] = 1;
                                    cfg.tensor_b_thread_lengths[2] = 1;

                                    // global vector load puts limitations on the sizes of the thread slices (at most dwordx4 can be used) 
                                    if ( cfg.tensor_a_thread_lengths[3] > max_vector_size || cfg.tensor_b_thread_lengths[3] > max_vector_size )
				         break; 

                                    tensor_a_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_a_thread_lengths[0], cfg.tensor_a_thread_lengths[3], max_vector_size); 
                                    tensor_b_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_a_thread_lengths[0], cfg.tensor_a_thread_lengths[3], max_vector_size); 

                                    // Limitation due to large sgpr consumption in precache soffset
                                    if ( tensor_a_soffset_sgprs + tensor_b_soffset_sgprs > get_available_sgprs_for_soffset(true) )
                                         break;

                                    this->configs.push_back(cfg);
                                } while(0); 
                            };
#endif                               
                            // use dimension k1e for thread slice for gemm_k of tensor_a/tensor_b
                            for(int sliceSize=2; sliceSize <= cfg.gemm_k_per_block; sliceSize *= 2) {
                                cfg.tensor_a_thread_lengths[1] = sliceSize;
                                cfg.tensor_a_thread_lengths[0] = 1;
                                cfg.tensor_a_cluster_lengths[1] = cfg.gemm_k_per_block / sliceSize;

                                int n_k1e = cfg.tensor_a_thread_lengths[1] * cfg.tensor_a_cluster_lengths[1];

                                // this is a situation difficult to handle, so just give it up
                                if ( std::string(precision) == "fp16" && n_k1e < 4 )
                                     continue;

                                cfg.tensor_a_cluster_lengths[3] = blockSize / cfg.tensor_a_cluster_lengths[1];

                                cfg.tensor_b_cluster_lengths[1] = cfg.tensor_a_cluster_lengths[1];
                                cfg.tensor_b_cluster_lengths[3] = cfg.tensor_a_cluster_lengths[3];
                                cfg.tensor_b_thread_lengths[0] = cfg.tensor_a_thread_lengths[0];
                                cfg.tensor_b_thread_lengths[1] = cfg.tensor_a_thread_lengths[1];

                                if ( cfg.tensor_a_cluster_lengths[3] > cfg.gemm_m_per_block  || cfg.tensor_b_cluster_lengths[3] > cfg.gemm_n_per_block )
                                     continue;
#if GENERATE_REDUCED_CONFIGS == 0 
                                bool last_cfg_selected = false; 
                                
                                // use c0/n0 for thread slice for gemm_m/gemm_n
                                do {
                                    cfg.tensor_a_thread_lengths[2] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3];
                                    cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3];
                                    cfg.tensor_a_thread_lengths[3] = 1;
                                    cfg.tensor_b_thread_lengths[3] = 1;

                                    tensor_a_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_a_thread_lengths[1], cfg.tensor_a_thread_lengths[2], 1);
                                    tensor_b_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_b_thread_lengths[1], cfg.tensor_b_thread_lengths[2], 1);

                                    // Limitation due to large sgpr consumption in precache soffset
                                    if ( tensor_a_soffset_sgprs + tensor_b_soffset_sgprs > get_available_sgprs_for_soffset(true) )
                                         break; 

                                    if ( unmerge_sub_n % cfg.tensor_b_thread_lengths[2] != 0) 
					 break; 

                                    this->configs.push_back(cfg);

                                    last_cfg_selected = true;
                                } while(0); 

                                if ( last_cfg_selected && cfg.tensor_a_thread_lengths[2] == 1 && cfg.tensor_b_thread_lengths[2] == 1 )
                                     continue;				
#endif
                                // use c1/n1b for thread slice for gemm_m/gemm_n
                                do {
                                    cfg.tensor_a_thread_lengths[3] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3];
                                    cfg.tensor_b_thread_lengths[3] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3];
                                    cfg.tensor_a_thread_lengths[2] = 1;
                                    cfg.tensor_b_thread_lengths[2] = 1;

                                    // global vector load puts limitations on the sizes of the thread slices (at most dwordx4 can be used) 
                                    if ( cfg.tensor_a_thread_lengths[3] > max_vector_size || cfg.tensor_b_thread_lengths[3] > max_vector_size )
				         break; 

                                    tensor_a_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_a_thread_lengths[1], cfg.tensor_a_thread_lengths[3], max_vector_size); 
                                    tensor_b_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_b_thread_lengths[1], cfg.tensor_b_thread_lengths[3], max_vector_size);

                                    // Limitation due to large sgpr consumption in precache soffset
                                    if ( tensor_a_soffset_sgprs + tensor_b_soffset_sgprs > get_available_sgprs_for_soffset(true) )
                                         break;

                                    this->configs.push_back(cfg);
                                } while(0);
                            };   // end of for(...)
		        }
			else { 
			    // with nxe == 1, vector load can be used with Wei on dim-c1 when x == y == 1, wo we still need to generate configs where tensor_a_thread_length[3] > 1.
			    // for tensor_b, we only generate configs where tensor_b_thread_length[1], tensor_b_thread_length[3] are forced to be 1
			   
                            // use dimension k0 for thread slice for gemm_k of tensor_a/tensor_b
                            for(int sliceSize=1; sliceSize <= cfg.gemm_k_per_block; sliceSize *= 2) {
                                cfg.tensor_a_thread_lengths[0] = sliceSize; 
				cfg.tensor_a_thread_lengths[1] = 1; 
				cfg.tensor_a_cluster_lengths[1] = cfg.gemm_k_per_block / sliceSize; 

                                int n_k1e = cfg.tensor_a_thread_lengths[1] * cfg.tensor_a_cluster_lengths[1]; 

                                // this is a situation difficult to handle, so just give it up
                                if ( std::string(precision) == "fp16" && n_k1e < 4 ) 
				     continue; 

				cfg.tensor_a_cluster_lengths[3] = blockSize / cfg.tensor_a_cluster_lengths[1]; 
                                cfg.tensor_b_cluster_lengths[1] = cfg.tensor_a_cluster_lengths[1]; 
                                cfg.tensor_b_cluster_lengths[3] = cfg.tensor_a_cluster_lengths[3]; 

                                if ( cfg.tensor_a_cluster_lengths[3] > cfg.gemm_m_per_block  || cfg.tensor_b_cluster_lengths[3] > cfg.gemm_n_per_block )
			             continue; 

                                cfg.tensor_b_thread_lengths[0] = cfg.tensor_a_thread_lengths[0]; 
                                cfg.tensor_b_thread_lengths[1] = cfg.tensor_a_thread_lengths[1]; 

			        // use dimension c0/n0 for thread slice for gemm_m/gemm_n
                                do {
                                    cfg.tensor_a_thread_lengths[2] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3]; 
                                    cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3]; 
                                    cfg.tensor_a_thread_lengths[3] = 1; 
                                    cfg.tensor_b_thread_lengths[3] = 1; 

                                    tensor_a_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_a_thread_lengths[0], cfg.tensor_a_thread_lengths[2], 1);
                                    tensor_b_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_b_thread_lengths[0], cfg.tensor_b_thread_lengths[2], 1);
                         
                                    // This is needed since some configurations consume too many scale registers
                                    if ( tensor_a_soffset_sgprs + tensor_b_soffset_sgprs > get_available_sgprs_for_soffset(false) )
				         break; 
			        	
                                    if ( unmerge_sub_n % cfg.tensor_b_thread_lengths[2] != 0) 
			                 break; 

                                    this->configs.push_back(cfg); 
                                } while(0); 
#if GENERATE_REDUCED_CONFIGS == 0
                                // use dimension c1/n0 for thread slice for gemm_m/gemm_n
                                do {
                                    cfg.tensor_a_thread_lengths[3] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3];
                                    cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3];  // tensor_b keep using n0 since it could not use vector load/store
                                    cfg.tensor_a_thread_lengths[2] = 1; 
                                    cfg.tensor_b_thread_lengths[3] = 1; 

                                    if ( cfg.tensor_a_thread_lengths[3] == 1) 
					 break; 

                                    // global vector load puts limitations on the sizes of the thread slices (at most dwordx4 can be used) 
                                    if ( cfg.tensor_a_thread_lengths[3] > max_vector_size)
                                         break;

                                    tensor_a_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_a_thread_lengths[0], cfg.tensor_a_thread_lengths[3], max_vector_size);
                                    tensor_b_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_b_thread_lengths[0], cfg.tensor_b_thread_lengths[2], 1);

                                    // This is needed since some configurations consume too many scale registers
                                    if ( tensor_a_soffset_sgprs + tensor_b_soffset_sgprs > get_available_sgprs_for_soffset(false) )
                                         break;

                                    if ( unmerge_sub_n % cfg.tensor_b_thread_lengths[2] != 0)
                                         break;

                                    this->configs.push_back(cfg);
                                } while(0);
#endif				
			    }; 
			    
                            // use dimension k1e for thread slice for gemm_K of tensor_a/tensor_b
			    for(int sliceSize=2; sliceSize <= cfg.gemm_k_per_block; sliceSize *= 2) {
                                cfg.tensor_a_thread_lengths[0] = 1;
                                cfg.tensor_a_thread_lengths[1] = sliceSize;
                                cfg.tensor_a_cluster_lengths[1] = cfg.gemm_k_per_block / sliceSize;

                                int n_k1e = cfg.tensor_a_thread_lengths[1] * cfg.tensor_a_cluster_lengths[1];

                                // this is a situation difficult to handle, so just give it up
                                if ( std::string(precision) == "fp16" && n_k1e < 4 )
                                     continue;

                                cfg.tensor_a_cluster_lengths[3] = blockSize / cfg.tensor_a_cluster_lengths[1];
                                cfg.tensor_b_cluster_lengths[1] = cfg.tensor_a_cluster_lengths[1];
                                cfg.tensor_b_cluster_lengths[3] = cfg.tensor_a_cluster_lengths[3];

                                if ( cfg.tensor_a_cluster_lengths[3] > cfg.gemm_m_per_block  || cfg.tensor_b_cluster_lengths[3] > cfg.gemm_n_per_block )
                                     continue;

                                cfg.tensor_b_thread_lengths[0] = cfg.tensor_a_thread_lengths[0]; 
                                cfg.tensor_b_thread_lengths[1] = cfg.tensor_a_thread_lengths[1]; 

                                // use dimension c1/n0 for thread slice for gemm_m/gemm_n
                                do {
                                    cfg.tensor_a_thread_lengths[3] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3];
                                    cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3];
                                    cfg.tensor_a_thread_lengths[2] = 1; 
                                    cfg.tensor_b_thread_lengths[3] = 1; 

                                    // we don't need this config 
                                    if ( cfg.tensor_a_thread_lengths[3] == 1)
                                         break;

                                    // global vector load puts limitations on the sizes of the thread slices (at most dwordx4 can be used) 
                                    if ( cfg.tensor_a_thread_lengths[3] > max_vector_size)
                                         break; 

                                    tensor_a_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_a_thread_lengths[1], cfg.tensor_a_thread_lengths[3], max_vector_size);
                                    tensor_b_soffset_sgprs = get_num_soffset_sgprs(cfg.tensor_b_thread_lengths[1], cfg.tensor_b_thread_lengths[2], 1);

                                    // This is needed since some configurations consume too many scale registers
                                    if ( tensor_a_soffset_sgprs + tensor_b_soffset_sgprs > get_available_sgprs_for_soffset(false) )
                                         break;

                                    if ( unmerge_sub_n % cfg.tensor_b_thread_lengths[2] != 0)
                                         break;

                                    this->configs.push_back(cfg);
                                } while(0);
			    }; 
		        }; 			
                   };
             };
         };
    };

    output_configurations(this->configs, "k0xk1ExC0xC1", "K0xK1ExN0xN1B", ofs);

    std::cout << std::endl << this->configs.size() << " configs produced !" << std::endl;
}; 

struct BwdNchwSorterClass : public basic_config_sorter
{
  bool operator()(igemm_gtc_tunable_t &cfg1, igemm_gtc_tunable_t &cfg2)
  {
     if ( cfg1.gemm_m_per_block > cfg2.gemm_m_per_block )
          return(true);
     if ( cfg1.gemm_m_per_block < cfg2.gemm_m_per_block )
          return(false);

     // larger work-group size is preferred
     int blockSize_1 = cfg1.tensor_b_cluster_lengths[1] * cfg1.tensor_b_cluster_lengths[3];
     int blockSize_2 = cfg2.tensor_b_cluster_lengths[1] * cfg2.tensor_b_cluster_lengths[3];

     if ( blockSize_1 > blockSize_2 )
          return(true);
     if ( blockSize_1 < blockSize_2 )
          return(false);

     // simutaneously accessing by more threads (bigger cluster size) on faster dimension could benefit the performance
     if ( cfg1.tensor_b_cluster_lengths[3] > cfg2.tensor_b_cluster_lengths[3] )
          return(true);
     if ( cfg1.tensor_b_cluster_lengths[3] < cfg2.tensor_b_cluster_lengths[3] )
          return(false);

     if ( cfg1.gemm_k_per_block > cfg2.gemm_k_per_block )
          return(true);
     if ( cfg1.gemm_k_per_block < cfg2.gemm_k_per_block )
          return(false);

     // This is needed to ensure tunable with nxe==0 is selected for x=y=1 dilation_x=dilation_y=1, stride_x=stride_y=1, pad_x=pad_y=0
     // Check the tunable_is_valid() which is not touched by me 
     if ( cfg1.nxe < cfg2.nxe )
          return(true);
     if ( cfg1.nxe > cfg2.nxe )
          return(false);

     if ( cfg1.nxb > cfg2.nxb )
          return(true);
     if ( cfg1.nxb < cfg2.nxb )
          return(false);

     if ( cfg1.wave_tile_k < cfg2.wave_tile_k )
          return(true);

     if ( cfg1.wave_tile_k > cfg2.wave_tile_k )
          return(false);

     // The config which can use vector load/store on dim n1b is preferred 
     if ( cfg1.nxe == 0 && cfg2.nxe == 0) {
          if ( cfg1.tensor_b_thread_lengths[3] > cfg2.tensor_b_thread_lengths[3] )
               return(true);
          if ( cfg1.tensor_b_thread_lengths[3] < cfg2.tensor_b_thread_lengths[3] )
               return(false);
     };

     // The config which can use vector load/store on dim c is preferred 
     if ( cfg1.tensor_a_thread_lengths[3] > cfg2.tensor_a_thread_lengths[3] )
          return(true);
     if ( cfg1.tensor_a_thread_lengths[3] < cfg2.tensor_a_thread_lengths[3] )
          return(false);

     // for bwd-fp16, having thread slice on k0 or k1e has differrent meaning (pack_d0 or not)
     if ( cfg1.tensor_b_thread_lengths[3] > 1 && cfg2.tensor_b_thread_lengths[3] > 1 ) {
          // if vector load is used on n1b, we prefer to pack k1 
          if ( cfg1.tensor_b_thread_lengths[1] > cfg2.tensor_b_thread_lengths[1] )
               return(true);
          if ( cfg1.tensor_b_thread_lengths[1] < cfg2.tensor_b_thread_lengths[1] )
               return(false);
     }
     else {
          // if vector load is not used on n1b, we prefer to have slice on k0 instead of k1, so that
          // simulaneously access by multiple lanes cover smaller address space (better L2 hit)
          if ( cfg1.tensor_b_thread_lengths[1] < cfg2.tensor_b_thread_lengths[1] )
               return(true);
          if ( cfg1.tensor_b_thread_lengths[1] > cfg2.tensor_b_thread_lengths[1] )
               return(false);
     };

     // for bwd-fp16, having thread slice on k0 or k1e has differrent meaning (pack_d0 or not)
     if ( cfg1.tensor_a_thread_lengths[3] > 1 && cfg2.tensor_a_thread_lengths[3] > 1 ) {
          // if vector load is used on c1, we prefer to pack k1 
          if ( cfg1.tensor_a_thread_lengths[1] > cfg2.tensor_a_thread_lengths[1] )
               return(true);
          if ( cfg1.tensor_a_thread_lengths[1] < cfg2.tensor_a_thread_lengths[1] )
               return(false);
     }
     else {
          // if vector load is not used on c1, we prefer to have slice on k0 instead of k1, so that
          // simulaneously access by multiple lanes cover smaller address space (better L2 hit)
          if ( cfg1.tensor_a_thread_lengths[1] < cfg2.tensor_a_thread_lengths[1] )
               return(true);
          if ( cfg1.tensor_a_thread_lengths[1] > cfg2.tensor_a_thread_lengths[1] )
               return(false);
     };

     return(false);
  };
};

#endif
