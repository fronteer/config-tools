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
#ifndef __FWD_NCHW_CONFIG_HPP__
#define __FWD_NCHW_CONFIG_HPP__

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

class fwd_nchw_config : public basic_igemm_config
{
public:
    fwd_nchw_config() = default;
    ~fwd_nchw_config() = default;

    fwd_nchw_config(const fwd_nchw_config&) = delete;
    fwd_nchw_config& operator=(fwd_nchw_config&) = delete;

    void generate_configs(const char *precision, const char *config_file);
private:
    std::vector<igemm_gtc_tunable_t> configs;

    int getMaximumSlice_a_c1e(int gemm_k_per_block, int blockSize, int macro_tile_m);
    int getMaximumCluster_b_n1b(int gemm_k_per_block, int blockSize, int macro_tile_n);
};

int fwd_nchw_config::getMaximumSlice_a_c1e(int gemm_k_per_block, int blockSize, int macro_tile_m)
{
    int a_slice_size=8; 

    for (; a_slice_size > 1; a_slice_size /= 2) 
        if ( blockSize / (gemm_k_per_block/a_slice_size) < macro_tile_m ) 
             break; 	

    return(a_slice_size); 
}; 

int fwd_nchw_config::getMaximumCluster_b_n1b(int gemm_k_per_block, int blockSize, int macro_tile_n)
{
    int b_cluster_size = std::min(blockSize, macro_tile_n);

    for (; b_cluster_size > 1; b_cluster_size /= 2) 
        if ( blockSize / b_cluster_size <= gemm_k_per_block )
	     break; 
 
    return(b_cluster_size); 
}; 

void fwd_nchw_config::generate_configs(const char *precision, const char *config_file)
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
         cfg.direction = "fwd";
	 cfg.precision = precision; 

         int blockSize = waveSize * xm.waves; 

         for (int nxe=0; nxe < 2; nxe += 1)  {
              cfg.nxe = nxe; 	
	      for (int nxb=1; nxb < 17; nxb *= 4) {
                   if ( cfg.gemm_n_per_block % nxb != 0 ) 
			continue;  

                   cfg.nxb = nxb;

                   // consider the gemm_k_per_block sizes to be 2x, 4x, 8x that of the k_per_inst of the specific xlops instruction
                   // for fp32, gemm_k_per_block must be at lest 4-times multiplier of k_per_inst
                   int lower_k_shifts = std::string(precision) == "fp16"? 1 : 2;
                   int upper_k_shifts = std::string(precision) == "fp16"? 3 : 4;

                   for (int k=lower_k_shifts; k < upper_k_shifts; k++) {
                        cfg.gemm_k_per_block = xm.wave_tile_k << k;             

                        if ( cfg.gemm_k_per_block / 8 > blockSize )  // this should not occurr easily
		             continue;  

                        if ( blockSize / (cfg.gemm_k_per_block/1) > cfg.gemm_m_per_block ) // this could occurr easily for small value of gemm_m_per_block 
			     continue; 	

                        if ( blockSize / std::min(blockSize, cfg.gemm_n_per_block) > cfg.gemm_k_per_block )
			     continue; 

                        int slice_a_c1e = getMaximumSlice_a_c1e(cfg.gemm_k_per_block, blockSize, cfg.gemm_m_per_block); 

                        // We have the following assumption to generate fwd configs:
                        // 1) cluster dimension is always the lower dimension of gemm_k/gemm_m/gemm_n (c1e/k1/n1b)
                        // 2) c0 is not used for both cluster and slice allocation 
			// 3) gemm_m/gemm_n uses lower higher dimensions for slice (k0, n0)

	                cfg.tensor_a_thread_lengths[0] = 1; 
			cfg.tensor_b_thread_lengths[0] = 1; 
                        cfg.tensor_a_cluster_lengths[0] = 1; 
		        cfg.tensor_b_cluster_lengths[0] = 1; 
                        cfg.tensor_a_cluster_lengths[2] = 1; 
		        cfg.tensor_b_cluster_lengths[2] = 1;
                        cfg.tensor_a_thread_lengths[3] = 1; 
			cfg.tensor_b_thread_lengths[3] = 1; 

	                cfg.tensor_a_thread_lengths[1] = slice_a_c1e; 
	                cfg.tensor_a_cluster_lengths[1] = cfg.gemm_k_per_block / slice_a_c1e; 
	                cfg.tensor_a_cluster_lengths[3] = blockSize / cfg.tensor_a_cluster_lengths[1]; 
	                cfg.tensor_a_thread_lengths[2] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3]; 

                        int cluster_b_n1b = getMaximumCluster_b_n1b(cfg.gemm_k_per_block, blockSize, cfg.gemm_n_per_block); 

		        cfg.tensor_b_cluster_lengths[3] = cluster_b_n1b; 
		        cfg.tensor_b_cluster_lengths[1] = blockSize / cluster_b_n1b; 
			cfg.tensor_b_thread_lengths[1] = cfg.gemm_k_per_block / cfg.tensor_b_cluster_lengths[1];
			cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3];  

                        configs.push_back(cfg); 

                        // we need a config which has tensor_b_thread_lengths[1] = 1 to support the cases where either x != 1 or y != 1
                        if ( cfg.tensor_b_cluster_lengths[1] != cfg.gemm_k_per_block && blockSize / cfg.gemm_k_per_block <= cfg.gemm_n_per_block ) {
                             cfg.tensor_b_thread_lengths[0] = 1; 
			     cfg.tensor_b_thread_lengths[1] = 1; 
			     cfg.tensor_b_cluster_lengths[0] = 1; 
                             cfg.tensor_b_cluster_lengths[1] = cfg.gemm_k_per_block; 
                             cfg.tensor_b_cluster_lengths[2] = 1; 
                             cfg.tensor_b_cluster_lengths[3] = blockSize / cfg.tensor_b_cluster_lengths[1]; 
			     cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3]; 
			     cfg.tensor_b_thread_lengths[3] = 1; 

                             // to satisfy unmerge_sub_n % nb_n0 == 0 
                             if ( (cfg.gemm_n_per_block / cfg.nxb ) % cfg.tensor_b_thread_lengths[2] == 0 ) 
			           configs.push_back(cfg); 
			}; 
                   }; 
	     }; 
         };	
    };  

    output_configurations(this->configs, "C0xC1ExK0xK1", "C0xC1ExN0xN1B", ofs);

    std::cout << std::endl << this->configs.size() << " configs produced !" << std::endl;
};

struct FwdNchwSorterClass : public basic_config_sorter
{
  bool operator()(igemm_gtc_tunable_t &cfg1, igemm_gtc_tunable_t &cfg2)
  {
     // it seems larger size of gemm_k_per_block is not very helpful ?
     if ( cfg1.gemm_k_per_block > cfg2.gemm_k_per_block )
          return(true);
     if ( cfg1.gemm_k_per_block < cfg2.gemm_k_per_block )
          return(false);

     int blockSize_1 = cfg1.tensor_b_cluster_lengths[1] * cfg1.tensor_b_cluster_lengths[3];
     int blockSize_2 = cfg2.tensor_b_cluster_lengths[1] * cfg2.tensor_b_cluster_lengths[3];

     if ( blockSize_1 > blockSize_2 )
          return(true);
     if ( blockSize_1 < blockSize_2 )
          return(false);

     // ta_c1e is per-thread vector_load size, bigger is better for performance
     if ( cfg1.tensor_a_cluster_lengths[1] > cfg2.tensor_a_cluster_lengths[1] )
          return(true);
     if ( cfg1.tensor_a_cluster_lengths[1] < cfg2.tensor_a_cluster_lengths[1] )
          return(false);

     // simutaneously accessing by more threads (bigger cluster size) on faster dimension could benefit the performance
     if ( cfg1.tensor_b_cluster_lengths[3] > cfg2.tensor_b_cluster_lengths[3] )
          return(true);
     if ( cfg1.tensor_b_cluster_lengths[3] < cfg2.tensor_b_cluster_lengths[3] )
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

     if ( cfg1.wave_tile_k > cfg2.wave_tile_k )
          return(true);

     if ( cfg1.wave_tile_k < cfg2.wave_tile_k )
          return(false);

     return(false);
  };
}; 

#endif

