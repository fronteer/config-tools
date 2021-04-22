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
#ifndef __BWD_NHWC_CONFIG_HPP__
#define __BWD_NHWC_CONFIG_HPP__

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

class bwd_nhwc_config : public basic_igemm_config
{
public:
    bwd_nhwc_config() = default;
    ~bwd_nhwc_config() = default;

    bwd_nhwc_config(const bwd_nhwc_config&) = delete;
    bwd_nhwc_config& operator=(bwd_nhwc_config&) = delete;

    void generate_configs(const char *precision, const char *config_file);
private:
    std::vector<igemm_gtc_tunable_t> configs;
}; 

void bwd_nhwc_config::generate_configs(const char *precision, const char *config_file)
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

         cfg.tensor_layout = "nhwc"; 
         cfg.direction = "bwd"; 
	 cfg.precision = precision; 

         int blockSize = waveSize * xm.waves;

         int max_vector_size = (std::string(precision) == "fp16") ? 8 : 4; 
         int max_k1_slice_size = max_vector_size; 
	 int min_k1_slice_size = (std::string(precision) == "fp16")? 4 : 1; 
         int max_c1_slice_size = max_vector_size; 
	 int min_c1_slice_size = 1; 

         // the following fields have constant value 1
         cfg.tensor_a_thread_lengths[0] = 1; 
	 cfg.tensor_a_thread_lengths[3] = 1; 
	 cfg.tensor_a_cluster_lengths[1] = 1;
	 cfg.tensor_a_cluster_lengths[2] = 1; 

	 cfg.tensor_b_thread_lengths[1] = 1; 
	 cfg.tensor_b_thread_lengths[2] = 1; 
	 cfg.tensor_b_cluster_lengths[0] = 1; 
	 cfg.tensor_b_cluster_lengths[2] = 1; 

         for (int nxe=0; nxe < 2; nxe += 1)  {
              cfg.nxe = nxe;
              cfg.nxb = 1;      // nxb is not used by bwd nhwc 

              // consider the gemm_k_per_block sizes to be 2x, 4x, 8x that of the k_per_inst of the specific xlops instruction
		   
              // for fp32, gemm_k_per_block must be at lest 4-times multiplier of k_per_inst
              int lower_k_shifts = std::string(precision) == "fp16"? 1 : 2; 
              int upper_k_shifts = std::string(precision) == "fp16"? 3 : 4;

              for (int k=lower_k_shifts; k < upper_k_shifts; k++) { 
                   cfg.gemm_k_per_block = xm.wave_tile_k << k;
 
		   for (int k1_slice=min_k1_slice_size; k1_slice <= max_k1_slice_size; k1_slice *= 2) {
                        cfg.tensor_a_thread_lengths[1] = k1_slice; 
		        cfg.tensor_a_cluster_lengths[0] = cfg.gemm_k_per_block / k1_slice; 
                        if ( cfg.tensor_a_cluster_lengths[0] == 0 )
			     continue; 
                        cfg.tensor_a_cluster_lengths[3] = blockSize / cfg.tensor_a_cluster_lengths[0]; 
                        if ( cfg.tensor_a_cluster_lengths[3] == 0 )
			     continue; 
                        cfg.tensor_a_thread_lengths[2] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3]; 
                        if ( cfg.tensor_a_thread_lengths[2] == 0 )
			     continue; 

                        // for fp16, lower gemm_k dim size must be at least 4 so that the gemm_k_pack can be accurately implemented
                        if ( std::string(precision) == "fp16" && cfg.tensor_a_thread_lengths[1] < 4 ) 
			     continue; 	

			for (int c1_slice=min_c1_slice_size; c1_slice <= max_c1_slice_size; c1_slice *= 2) {
                             cfg.tensor_b_thread_lengths[3] = c1_slice; 
			     cfg.tensor_b_cluster_lengths[3] = cfg.gemm_n_per_block / c1_slice; 
                             if ( cfg.tensor_b_cluster_lengths[3] == 0 )
				  continue;  
			     cfg.tensor_b_cluster_lengths[1] = blockSize / cfg.tensor_b_cluster_lengths[3]; 
                             if ( cfg.tensor_b_cluster_lengths[1] == 0 )
				  continue;  
			     cfg.tensor_b_thread_lengths[0] = cfg.gemm_k_per_block / cfg.tensor_b_cluster_lengths[1]; 
                             if ( cfg.tensor_b_thread_lengths[0] == 0 )
				  continue;  

                             // for fp16, lower gemm_k dim size must be at least 4 so that the gemm_k_pack can be accurately implemented
                             if ( std::string(precision) == "fp16" && cfg.tensor_b_cluster_lengths[1] < 4 )
				  continue;  

                             int k0_slice = cfg.tensor_b_thread_lengths[0]; 

                             // gemm_k_per_block must be divided exactly by k0*k1 (required by the bwd nhwc kernel implementation)
			     if ( cfg.gemm_k_per_block % (k0_slice*k1_slice) != 0 )
				  continue; 

                             this->configs.push_back(cfg); 
			}; 
		   };  
              };
         };
    };

    output_configurations(this->configs, "EK2K0xK1xN0xN1B", "K0xK1K2ExC0xC1", ofs);

    std::cout << std::endl << this->configs.size() << " configs produced !" << std::endl;
}; 

bool BwdNhwcSorter(igemm_gtc_tunable_t &cfg1, igemm_gtc_tunable_t &cfg2)
{
     // larger work-group size is preferred
     int blockSize_1 = cfg1.tensor_a_cluster_lengths[0] * cfg1.tensor_a_cluster_lengths[3];
     int blockSize_2 = cfg2.tensor_a_cluster_lengths[0] * cfg2.tensor_a_cluster_lengths[3];

     if ( blockSize_1 > blockSize_2 )
          return(true);
     if ( blockSize_1 < blockSize_2 )
          return(false);

     if ( cfg1.gemm_n_per_block > cfg2.gemm_n_per_block )
          return(true);
     if ( cfg1.gemm_n_per_block < cfg2.gemm_n_per_block )
          return(false);

     // The config which can use vector load/store on dim k1 is preferred 
     if ( cfg1.tensor_a_thread_lengths[1] > cfg2.tensor_a_thread_lengths[1] )
          return(true);
     if ( cfg1.tensor_a_thread_lengths[1] < cfg2.tensor_a_thread_lengths[1] )
          return(false);

     // The config which can use vector load/store on dim c1 is preferred 
     if ( cfg1.tensor_b_thread_lengths[3] > cfg2.tensor_b_thread_lengths[3] )
          return(true);
     if ( cfg1.tensor_b_thread_lengths[3] < cfg2.tensor_b_thread_lengths[3] )
          return(false);

     // bigger size in ta_n0 is preferred since this leads to smaller space simultaneously accessed by threads in a warp
     if ( cfg1.tensor_a_thread_lengths[2] > cfg2.tensor_a_thread_lengths[2] )
          return(true);
     if ( cfg1.tensor_a_thread_lengths[2] < cfg2.tensor_a_thread_lengths[2] )
          return(false);

     if ( cfg1.gemm_k_per_block > cfg2.gemm_k_per_block )
          return(true);
     if ( cfg1.gemm_k_per_block < cfg2.gemm_k_per_block )
          return(false);

     // bigger size in tb_k0 is preferred since this leads to smaller space simultaneously accessed by threads in a warp
     if ( cfg1.tensor_b_thread_lengths[0] > cfg2.tensor_b_thread_lengths[0] )
          return(true);
     if ( cfg1.tensor_b_thread_lengths[0] < cfg2.tensor_b_thread_lengths[0] )
          return(false);

     // This is needed to ensure tunable with nxe==0 is selected for x=y=1 dilation_x=dilation_y=1, stride_x=stride_y=1, pad_x=pad_y=0
     // Check the tunable_is_valid() which is not touched by me 
     if ( cfg1.nxe < cfg2.nxe )
          return(true);
     if ( cfg1.nxe > cfg2.nxe )
          return(false);

     if ( cfg1.wave_tile_k > cfg2.wave_tile_k )
          return(true);

     if ( cfg1.wave_tile_k < cfg2.wave_tile_k )
          return(false);

     return(false);
};

#endif
