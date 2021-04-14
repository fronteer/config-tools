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
 * The above copyright notice and this permission notice shall be included in
 *all
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
#ifndef __CONFIG_COMM_HPP__
#define __CONFIG_COMM_HPP__

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

//#include "config_parser.h"
//#include "igemm_gtc_base.h"

typedef struct {
    int macro_tile_m;
    int macro_tile_n;
    int wave_tile_m;
    int wave_tile_n;
    int wave_tile_k;
    int waves;
    int wave_repeat_m;
    int wave_repeat_n;
    int wave_step_m;
    int wave_step_n;
} xdlops_mapping_t; 

#ifndef USE_REDUCED_XDLOPS_MAPPINGS
#define USE_REDUCED_XDLOPS_MAPPINGS 1 
#endif

#ifndef GENERATE_REDUCED_CONFIGS
#define GENERATE_REDUCED_CONFIGS 1
#endif

// The mappings are strictly ranked in macro-tile sizes (gemm_m_per_block, gemm_n_per_block)
static xdlops_mapping_t xdlops_mappings_fp16[] = {
        { 256, 128,  64,  32,  4, 4,  2,  2,  1,  1,  },
        { 256, 128,  32,  32,  8, 4,  2,  2,  2,  1,  },
        { 128, 256,  32,  64,  4, 4,  2,  2,  1,  1,  },
        { 128, 256,  32,  32,  8, 4,  2,  2,  1,  2,  },
        { 256, 64 ,  64,  16,  4, 4,  2,  2,  1,  1,  },
        { 128, 128,  32,  32,  4, 4,  2,  2,  1,  1,  },
        { 128, 128,  32,  32,  8, 4,  2,  2,  1,  1,  },
        { 128, 128,  16,  16, 16, 4,  2,  2,  2,  2,  },	
#if USE_REDUCED_XDLOPS_MAPPINGS	== 0
        { 128, 128,  32,  64,  4, 4,  1,  1,  2,  1,  },
#endif	
        { 64 , 256,  16,  64,  4, 4,  2,  2,  1,  1,  },
#if USE_REDUCED_XDLOPS_MAPPINGS == 0	
        { 64 , 256,  32,  64,  4, 4,  1,  1,  1,  2,  },
#endif
        { 64 , 256,  32,  32,  8, 4,  2,  2,  1,  1,  }, 
        { 256, 32 ,  64,  4 ,  4, 4,  2,  2,  1,  2,  },
        { 128, 64,   16,  16, 16, 4,  2,  2,  2,  1,  },
        { 128, 64 ,  32,  8 ,  4, 4,  2,  2,  1,  2,  },
        { 64 , 128,  8 ,  32,  4, 4,  2,  2,  2,  1,  },
#if USE_REDUCED_XDLOPS_MAPPINGS	== 0
        { 64 , 128,  32,  64,  4, 4,  1,  1,  1,  1,  },
        { 64 , 128,  64,  32,  4, 4,  1,  1,  1,  1,  },
#endif	
        { 64 , 128,  32,  32,  8, 4,  1,  1,  1,  2,  },
        { 32 , 256,  4 ,  64,  4, 4,  2,  2,  2,  1,  },
        { 256, 16 ,  64,  4 ,  4, 4,  2,  2,  1,  1,  },
        { 128, 32 ,  32,  8 ,  4, 4,  2,  2,  1,  1,  },
        { 64 , 64 ,  16,  16,  4, 4,  2,  2,  1,  1,  },
#if USE_REDUCED_XDLOPS_MAPPINGS == 0	
        { 64 , 64 ,  16,  16, 16, 4,  2,  2,  1,  1,  },
        { 64 , 64 ,  16,  16, 16, 4,  1,  1,  2,  2,  },
#endif
        { 32 , 128,  8 ,  32,  4, 4,  2,  2,  1,  1,  },
#if USE_REDUCED_XDLOPS_MAPPINGS	== 0
        { 32 , 128,  16,  64,  4, 4,  1,  1,  1,  1,  },
#endif
        { 16 , 256,  4 ,  64,  4, 4,  2,  2,  1,  1,  },
        { 128, 16 ,  64,  16,  4, 2,  1,  1,  1,  1,  },
        { 64 , 32 ,  32,  8 ,  4, 4,  1,  1,  1,  2,  },
        { 32 , 64 ,  8 ,  32,  4, 4,  1,  1,  2,  1,  },
        { 16 , 128,  16,  64,  4, 2,  1,  1,  1,  1,  },
        { 64 , 16 ,  64,  4 ,  4, 4,  1,  1,  1,  1,  },
#if USE_REDUCED_XDLOPS_MAPPINGS == 0	
        { 64 , 16 ,  64,  4 ,  4, 2,  1,  1,  1,  2,  },
#endif
        { 32 , 32 ,  16,  16,  4, 4,  1,  1,  1,  1,  },
        { 32 , 32 ,  16,  16, 16, 4,  1,  1,  1,  1,  },
        { 16 , 64 ,  4 ,  64,  4, 4,  1,  1,  1,  1,  },
#if USE_REDUCED_XDLOPS_MAPPINGS	== 0
        { 16 , 64 ,  4 ,  64,  4, 2,  1,  1,  2,  1,  },
#endif
        { 64 , 8  ,  64,  4 ,  4, 2,  1,  1,  1,  1,  },
        { 32 , 16 ,  32,  8 ,  4, 2,  1,  1,  1,  1,  },
        { 32 , 16 ,  32,  8 ,  4, 1,  1,  1,  1,  2,  },
        { 16 , 32 ,  8 ,  32,  4, 2,  1,  1,  1,  1,  },
        { 16 , 32 ,  8 ,  32,  4, 1,  1,  1,  2,  1,  },
        { 8  , 64 ,  4 ,  64,  4, 2,  1,  1,  1,  1,  },
        { 64 , 4  ,  64,  4 ,  4, 1,  1,  1,  1,  1,  },
        { 16 , 16 ,  16,  16,  4, 1,  1,  1,  1,  1,  },
        { 4  , 64 ,  4 ,  64,  4, 1,  1,  1,  1,  1,  },
}; 

static xdlops_mapping_t xdlops_mappings_fp32[] = {
        { 256, 128,  64,  32,  1, 4,  2,  2,  1,  1,  },
        { 256, 128,  32,  32,  2, 4,  2,  2,  2,  1,  },
        { 128, 256,  32,  64,  1, 4,  2,  2,  1,  1,  },
        { 128, 256,  32,  32,  2, 4,  2,  2,  1,  2,  },
        { 256, 64 ,  64,  16,  1, 4,  2,  2,  1,  1,  },
        { 256, 64 ,  32,  32,  2, 4,  2,  2,  1,  1,  },
        { 64 , 256,  16,  64,  1, 4,  2,  2,  1,  1,  },
        { 64 , 256,  32,  32,  2, 4,  2,  2,  1,  1,  },
        { 256, 32 ,  64,  4 ,  1, 4,  2,  2,  1,  2,  },
        { 256, 32 ,  32,  32,  2, 4,  2,  1,  1,  1,  },
        { 32 , 256,  4 ,  64,  1, 4,  2,  2,  2,  1,  },
        { 32 , 256,  32,  32,  2, 4,  1,  2,  1,  1,  },
        { 256, 16 ,  64,  4 ,  1, 4,  2,  2,  1,  1,  },
        { 16 , 256,  4 ,  64,  1, 4,  2,  2,  1,  1,  },
        { 128, 128,  32,  32,  1, 4,  2,  2,  1,  1,  },
        { 128, 128,  32,  32,  2, 4,  2,  2,  1,  1,  },
        { 128, 128,  32,  64,  1, 4,  1,  1,  2,  1,  },
        { 128, 64 ,  32,  8 ,  1, 4,  2,  2,  1,  2,  },
        { 128, 64 ,  32,  32,  2, 4,  2,  1,  1,  1,  },
        { 64 , 128,  8 ,  32,  1, 4,  2,  2,  2,  1,  },
        { 64 , 128,  32,  64,  1, 4,  1,  1,  1,  1,  },
        { 64 , 128,  64,  32,  1, 4,  1,  1,  1,  1,  },
        { 64 , 128,  32,  32,  2, 4,  1,  2,  1,  1,  },   
        { 128, 32 ,  32,  8 ,  1, 4,  2,  2,  1,  1,  },
        { 128, 32 ,  16,  16,  4, 4,  2,  2,  1,  1,  },
        { 32 , 128,  8 ,  32,  1, 4,  2,  2,  1,  1,  },
        { 32 , 128,  16,  64,  1, 4,  1,  1,  1,  1,  },
        { 32 , 128,  16,  16,  4, 4,  2,  2,  1,  1,  },
        { 64 , 64 ,  16,  16,  1, 4,  2,  2,  1,  1,  },
        { 64 , 64 ,  16,  16,  4, 4,  2,  2,  1,  1,  },
        { 64 , 64 ,  32,  32,  2, 4,  1,  1,  1,  1,  },  
        { 128, 16 ,  64,  16,  1, 2,  1,  1,  1,  1,  }, 
        { 128, 16 ,  16,  16,  4, 4,  2,  1,  1,  1,  },
        { 16 , 128,  16,  64,  1, 2,  1,  1,  1,  1,  }, 
        { 16 , 128,  16,  16,  4, 4,  1,  2,  1,  1,  }, 
        { 64 , 32 ,  32,  8 ,  1, 4,  1,  1,  1,  2,  },
        { 64 , 32 ,  16,  16,  4, 4,  2,  1,  1,  1,  },
        { 32 , 64 ,  8 ,  32,  1, 4,  1,  1,  2,  1,  },
        { 32 , 64 ,  16,  16,  4, 4,  1,  2,  1,  1,  },
        { 32 , 32 ,  16,  16,  1, 4,  1,  1,  1,  1,  },
        { 32 , 32 ,  16,  16,  4, 4,  1,  1,  1,  1,  },
        { 64 , 16 ,  64,  4 ,  1, 4,  1,  1,  1,  1,  },
        { 64 , 16 ,  16,  16,  4, 4,  1,  1,  1,  1,  },
        { 64 , 16 ,  16,  16,  4, 2,  2,  1,  1,  1,  },
        { 16 , 64 ,  4 ,  64,  1, 4,  1,  1,  1,  1,  },
        { 16 , 64 ,  16,  16,  4, 4,  1,  1,  1,  1,  },
        { 16 , 64 ,  16,  16,  4, 2,  1,  2,  1,  1,  },
        { 64 , 16 ,  64,  4 ,  1, 2,  1,  1,  1,  2,  },
        { 16 , 64 ,  4 ,  64,  1, 2,  1,  1,  2,  1,  },
        { 64 , 8  ,  64,  4 ,  1, 2,  1,  1,  1,  1,  },
        { 8  , 64 ,  4 ,  64,  1, 2,  1,  1,  1,  1,  },
        { 32 , 16 ,  32,  8 ,  1, 2,  1,  1,  1,  1,  },
        { 32 , 16 ,  16,  16,  4, 2,  1,  1,  1,  1,  },
        { 16 , 32 ,  8 ,  32,  1, 2,  1,  1,  1,  1,  },
        { 16 , 32 ,  16,  16,  4, 2,  1,  1,  1,  1,  },
        { 32 , 16 ,  32,  8 ,  1, 1,  1,  1,  1,  2,  },
        { 16 , 32 ,  8 ,  32,  1, 1,  1,  1,  2,  1,  },
        { 64 , 4  ,  64,  4 ,  1, 1,  1,  1,  1,  1,  },
        { 4  , 64 ,  4 ,  64,  1, 1,  1,  1,  1,  1,  },
        { 16 , 16 ,  16,  16,  1, 1,  1,  1,  1,  1,  },
        { 16 , 16 ,  16,  16,  4, 1,  1,  1,  1,  1,  }
};

#define NUM_XDLOPS_MAPPING_FP32 (sizeof(xdlops_mappings_fp32)/sizeof(xdlops_mapping_t))
#define NUM_XDLOPS_MAPPING_FP16 (sizeof(xdlops_mappings_fp16)/sizeof(xdlops_mapping_t))

static const int waveSize = 64; 

static inline void output_single_config(const igemm_gtc_tunable_t & cfg, const std::string & direction, const std::string & precision, const std::string & layout,
	                                const char *tensor_a_desc, const char *tensor_b_desc, std::ostream &myout)
{
         assert( direction == cfg.direction && precision == cfg.precision && layout == cfg.tensor_layout ); 

         myout << "#--------------------------- " << cfg.gemm_m_per_block << "x" << cfg.gemm_n_per_block << std::endl;
         
	 const char *sectionMark;
	
	 if ( direction == "fwd" )
	     sectionMark = "[igemm_fwd_gtc]";
	 else 
	     sectionMark = "[igemm_bwd_gtc]";

         myout << sectionMark  << std::endl;
         myout << "gemm_m_per_block         = " << cfg.gemm_m_per_block << std::endl;
         myout << "gemm_n_per_block         = " << cfg.gemm_n_per_block << std::endl;
         myout << "gemm_k_per_block         = " << cfg.gemm_k_per_block << std::endl;
         myout << "wave_tile_m              = " << cfg.wave_tile_m << std::endl;
         myout << "wave_step_m              = " << cfg.wave_step_m << std::endl;
         myout << "wave_repeat_m            = " << cfg.wave_repeat_m << std::endl;
         myout << "wave_tile_n              = " << cfg.wave_tile_n << std::endl;
         myout << "wave_step_n              = " << cfg.wave_step_n << std::endl;
         myout << "wave_repeat_n            = " << cfg.wave_repeat_n << std::endl;

         if ( precision == "fp16" ) 
              myout << "wave_tile_k              = " << cfg.wave_tile_k << std::endl;

	 std::string tensor_a_comment =  std::string("    #  ") + tensor_a_desc; 
	 std::string tensor_b_comment =  std::string("    #  ") + tensor_b_desc; 

         myout << "tensor_a_thread_lengths  = [" << cfg.tensor_a_thread_lengths[0] <<  ", " << cfg.tensor_a_thread_lengths[1] << ", ";
         myout << cfg.tensor_a_thread_lengths[2] << ", " << cfg.tensor_a_thread_lengths[3]   << "]" << tensor_a_comment << std::endl;

         myout << "tensor_a_cluster_lengths = [" << cfg.tensor_a_cluster_lengths[0] << ", " << cfg.tensor_a_cluster_lengths[1] << ", ";
         myout << cfg.tensor_a_cluster_lengths[2] << ", " << cfg.tensor_a_cluster_lengths[3] << "]" << tensor_a_comment << std::endl;

         myout << "tensor_b_thread_lengths  = [" << cfg.tensor_b_thread_lengths[0] <<  ", " << cfg.tensor_b_thread_lengths[1] << ", ";
         myout << cfg.tensor_b_thread_lengths[2] << ", " << cfg.tensor_b_thread_lengths[3]   << "]" << tensor_b_comment << std::endl;

         myout << "tensor_b_cluster_lengths = [" << cfg.tensor_b_cluster_lengths[0] << ", " << cfg.tensor_b_cluster_lengths[1] << ", ";
         myout << cfg.tensor_b_cluster_lengths[2] << ", " << cfg.tensor_b_cluster_lengths[3] << "]" << tensor_b_comment << std::endl;

         myout << "tensor_layout            = " << '\'' << cfg.tensor_layout << '\'' << std::endl; 

         myout << "direction                = " << '\'' << direction << '\'' << std::endl;
         myout << "precision                = " << '\'' << precision << '\'' << std::endl;

         myout << "nxb                      = " << cfg.nxb << std::endl;
         myout << "nxe                      = " << cfg.nxe << std::endl;
};

static void output_configurations(std::vector<igemm_gtc_tunable_t> &configs, const char *tensor_a_desc, const char *tensor_b_desc, std::ostream &myout)
{
    static const char *arch="\'gfx908\'";
    static const char *code_object="\'cov3\'";
    static const char *mode = "\'flat\'";

    myout << "[codegen]" << std::endl;
    myout << "arch = " << arch << std::endl;
    myout << "code_object = " << code_object << std::endl;
    myout << "mode = " << mode << std::endl;

    myout << std::endl;

    if (configs.size() <= 0)
	return; 

    std::string direction(configs[0].direction);
    std::string precision(configs[0].precision);
    std::string layout(configs[0].tensor_layout);  

    for (const auto& cfg : configs) {
         myout << std::dec;
         myout << std::endl;
         output_single_config(cfg, direction, precision, layout, tensor_a_desc, tensor_b_desc, myout);
    };
};

class basic_igemm_config
{
public:
    basic_igemm_config() = default;
    ~basic_igemm_config() = default;

    basic_igemm_config(const basic_igemm_config&) = delete;
    basic_igemm_config& operator=(basic_igemm_config&) = delete;

    virtual void generate_configs(const char *precision, const char *config_file) = 0;
private:
};

struct basic_config_sorter
{
public:
   virtual bool operator()(igemm_gtc_tunable_t &cfg1, igemm_gtc_tunable_t &cfg2) { return true; };  
}; 

#endif

