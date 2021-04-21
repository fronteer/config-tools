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
#include <stdexcept>

#include "config_parser.hpp"
#include "igemm_gtc_base.hpp"

#include "bwd_nchw_config.hpp"
#include "bwd_nhwc_config.hpp"
#include "fwd_nchw_config.hpp"


static bool simpleSorter(int i,int j) { return (i>j); }; 

static std::vector<igemm_gtc_tunable_t> ordered_configs; 

int main(int argc, char **argv) 
{
    if ( argc != 3 ) {
         fprintf(stdout, "Usage: %s, <input configuration file> <output configuration file>\n", argv[0]);
         return(-1);
    };

    const char *config_file = argv[1];

    config_parser_t config_parser(config_file);
    auto content = config_parser.parse();
   
    std::ofstream ofs(argv[2], std::ofstream::out);

    auto tunables = igemm_gtc_tunable_from_config(content);
    if (tunables.size() == 0){
        fprintf(stdout, "no tunable specified, may not work\n");
        return 0;
    }
    fprintf(stdout, "tunables:%d\n", (int)tunables.size());

    std::string direction(tunables[0].direction);
    std::string precision(tunables[0].precision); 
    std::string layout(tunables[0].tensor_layout);

    std::cout << std::endl << "layout = " << layout << std::endl; 

    // "indexed_configs" is used to classify the configs according to the size of the macro-tile
    std::map< int, std::vector<igemm_gtc_tunable_t> > indexed_configs; 
    std::map< int, std::vector<igemm_gtc_tunable_t> >::iterator it;
    std::vector<int> mt_sizes; 

    int count=0; 
    for (const auto& tunable : tunables)  {
         assert(direction == tunable.direction && std::string(precision) == tunable.precision && layout == tunable.tensor_layout); 

         auto mt = tunable.gemm_m_per_block*tunable.gemm_n_per_block; 

         it = indexed_configs.find(mt); 

         if ( it == indexed_configs.end() ) {
              std::vector<igemm_gtc_tunable_t> tmpVector;

              indexed_configs.insert( std::make_pair(mt, tmpVector) );
              it = indexed_configs.find(mt);

              mt_sizes.push_back(mt);
         }

         assert(it != indexed_configs.end());

         count++;
         it->second.push_back(tunable);
    }

    fprintf(stdout, "%d configurations checked\n", count); 

    std::sort(mt_sizes.begin(), mt_sizes.end(), simpleSorter); 

    for (auto mt : mt_sizes) {
         it = indexed_configs.find(mt);

         if ( it != indexed_configs.end() ) {
              fprintf(stdout, "Macro-tile %d, number of configurations %d\n", mt, (int)it->second.size());

              if ( layout == "nchw" ) 
                   std::sort(it->second.begin(), it->second.end(), BwdNchwSorter);
              else 
              if ( layout == "nhwc" )  	
                   std::sort(it->second.begin(), it->second.end(), BwdNhwcSorter);

              for (const auto&  tunable : it->second)
                   ordered_configs.push_back(tunable);
         };
    };

    fprintf(stdout, "\nSize of the orderred configs array %d\n", (int)ordered_configs.size()); 

    if ( layout == "nchw" )
         output_configurations(ordered_configs, "k0xk1ExC0xC1", "K0xK1ExN0xN1B", ofs);
    else 
    if ( layout == "nhwc" )
         output_configurations(ordered_configs, "EK2K0xK1xN0xN1B","K0xK1K2ExC0xC1", ofs);
};

