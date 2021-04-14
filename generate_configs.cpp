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
#include <string>

#include "bwd_nchw_config.hpp"
#include "bwd_nhwc_config.hpp"
#include "fwd_nchw_config.hpp"

int main(int argc, char **argv)
{
    if ( argc != 5 ) {
         fprintf(stdout, "Usage: %s, <direction(fwd,bwd,wrw)> <precision(fp32,fp16)> <layout(nchw,nhwc)> <output configuration file> \n", argv[0]);
         return(-1);
    };

    std::string direction(utility_lower_string(argv[1]));
    std::string precision(utility_lower_string(argv[2]));
    std::string layout(utility_lower_string(argv[3])); 

    if ( direction != "fwd" && direction != "bwd" && direction != "wrw") {
         std::cout <<  "Invalid convolution direction!" << std::endl;
         return(-2);
    };

    if ( precision != "fp32" && precision != "fp16" ) {
         std::cout <<  "Invalid data precison!" << std::endl;
         return(-2);
    }; 

    if ( layout != "nchw" && layout != "nhwc" ) {
         std::cout <<  "Invalid tensor layout!" << std::endl;
         return(-2);
    }; 

    const char *config_file = argv[4];

    std::unique_ptr<basic_igemm_config> pConfig;  

    if ( direction == "bwd" && layout == "nchw" ) 
         pConfig.reset( new bwd_nchw_config() ); 

    if ( direction == "bwd" && layout == "nhwc" ) 
         pConfig.reset( new bwd_nhwc_config() ); 

    if ( direction == "fwd" && layout == "nchw" ) 
         pConfig.reset( new fwd_nchw_config() ); 

    pConfig->generate_configs(precision.c_str(), config_file); 
}; 

