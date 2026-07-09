#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

namespace mycute{
    inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
        if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(status)
                    << " from " << expr
                    << " at " << file << ":" << line << "\n";
        std::exit(EXIT_FAILURE);
        }
    }

    }  // namespace mycute

#define MYCUTE_CHECK_CUDA(call) \
    ::mycute::check_cuda((call), #call, __FILE__, __LINE__)
