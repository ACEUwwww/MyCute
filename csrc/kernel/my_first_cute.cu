#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/util/print.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(status)                \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

using namespace cute;

static constexpr int kM = 16;
static constexpr int kN = 32;
static constexpr int kK = 8;
static constexpr int kBlockM = 8;
static constexpr int kBlockN = 16;
static constexpr int kBlockK = 8;

__global__ void my_cute_gemm(const float* A, const float* B, float* C) {
    using namespace cute;

    // global Tensor Encode
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(Int<kM>{}, Int<kK>{}),
                            make_stride(Int<kK>{}, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(Int<kN>{}, Int<kK>{}),
                            make_stride(Int<1>{}, Int<kN>{}));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(Int<kM>{}, Int<kN>{}),
                            make_stride(Int<kN>{}, Int<1>{}));
                    
    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    // To tile view
    Tensor gA = local_tile(mA, make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                            make_coord(tile_m, Int<0>{}));
    Tensor gB = local_tile(mB, make_shape(Int<kBlockN>{}, Int<kBlockK>{}),
                            make_coord(tile_n, Int<0>{}));
    Tensor gC = local_tile(mC, make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                            make_coord(tile_m, tile_n));

    auto tiled_mma = make_tiled_mma(UniversalFMA<float, float, float, float>{},
                                    Layout<Shape<Int<kBlockM>, Int<kBlockN>, _1>>{});
    
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    Tensor tAgA = thr_mma.partition_A(gA);
    Tensor tBgB = thr_mma.partition_B(gB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tArA = thr_mma.make_fragment_A(tAgA);
    Tensor tBrB = thr_mma.make_fragment_B(tBgB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    clear(tCrC);

    copy(tAgA, tArA);
    copy(tBgB, tBrB);

    gemm(thr_mma, tArA, tBrB, tCrC);

    copy(tCrC, tCgC);

}


int main() {
    static int ElementA = kM * kK;
    static int ElementB = kK * kN;
    static int ElementC = kM * kN;

    std::vector<float> hA(ElementA);
    std::vector<float> hB(ElementB);
    std::vector<float> hC(ElementC);
    std::vector<float> ref(ElementC);

    for (int m = 0; m < kM; ++m) {
        for (int k = 0; k < kK; ++k) {
            hA[m * kK + k] = static_cast<float>((m + 1) * 0.25f +k);
        }
    }

    for (int k = 0; k < kK; ++k) {
        for (int n = 0; n < kN; ++n) {
            hB[k * kN + n] = static_cast<float>((n + 1) * 0.125f - k * 0.5f);
        }
    }

    for (int m = 0; m < kM; ++m) {
        for (int n = 0; n < kN; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < kK; ++k) {
                acc += hA[m * kK + k] * hB[k * kN + n];
            }
            ref[m * kN + n] = acc;
        }
    }


    float* dA;
    float* dB;
    float* dC;
    CHECK_CUDA(cudaMalloc(&dA, ElementA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, ElementB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, ElementC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), ElementA * sizeof(float),
                        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), ElementB * sizeof(float),
                        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, ElementC * sizeof(float)));


    dim3 block(kBlockM * kBlockN);
    dim3 grid(kN / kBlockN, kM / kBlockM);
    my_cute_gemm<<<grid, block>>>(dA, dB, dC);


    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, ElementC * sizeof(float),
                            cudaMemcpyDeviceToHost));

    int errors = 0;
    float max_abs_diff = 0.0f;
    for (int idx = 0; idx < ElementC; ++idx) {
        float diff = std::fabs(hC[idx] - ref[idx]);
        max_abs_diff = std::max(max_abs_diff, diff);
        if (diff > 1.0e-4f) {
        if (errors < 5) {
            std::cerr << "mismatch idx=" << idx << " got=" << hC[idx]
                    << " expected=" << ref[idx] << " diff=" << diff << "\n";
        }
        ++errors;
        }
    }


    auto tiled_mma =
      make_tiled_mma(UniversalFMA<float, float, float, float>{},
                     Layout<Shape<Int<kBlockM>, Int<kBlockN>, _1>>{});

    std::cout << "problem: C(" << kM << "," << kN << ") = A(" << kM << ","
                << kK << ") * B(" << kK << "," << kN << ")\n";
    std::cout << "CTA tile: (" << kBlockM << "," << kBlockN << "," << kBlockK
                << "), block=" << block.x << ", grid=(" << grid.x << ", "
                << grid.y << ")\n";
    std::cout << "tiled_mma: ";
    print(tiled_mma);
    std::cout << "\n";

    if (errors == 0) {
        std::cout << "PASS: max_abs_diff=" << max_abs_diff << "\n";
    } else {
        std::cout << "FAIL: errors=" << errors << " max_abs_diff=" << max_abs_diff
                << "\n";
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return errors == 0 ? 0 : 1;

}
