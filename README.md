# MyCute

This is a small practice repository for hand-writing CUTE/CUTLASS examples.

Current files:

- `csrc/kernel/my_first_cute.cu`: first CUTE GEMM practice file.
- `csrc/kernel/cute_wmma.cu`: WMMA/Tensor Core CUTE GEMM practice file.

Build example:

```bash
nvcc -std=c++17 --expt-relaxed-constexpr \
  -I/mdata/pretrain/shiweisong/willion/third-party/cutlass/include \
  csrc/kernel/my_first_cute.cu -o my_first_cute
```

```bash
nvcc -std=c++17 --expt-relaxed-constexpr \
  -I/mdata/pretrain/shiweisong/willion/third-party/cutlass/include \
  -arch=sm_80 \
  csrc/kernel/cute_wmma.cu -o cute_wmma
```

The generated executables are ignored by git.
