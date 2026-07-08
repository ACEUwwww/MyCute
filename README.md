# MyCute

This is a small practice repository for hand-writing CUTE/CUTLASS examples.

Current files:

- `my_first_cute.cu`: first CUTE GEMM practice file.

Build example:

```bash
nvcc -std=c++17 --expt-relaxed-constexpr \
  -I/mdata/pretrain/shiweisong/willion/third-party/cutlass/include \
  my_first_cute.cu -o my_first_cute
```

The generated executable `my_first_cute` is ignored by git.
