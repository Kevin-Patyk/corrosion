# corrosion

GPU programming in Rust, from first kernel to tensor cores.

A ten-stage learning path through CUDA, written entirely in Rust with
NVIDIA's cuda-oxide: the classic parallel algorithms, kernel
optimization, and a small autograd engine at the end. Each stage is
its own crate in this workspace, with Programming Massively Parallel
Processors (PMPP) as the theory track.

| Stage | Project                                | Stack               |
|-------|----------------------------------------|---------------------|
| 0     | Hello, buffer                          | cuda-oxide          |
| 1     | Image filters                          | cuda-oxide + image  |
| 2     | Path tracer                            | cuda-oxide + image  |
| 3     | Bandwidth benchmark, histogram         | cuda-oxide          |
| 4     | Reduction, scan, radix sort            | cuda-oxide          |
| 5     | N-body                                 | cuda-oxide + image  |
| 6     | Tiled matmul                           | cuda-oxide          |
| 7     | The CUDA ladder                        | cuda-oxide + Nsight |
| 8     | Baby PyTorch                           | cuda-oxide          |
| 9     | Reading PTX, the compiler, drivers     | —                   |

Requires an NVIDIA GPU, CUDA toolkit 13.0+, clang-21 and cargo-oxide.
The pinned nightly toolchain is set in `rust-toolchain.toml`; run
`cargo oxide doctor` to check your setup.

## License

MIT OR Apache-2.0