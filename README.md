# corrosion

GPU programming in Rust, from first dispatch to tensor cores.

A nine-stage learning path: compute shaders in wgpu, the classic
parallel algorithms, CUDA kernel optimization with cudarc, and a
small autograd engine at the end. Each stage is its own crate in
this workspace.

| Stage | Project                          | Stack        |
|-------|----------------------------------|--------------|
| 0     | Hello, buffer                    | wgpu         |
| 1     | Image filters                    | wgpu         |
| 2     | Path tracer                      | wgpu         |
| 3     | Bandwidth benchmark, histogram   | wgpu         |
| 4     | Reduction, scan, radix sort      | wgpu         |
| 5     | N-body                           | wgpu + winit |
| 6     | Tiled matmul                     | wgpu, CubeCL |
| 7     | The CUDA ladder                  | cudarc       |
| 8     | Baby PyTorch                     | cudarc       |
| 9     | Reading PTX, naga, drivers       | —            |

Requires an NVIDIA GPU and the CUDA toolkit for stages 7–8.

## License

MIT OR Apache-2.0
