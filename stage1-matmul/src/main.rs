//! Stage 1a - Matrix Multiplication (naive)
//! 
//! PMPP: Chapter 3 (multidimensional grids and data), section on matrix multiplication.
//! This is the first 2D kernel; the image filters come next and reuse everything here.
//! Stage 6 makes this same kernel fast (tiling).
//! 
//! ## What changes from Stage 0
//! 
//! Stage 0's data was one flat row, so each thread needed one number: its seat.
//! A matrix has rows and column, so here each thread gets a (row, col). Each block
//! is a square tile of threads, and the grid lats the files over the output matrix.
//! 
//! The CPU-side procedure is unchanged. Everything new is about the second dimension.
//! 
//! ## What is matrix multiplication?
//! 
//! C = A x B, all three WIDTH x WIDTH (square). Each element C[row][col] is
//! row `row` of A times column `col` of B: multiply them pair by pair and 
//! add up.
//! 
//!     C[row][col] = A[row][0] * B[0][col]
//!                 + A[row][1] * B[1][col]
//!                 + ...
//!                 + A[row][WIDTH-1] * B[WIDTH-1][col]
//! 
//! No C element depends on another, so: one thread per C element. Each thread
//! reads a whole row of A and whole column of B, and writes ONE element of C.
//! 
//! ## How a matrix sits in memory
//! 
//! Memory is one long line; it has no rows. Matrices are stored ROW-MAJOR:
//! row 0, then row 1, and so on. Element (row, col) is at:
//! 
//!     index = row * width + col
//! 
//!     row-major:  row stride = width, column stride = 1
//!     column-major: row stride = 1, column stride = height
//! 
//! The hardware only sees addresses; the strides WE use in the formula are what make the
//! data "row-major".
//! 
//! In matmul, both directions show up. A and B are stored the same way, but walking along a row of A
//! is +1 per step, while walking down a column of B skips a whole row per step: +width.
//! 
//! ## Threads in 2D
//! 
//! Block number and position in the block now each have an x and a y part. 
//! The GLOBAL INDEX formula, one per direction:
//! 
//!     col = block.x * block width + position.x (x runs ACROSS)
//!     row = block.y * block height + position.y (y runs DOWN)
//! 
//! x is the COLUMN, y is the ROW (CUDA and PMPP's convention).
//! 
//! Each thread does two steps, in order:
//! 
//!     1. Mapping: which element is my job? -> (row, col)
//!        (global index formula)
//!     2. Linearizing: where does it live in memory? -> row * width + col
//!        (strides)
//! 
//! Mapping happens once: the thread turns its block and position numbers into the
//! element it's responsible for. Linearizing happens every time the thread reads or 
//! write memory: its C element once, and each A and B element in the loop. The
//! matrices are already stored flat; linearizing just finds each element's spot in the line.
//! 
//! In Stage 0 both collapsed into index_1d: the data was already flat, so seat =
//! element = memory index.  In 2D they come apart.
//! 
//! Why x is the column: threads next to each other in x run together, and memory
//! hands over neighbouring bytes in one go. With x -> col and row-major data,
//! neighbouring threads touch neighbouring memory, which is fast. (For column-major
//! data it would flip: x -> row.) This only affects speed, never correctness.
//! More in PMPP ch 4. and 6.
//! 
//! ## Launch shape and the bounds check
//! 
//! Blocks are 16 x 16 = 256 threads (a multiple of 32, like Stage 0's 256).
//! The grid rounds UP in both directions:
//! 
//!     blocks_x = width.div_ceil(16)
//!     blocks_y = height.div_ceil(16)
//! 
//! so edge blocks hang over the right and bottom of the matrix. WIDTH is deliberately
//! not a multiple of 16, so those idle threads really exist.
//! 
//! The bounds check has two parts: col < width, AND row < height. It must happen
//! BEFORE linearizing: a too-big col doesn't give an out-of-range index, it wraps into the next row
//! and lands on someone else's element.
//! 
//! In short: the bounds check makes sure only threads that have a real element do any work.
//! The grid always has at least as many threads as elements, so the extras must sit out.
//! Each thread checks the (row, col) it got from mapping:
//! 
//!     thread[col] < width AND thread[row] < height -> work
//!     otherwise -> do nothing
//! 
//! Order: map -> check -> linearize
//! 
//! Mapping is just arithmetic. It gives every thread available a (row, col) without knowing how big the
//! matrix is. In a 5x5 matrix example with an 8x8 grid, mapping still hands out positions like (row 0, col 6)
//! or (row 6, col 2), and those don't exist in the matrix. That's why the check comes right after mapping:
//! it's the first point where a thread can tell its (row, col) is off the matrix, and it stops before trying
//! to touch memory.
//! 
//! ## What are we achieving in this file?
//! 
//!     Part 1: context + stream, make A and B on the CPU, copy them over, make a zeroed box for C, load the kernel
//!     Part 2: the kernel, a 2D launch config, prepare, launch
//!     Part 3: copy C back, check it against the same multiplication done on the CPU
//! 
//! In cuda-oxide: #[launch_contract(domain = 2, ...)], LaunchConfig2D, and
//! a 2D thread index from the `thread` module (it does the mapping and the
//! col < width check for us). Details where they're used.

// cuda_device = the GPU-side crate: things used INSIDE of kernels
// DisjointSlice: an output array where each thread may write only its own element
// kernel: the attribute that marks the function as GPU code
// launch bounds,
// launch contract: attributes describing how the kernel may be launched
// thread: functions that tell a thread where it is (index_2d etc.)
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]

fn main() {
    println!("Hello, world!");
}
