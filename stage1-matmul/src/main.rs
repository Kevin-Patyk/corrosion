//! Stage 1a - Matrix Multiplication (naive)
//! 
//! PMPP: Chapter 3 (multidimensional grids and data), section on matrix multiplication.
//! This is the first 2D kernel; the image filters come next and reuse everything here.
//! Stage 6 makes this same kernel fast (tiling).
//! 
//! ## What changes from Stage 0
//! 
//! Stage 0's data was one flat row, so each thread needed one number: its seat.
//! A matrix has rows and columns, so each thread gets a (row, column). Each block
//! is a square tile of threads, and the grid lays the tiles over the output matrix.
//! 
//! The CPU-side procedure is unchanged. Everything new is about the second dimension.
//! 
//! ## What is matrix multiplication?
//! 
//! C = A X B, all three N x N. Each element C[row][col] is row `row` of A times column
//! `col` of B: multiply them pair by pair and add them up.
//! 
//! C[row][col] = A[row][0] * B[0][col] + A[row][1] * B[1][col] + ...
//!
//! Every C element is independent of the others, so: one thread per C element. 
//! 
//! ## How a matrix sits in memory
//! 
//! Memory is one long line; it has no rows. Matrices are stored ROW-MAJOR:
//! row 0, then row 1, and so on. Element (row, col) is at
//! 
//! index = row * N + col
//! 
//! The numbers are the formula are "strides": how far to jump in memory for one step
//! in each direction. Row-major: column stride 1, row stride N. Column-major swaps them.
//! The hardware only sees addresses; the strides WE use are what make the data "row-major". 
//! Turning (row, col) into one memory index this way is called linearizing.
//! 
//! In matmul both direction appear: walking along A's row is stride 1 (neighbours in memory),
//! walking down B's column is stride N (a whole row apart each step). Same formula, different
//! directions of travel.
//! 
//! ## Threads in 2D
//! 
//! Block number and position in the block now have an x and y part:
//! 
//!     col = block.x * block width + position.x (x runs across)
//!     row = block.y * block height + position.y (y runs across)
//! 
//! x is the COLUMN, y is the ROW (CUDA and PMPP's convention).
//! 
//! Why x is the column: threads with neighbouring x read at the same time,
//! and memory hands over neighbouring bytes in one go. With x -> column and row-major data,
//! neighbouring threads read neighbouring memory, which is fast. Following the convention
//! gets this automatically. 
//! 
//! Blocks are 16 x 16 = 256 threads. The grid rounds UP in both directions,
//! so edge blocks hang over the matrix, and the bounds check has 2 parts:
//! col < N and row < N. N is deliberately not a multiple of 16, so the check matters 
//! (Stage 0's 1000-not-1024 idea, in 2D).
//! 
//! 

fn main() {
    println!("Hello, world!");
}
