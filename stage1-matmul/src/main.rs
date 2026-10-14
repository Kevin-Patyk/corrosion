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

// cuda_core: the CPU-side CUDA library (talks to the driver).
// CudaContext = our connection to one GPU
// DeviceBuffer = an array living in GPU memory
// LaunchConfig2D = the launch shape for a 2D kernel:
//      how many blocks (grid) and threads per block, in x (cols) and y (rows)
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
// cuda_device = the GPU-side crate: things used INSIDE of kernels
// DisjointSlice: an output array where each thread may write only its own element
// kernel: the attribute that marks the function as GPU code
// launch bounds,
// launch contract: attributes describing how the kernel may be launched
// thread: functions that tell a thread where it is (index_2d etc.)
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
// cuda_host = the CPU-side crate. cuda_module is the attribute that turns a Rust
// module into a loadable GPU module
use cuda_host::cuda_module;

// Matrix side length. Deliberately NOT a multiple of 16, so the last block hangs off the edge and the bounds check
// actually gets exercised. A const (known at compile time) so it can be the stride in Index2D below.
const WIDTH: usize = 100;

// #[cuda_module]: everything inside this module is compiled for the GPU (to PTX) and embedded in the program.
// The macro also generates CPU-side helpers for it:
//  kernels::load(&ctx) -> loads the GPU code onto the device
//  module.prepare_matmul() -> checks the launch shape against the contract
// module.matmul(...) -> starts the kernel on a queue (stream)
// `kernels` is just the modules name; any name works, we call it via kernels::load
#[cuda_module]
mod kernels {
    // Plain Rust: bring WIDTH and the imports from the file into this module
    use super::*;

    // #[kernel]: this function is GPU code. The CPU only asks for it to 
    // start (later, in main); the body runs on the GPU, once per thread.
    // A kernel can't return a value: each thread writes its answer into 
    // GPU memory (here, its own element of `c`) and the CPU copies `c`
    // back afterwards.
    #[kernel]
    // Tells the compiler the most threads a block will ever have:
    // 16 x 16 = 256. Must match the block size in the contract.
    #[launch_bounds(256)] 
    // The launch contract: domain = 2 means a 2D launch (x and y used, z is 1).
    // block = (16, 16, 1) means the host must ask for exactly 16 x 16 blocks.
    // Because of this contract, the launch from the CPU can be safe (no unsafe).
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn matmul(
        // Inputs: read-only, row-major, WIDTH * WIDTH elements each.
        a: &[f32],
        b: &[f32],
        // Output: each thread writes exactly one element.
        // Index2D<WIDTH>: c is indexed by (row, col), row-major, row stride WIDTH.
        // Index2D<WIDTH> just says "c is indexed row-major with stride WIDTH."
        // The matching thread::index_2d::<WIDTH>() (used in the body) does two jobs:
        // 1. Linearizing: row * WIDTH + col -> this thread's one spot in c.
        // 2. Checks the column bound: col >= WIDTH gives None (no spot), so a
        // too-big col can't wrap into the next row. The row is NOT checked, we do that ourselves.
        // The formula is always row major; nothing "knows" the layout. A and B must be filled and
        // read with the same formula, or the results are wrong.
        // The stride is in the type, so using a different stride for c and the index is a compile error.
        // `mut` because we write into it.
        mut c: DisjointSlice<f32, thread::Index2D<WIDTH>>,
    ) {
        // --- How does a thread know where it is?
        // The GPU gives every thread its own built-in values (special registers, read-only):
        // blockIdx = which block it's in
        // threadIdx = its position in that block
        // blockDim = the block's size
        // thread::index_2d_col() just reads them and applies the mapping:
        //      col = blockIdx.x * blockDim.x + threadIdx.x
        //      row = blockIdx.y * blockDim.y + threadIdx.y
        // Same code in every thread, different values -> different (row, col).
        // `thread::` is only the module name.

        // --- 1. Map: which (row, col) of C belongs to this thread? ----
        // row = block.y * block height + position.y
        // col = block.x * block width + position.x
        // Just arithmetic: threads off the matrix get a (row, col) too.
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        // ---- 2. Bounds check, then linearize C's spot ----so 
        // index_2d takes no arguments: it reads this thread's position from the
        // GPU and works out row and col itself (same mapping formula as step 1).
        // Our `row` and `col` variables are a separate copy of the same numbers, used for reading A and B.
        // We can't pass col in on purpose: a wrong col could give two threads the same spot in C.
        // Reading it from the GPU guarantees every thread a unique spot, which is what makes c.get_mut safe.
        // It checks col < WIDTH, then computes row * WIDTH + col (linearizing).
        // None = too far right: this thread does nothing.
        // Our thread has a (row, col) in the matrix C, first we check if the col is within then width, then the row
        // Note: Both checks use WIDTH only because our matrix is square. For a non-square matrix, row would
        // be checked against the height instead.
        if let Some(c_idx) = thread::index_2d::<WIDTH>() {
            // index_2d does NOT Check the row, so we do it. It must come before the loop:
            // a too-big row would just read past the end of A.
            if row < WIDTH {
                // ---- 3. The job: C[row][col] = row `row` of A . column `col` of B ----
                // Multiply pairs and add them up (a dot product).
                let mut sum = 0.0f32;
                // Since each matrix is 100 x 100, each row of A and each column of B has 100 (WIDTH) elements
                // k is the step counter that walks through them, from 0 to 99
                for k in 0..WIDTH {
                    // Linearizing formula, once per read
                    // A: walk along the row `row` -> next element is +1
                    // B: walk down column `col` -> next element is +WIDTH
                    // Every a[...] / b[...] is range checked; out of range stops the 
                    // kernel with an error.
                    // We need to linearize and then index into a and b since a and b are plain
                    // slices: one line of memory with no rows.
                    // To find "the element at (row, col)" we need the linearizing formula, every time.
                    // A slice only understands "give me spot number n", so we have to turn it into a spot ourselves
                    // Because we launch in 2D, each thread gets a (row, col). Memory is flat, so that (row, col) is linearized
                    // (by index_2d) to find the spot in C the thread will fill. To fill it, the thread computes a dot product
                    // of row `row` of A and column `col` of B. A and B are flat too, so each position along that row and column 
                    // is linearized to find where it sits in memory.
                    // The position of (row, col) in C is used to determine which entire row is needed from A
                    // and which entire row is needed from B.
                    sum += a[row * WIDTH + k] * b[k * WIDTH + col];
                    // Memory is always flat (A, B, C alike). What matters is what we know:
                    // - a (row, col) position -> needs linearizing formula
                    // - a flat spot already -> no formula
                    // 2D launch: each threads gets (x, y) -> (col, row) so it thinks in
                    // (row, col) and must linearize. A[row][k] and B[k][col] are also
                    // (row, col) positions, in their own matrix -> same formula.
                }

                // ---- 4. Write this thread's one element of C ----
                // get_mut takes c_idx (the spot from step 2) and checks it against C's ACTUAL length.
                // Our row/col checks assume the CPU made c exactly WIDTH * WIDTH long, but the kernel can't see that:
                // if c were too small, the spot could pass our checks and still be pass the end.
                // get_mut is also the only safe way to write to a DisjointSlice (the no-check version is unsafe).
                // In our kernel None never happens: it's a safety net for a CPU-side bug. Cheap: once per thread.
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = sum;
                }
            }
        }
    }
}

// main returns a Result so we can use `?` below.
// `Ok(())` = everything worked. `Err(...)` = something failed.
// Box<dyn std::error::Error> means "any kind of error".
// The CUDA calls each have their own error type; this accepts all of them.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. Open the GPU + get the queue ----
    // Open GPU number 0 (we only have one).
    // Every later GPU step (memory, loading kernels) goes through ctx.
    // `?` = if this failed, stop main and return the error.
    let ctx = CudaContext::new(0)?;

    // The queue (CUDA calls its a "stream").
    // Copies and kernel launches are put into it and run in order.
    // default_stream() gives the context's ready-made queue.
    // This means we use the existing strea and do not have to create it ourselves
    // Every context comes with one queue already set up, called the default_stream
    // We can make extra streams ourselves, but will not do so right now.
    let stream = ctx.default_stream();
    // The CPU puts jobs into the queue (copy this data over, run this kernel, copy the result back)
    // and the GPU works through them in the order they were added.
    // Not everything goes throuygh the queue: setup steps like opening the GPU and loading kernels
    // go through the context.
    // Adding a job != job done. The CPU can add a job and move on before the GPU finished it.
    // That's why the order matters: the kernel can't start before the copy in front of it is done
    // and the copy back to the CPU waits for the kernel. 
    // Lastly, allocating happens immediately, not through the queue, but the copy or fill after
    // it allocates memory does go through the queue

    // ---- 2. Fill A and B on the CPU (host) ----
    // vec![0.0f32; n] = a Vec of n copies of 0.0 (f32) one flat line
    // `mut` because we change its elements below
    // _host = this copy lives in CPU memory (the GPU copies come next).
    let mut a_host = vec![0.0f32; WIDTH * WIDTH];
    let mut b_host = vec![0.0f32; WIDTH * WIDTH];

    // Walk the flat line spot by spot (no row/col needed).
    // % = remainder: keeps values small whole numbers (0..6, 0..12)
    // which f32 stores exactly -> GPU and CPU should match exactly
    // Different numbers (7, 13) so A and B differ and aren't symmetric,
    // so reading a wrong element gives a wrong value.
    // `as f32` turns the whole number (usize) into a float.
    for i in 0..WIDTH * WIDTH {
        a_host[i] = (i % 7) as f32;
        b_host[i] = (i % 13) as f32;
    }

    // ---- 3. Copy A and B to the GPU, make C -----
    // from_host: Reserve GPU memory, copy the CPU data into it,
    // and WAIT until the copy is done (so it's on the GPU when this returns).
    // from_host does 3 things before it returns:
    //      1. Reserve GPU memory
    //      2. Put "copy A over" in the queue
    //      3. Wait until the queue has finished that copy, then CPU moves on
    // &a_host = lend a_host (from_host reads from it). We keep a_host and 
    // use it again for the CPU check in step 5.
    // _dev = this copy lives in GPU (device) memory.
    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;

    // C: reserve WIDTH*WIDTH floats on the GPU, filled with zeros.
    // Nothing to copy: the kernel writes C.
    // Must be exactly WIDTH*WIDTH: the kernel's get_mut checks this length.
    // `mut` because the kernel writes into it (the launder takes &mut).
    // `: DeviceBuffer<f32>` tells Rust the element type (zerod has not data to guess it from).
    let mut c_dev: DeviceBuffer<f32> = DeviceBuffer::zeroed(&stream, WIDTH * WIDTH)?;

    // The above are fallible because they can fail (GPU out of memory, driver refuses the copy, etc.),
    // so they return a Result: Ok(buffer) or Err(error).
    // `?` = if Ok, take the buffer out; if Err, stop main and return the error.

    // ----- 4. Load the kernels, launch -----
    // load: give the GPU our compiled kernel code (PTX, built by cargo oxide)
    // unsafe: we promise the GPU Code bundle in this program is the one compiled
    // from our `kernels` module. cargo oxide makes sure of that, but Rust can't check
    // itself, so we must write `unsafe`.
    let module = unsafe { kernels::load(&ctx)? };

    // Blocks per direction. div_ceil = divide and round UP:
    // 100 / 16 = 6.25 -> 7 blocks -> 7 * 16 = 112 threads per direction.
    // 12 extra off the matrix (our bounds check skip them)
    // `as u32`: the launch takes u32 numbers
    let blocks = (WIDTH as u32).div_ceil(16);

    // LaunchConfig2D::new(grid, block, extra shared memory in bytes)
    // grid = (blocks in x, blocks in y) -> x = cols, y = rows
    // block = (16, 16) threads, must match the kernel's #[launch_contract]
    // 0 = no extra shared memory (on-chip block memory, ch. 5, unused)
    let config = LaunchConfig2D::new((blocks, blocks), (16, 16), 0);

    // prepare: checks the config against #[launch_contract] on the CPU,
    // before anything runs. Block not exactly (16, 16, 1) -> Err.
    // Also checks the GPU's limits (e.g. max threads per block).
    let prepared = module.prepare_matmul(config)?;
    // What gets checked and when:
    // - Build time: domain = 2 -> prepare_matmul only accepts a LaunchConfig2D; a, b, c must be the right buffer types
    // - prepare_matmul (run time, before anything runs): block exactly (16, 16, 1) per #[launch_contract], else Err;
    //      GPU limits (e.g. max threads per block).
    //      (launch_bounds(256) is mainly a promise to the compiler.)
    // - load checks nothing about the shape; the launch only checks the queue belongs to the same GPU.

    // Launch: put "run matmul" in the queue.
    // Arguments:  the queue, the checked config, then the kernel's arguments
    // in the same order as the kernel:
    //      a: &[f32] -> &a_dev (GPU reads only)
    //      b: &[f32] -> &b_dev (GPU reads only)
    //      c: DisjointSlice -> &mut c_dev (GPU writes)
    // Doesn't wait for the GPU to finish; the copy back in step 5 waits.
    module.matmul(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;

    // ---- 5. Copy C back, check against a CPU answer.
    // to_host_vec: put "copy C to the CPU" in the queue then WAIT.
    // The queue runs in order: the kernel finishes before the copy starts.
    // If the kernel crashed, the error shows up here (and `?` returns it).
    let c_host = c_dev.to_host_vec(&stream)?;

    // CPU reference: the same matmul, done by the CPU (slow, but simple).
    // For every element of C: the dot product of A's row and B's column,
    // same linearizing as the kernel.
    let mut mismatches = 0;
    for row in 0..WIDTH {
        for col in 0..WIDTH {
            let mut expected = 0.0f32;
            for k in 0..WIDTH {
                expected += a_host[row * WIDTH + k] * b_host[k * WIDTH + col];
            }
            // Compare. Fixed form from Stage 0: subtract FIRST, then .abs()
            // (the size of the difference, ignoring its sign).
            // 1e-5 = 0.00001, the allowed difference.
            let i = row * WIDTH + col;
            if (c_host[i] - expected).abs() > 1e-5 {
                // Print only the first 5, so a big failure doesn't flood the screen.
                if mismatches < 5 {
                    println!("mismatch at C[{row}][{col}]: GPU {}, CPU {expected}", c_host[i]);
                }
                mismatches += 1;
            }
        }
    }

    if mismatches == 0 {
        println!("SUCCESS: all {} elements match", WIDTH * WIDTH);
    } else {
        println!("FAILED: {mismatches} of {} elements wrong", WIDTH * WIDTH);
    }

    Ok(())
}
