//! Stage 0 - Vector Addition
//!
//! ## What is a GPU?
//!
//! A CPU has a few powerful cores (8-32), each built to run one complicated sequence of
//! instructions as fast as possible: big caches, branch prediction, out-of-order execution.
//!
//! A GPU makes the opposite trade. It has thousands of small, simple cores that all run the
//! SAME program at the same time, each on a different piece of data. Any single GPU core is
//! much slower than a CPU core, but if the work is "do this one thing to a million elements",
//! the GPU wins by sheer width. It also has its own memory with far higher bandwidth than
//! system RAM, because feeding thousands of cores is mostly a memory problem.
//!
//! Two consequences of everything in this roadmap:
//!     1. The GPU is a separate device with separate memory. Data has to be copied over
//!        to it and results copied back. Nothing is shared.
//!     2. The CPU does not call GPU functions. It launches work onto a stream (a queue of
//!        GPU work), and the GPU runs it asynchronously, later.
//!
//! Bandwidth: how many bytes per second can move between the memory and the cores, measured
//! in GB/s. System RAM manages roughly 50-100 GB/s; VRAM on a modern NVIDIA card manages
//! 500-1000+. Doubling a million floats is 4 MB in and 4 MB out and almost no arithmetic,
//! so the time it takes is set by bandwidth, not the core count.
//!
//! ## Why are we using it?
//!
//! For data-parallel work: the same operation applied independently to many elements.
//! The word "independently" is doing the work here. Computing element `i` must not need
//! the result of any other element, so the order doesn't matter and everything can happen
//! at once. Image filters, physics, sorting, matrix multiply, and neural networks all have
//! this shape. The function that runs once per element is called a kernel.
//!
//! NOTE: A useful test for any loop: could the iterations run in random order and still
//! give the same result? If yes, it's data parallel and maps straight onto a GPU. Stage 0
//! is deliberately the easiest case, so that the plumbing is the only hard part.
//!
//! We use cuda-oxide, NVIDIA's compiler for writing CUDA kernels in Rust. A normal Rust
//! function marked `#[kernel]` is compiled to PTX (NVIDIA's GPU assembly language), and the
//! host code (what the CPU runs) and the device code (the kernel) live in the same file.
//! PMPP's CUDA C examples map onto it directly.
//!
//! ## Terminology
//!
//! On a CPU you'd write a loop and put the per-element work in its body. On the GPU you
//! write ONLY the body, as a small function (the kernel), and tell the GPU how many times to
//! run it. The GPU launches that many copies at once, each told which element it's
//! responsible for.
//!
//! One running copy of the kernel is a "thread". Threads are grouped into "blocks", and all
//! the blocks from one launch form the "grid". A thread doesn't know how many copies exist.
//! It only knows its own index (block index * block size + thread index) and does its own
//! tiny job.
//!
//! (Graphics APIs such as WebGPU call the same thing a "compute shader", and a thread an
//! "invocation". Same idea, different vocabulary.)
//!
//! ## How the CPU talks to the GPU
//!
//! The GPU is a separate processor across the PCIe bus, and reaching it has a fixed
//! overhead. So the CPU doesn't wait for each piece of work to finish before sending the
//! next:
//!
//!     1. Launch: the CPU puts a kernel launch (or a copy) onto a stream. This returns
//!        almost immediately; the kernel has not run yet.
//!     2. Execute: the GPU runs the stream's work in order, on its own schedule, while
//!        the CPU carries on.
//!     3. Sync: when the CPU needs a result, it must wait for the stream to catch up.
//!        Copying results back to the host does this wait for you.
//!
//! Consequences to keep in mind while reading the code below:
//!     - Timing a launch on the CPU measures almost nothing, because the launch only
//!       queues the work. Stage 3 uses CUDA events, which are timestamps recorded on the
//!       GPU itself.
//!     - Work on one stream runs in order, so the copy back can't start before the
//!       kernel has finished.
//!     - A kernel's errors can show up later, at the next sync, not at the launch line.
//!
//! TODO: name the cuda-oxide calls for each of these once we've read the vecadd example.
//!
//! ## What are we achieving in this file?
//!
//! The smallest GPU program: take an array of f32 and double every element on the GPU.
//! The arithmetic is trivial on purpose. The point is the plumbing, which every later
//! stage reuses. It follows PMPP chapter 2's three parts:
//!
//!     Part 1: context + stream, allocate GPU memory, copy the input over
//!     Part 2: write the kernel, pick a launch configuration (blocks x threads), launch
//!     Part 3: copy the result back (which waits for the kernel), check it
//!
//! TODO: fill in the cuda-oxide names for each part after reading vecadd.
//!
//! Compared with the wgpu version (archived at tag `wgpu-archive`), CUDA hides three of
//! its seven steps: there are no bind groups (you pass buffers to the kernel like normal
//! arguments), no separate pipeline object, and no staging buffer you manage yourself
//! (copying back handles it).
//!
//! The round trip looks like this:
//!
//!   CPU array --copy--> GPU buffer --kernel--> GPU buffer --copy--> CPU array
//!                                  (parallel)
//!
//! NOTE: these two copies are where GPU programming stops being free.
//! Doubling six numbers on the GPU is far slower than doing it on the CPU, because the
//! copies and the setup cost more than the arithmetic. The GPU only wins once the work in
//! the middle is big enough to pay for the transfers, which is why the later stages care so
//! much about keeping data on the GPU between kernels instead of round-tripping it every
//! time. Stage 5 is the clearest example: the particle positions stay on the GPU for
//! thousands of steps, and only occasional snapshots come back to be saved as PNG frames.

// Imports -----
//
// cuda_core is the HOST side: everything the CPU does to drive the GPU (open a session, 
// make boxes, launch, copy back). More imports join this list as we need them.
use cuda_core::{CudaContext, DeviceBuffer};

// cuda_device is the GPU side: what kernel code can use (thread indices, 
// DisjointSlice) plus the two macros that mark kernels and bundle them.
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// LaunchConfig: the launch numbers (how many blocks, threads per block).
use cuda_core::simt::LaunchConfig;

// How many numbers we add. Deliberately not a multiple of 256 (the block size we 
// will launch with), so the last block has leftover threads with nothing to do,
// and the bounds check in the kernel actually matters.
const N: usize = 1000;

// The Kernel: The recipe (code) each GPU thread runs, compiled at BUILD time
//
// #[cuda_module] marks a module holding kernels. At build time it:
// - compiles every #[kernel] inside the GPU code and pack it into our program 
//      (the "one program, recipe packed inside" from our diagram)
// - generates kernels::load(&ctx), which hands that packed code to the GPU driver
//     at runtime (used at the end of Part 1 below);
// - generates a typed launch method per kernel (module.vecadd(...)),
//      checked by the compiler like any Rust call (used in Part 2)

// We can have several kernels in one mod kernels
// The macro's documentation says its generates one launch method per `#[kernel]`
// A single kernels::load(&ctx) loads them all at once, and you launch whichever you
#[cuda_module]
mod kernels {
    use super::*; // brings the imports above into this module

    // #[kernel] = CUDA C++'s __global__: a function the CPU launches and
    // thousands of GPU threads run at once. Every thread runs this SAME
    // code; the only difference between them is their seat number.
    //
    // How many threads? The kernel never says. The CPU picks that at
    // launch (Part 2): LaunchConfig::for_num_elems(1000) gives blocks of
    // 256 threads (fixed) and 1000 / 256 = 3.9, rounded up to 4 blocks.
    // So 4 x 256 = 1024 threads, 24 more than we have elements:
    //
    //      block 0          block 1          block 2          block 3
    //   [0 ... 255]      [0 ... 255]      [0 ... 255]      [0 ... 255]
    //   seats 0-255      seats 256-511    seats 512-767    seats 768-1023
    //                                                      (1000-1023 idle)
    //
    // Parameters (at launch, the CPU passes boxes of GPU memory for them):
    //   a, b: &[f32]   read-only inputs. Slices carry their length, so no
    //                  separate "n" parameter like in PMPP.
    //   c: DisjointSlice<f32>   the output. Rust normally allows only ONE
    //                  writer to a slice at a time (&mut [f32]), but here
    //                  1024 threads write at once. DisjointSlice allows it
    //                  with one rule: each thread may only write the ONE
    //                  element at its own seat number. No two threads touch
    //                  the same element ("disjoint" = no overlap), so they
    //                  can't interfere. Inside it's just like a slice
    //                  (start + length); only the way you access it is
    //                  restricted, via get_mut(idx) below. `mut` because
    //                  get_mut needs to change it.
    //
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        // This thread's seat number. The GPU gives each thread its block
        // number and its position in the block; index_1d combines them:
        //     seat = block number * 256 + position in block
        //     e.g. block 3, position 231 -> 3 * 256 + 231 = 999
        //
        // "1d" because our data is one flat row, so one number per thread
        // is enough. Images and matrices (Stages 1, 6) use index_2d (row +
        // column). The LAUNCH must match: index_1d checks the launch is
        // 1D, and for_num_elems always makes one. If it weren't, the index
        // would be marked invalid and get_mut would return None.
        //
        // It comes back wrapped in a ThreadIndex type (for seat 999 it's
        // essentially "999, valid"). Why not a plain number? get_mut only
        // accepts a ThreadIndex, and index_1d is the only way to get one.
        // If get_mut took a plain usize, we could write c.get_mut(5) and
        // all 1024 threads would write c[5] at once: the exact collision
        // DisjointSlice exists to prevent.

        // We are essentially just taking the blocks with threads and flattening it into a 1d
        // object so that we don't need to refer to the indices by (block number, thread index in block), 
        // such as (0, 234) - Block 0, thead number 234,
        // and we can just say "element i" instead.
        // The flattening hides that grouping from the code that handles the data.
        let idx = thread::index_1d();

        // Indexing = how threads map onto data: the rule that turns a 
        // thread's built-n numbers (block, position in block) into WHICH
        // piece of data it works on. Here the rule is simple: seat number = element number.
        // Later stages use other mappings:
        //  - images (Stages 1): row + column, one thread per pixel
        //  - more data than threads: each thread handles every Nth element (a "grid-stride loop");
        // - tiles (Stage 6): a black shares on tile of a matrix.
        // The mapping also affects SPEED: neighboring threads reading neighbouring memory is much faster
        // then scattered reads.

        // We do this because we need to map the GPU threads to elements in our data
        // The GPU only tells each thread where it sits in the launch (block and position)
        // Indexing translates that into which element of the data is the thread's job
        // The question every kernel answers first stays the same: which data is mine?
        
        // .get() unwraps it to a plain usize, for ordinary indexing of a,b.
        // Converts it from a ThreadIndex to as usize, the ordingary number type Rust uses for indexing
        let i = idx.get();
        
        // get_mut IS the bounds check (PMPP's `if (i < n`): Some for seats 0..999,
        // None for the 24 idle seats 1000...1023, which then do nothing.
        // Impossible to forget.
        // We are using if let Some() because get_mut returns an Option
        // if let handles both cases in one line:
        // if its some, we run the thread's element
        // if its None, it skips the block, thread does nothing
        // Would be the same as: `match c.get_mut(idx) { Some(c_elem) = > do something, None => {}}`
        // get_mut is called on c (the output) with the thread index - it hands back that threads element or None if it is past the end
        if let Some(c_elem) = c.get_mut(idx) {
            // Only seats 0..999 get here: get_mut returned Some, which it
            // only does when i < c's length (1000). a and b also hold 1000
            // elements, so a[i] and b[i] are in range too. (They're normal
            // bounds-checked Rust reads: if a were shorter, the read would panic;
            // on the GPU that stops the kernel and shows up as an error at the next wait.)
            *c_elem = a[i] + b[i];
        }
    }
}

fn main() {
    println!("=== Stage 0 - Vector Addition === \n");

    // Connect: Open a session with the GPU + get its queue -----
    //  
    // Open a session ("context") with GPU number 0, the first (and here only)
    // GPU in the machine. In CUDA's vocabulary, the physical GPU is the "device"; the context
    // is our session with it. Every body and loaded kernel we create from now on belongs to this session
    // and its cleaned up when it ends.
    //
    // This can fail (No NVIDIA GPU, driver problem), so it returns a Result.
    // For a learning project, we just stop with a message.
    // Starts the driver, picks GPU number 0, opens the session, and returns it in an Arc.
    let ctx = CudaContext::new(0).expect("Failed to open a session with GPU 0");
  
    // Get the session's ready-made queue ("stream"). Every GPU job we hand over
    // (fill a box, run the kernel, copy back) goes in the queue, and the GPU does them
    // one at a time, strictly in the order given.
    // The queue only holds the job, never the data
    //
    // Arc detail: an Arc is a pointer to ONE shared object plus a counter.
    // Cloning the Arc makes another pointer (count + 1), never a copy of the
    // object. default_stream takes `self: &Arc<Self>` (our ctx pointer, not the base context)
    // so that `self.clone()` can make a second pointer, which the stream stores in its field.
    // Now, the session stays alive for as long as the stream exists; the stream can never 
    // outlive the session.
    // counts now: session = 2 (ctx + the stream's), stream = 1.
    let stream = ctx.default_stream();
    // The CPU and GPU don't have to keep the pace: the CPU adds jobs quickly
    // and moves on, and the jobs wait in the queue until the GPU gets to them.
    // That's why the launch returns at once while the kernel runs later.

    // Arc = atomic reference count
    // Atomic: an operation that happens all at once, as one indivisible step. Other 
    // threads see it either not started or fully finished, never half-done.
    // Even "add 1 to the count" is secretly 3 steps: read the value, add 1, write it back
    // If 2 threads do this at once without atomics, there could be a data race
    // An atomic does all 3 steps at once and another thread has to wait its turn

    // `ctx` and `stream` are wrapped in Arc because everything we create later (boxes of GPU memory, the loaded 
    // kernel, the queue itself) only make sense while the session is still open. So each of those quietly holds its own 
    // Arc of what it depends on and the session stays alive until the LAST thing using it is gone, 
    // in whatever order our variables are dropped. That makes "box outlives its session" impossible without
    // lifetime annotations everywhere. This is true since `DeviceBuffer` contains its own `Arc<CudaContext>` as well.
    // Arc rather than Rc because a session can be used from several CPU
    // threads, and Arc keeps its counter correct when that happens.

    // Part 1: Set Up -----
    //
    // The input data, in ordinary CPU memory (plain Vecs, No GPU yet).
    // a[i] = i and b[i] = 2i, so every answer should be c[i] = 3i,
    // easy to check by eye.
    // Note, you don't need .iter() here since `Range` itself is an iterator,
    // not a collection like a Vec
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();
    
    println!("Inputs first (5):");
    println!("  a = {:?}", &a_host[0..5]);
    println!("  b = {:?}", &b_host[0..5]);

    // Boxes of GPU Memory ("device buffers") -----
    //
    // A DeviceBuffer is a fixed-size box of GPU memory holding one type (here f32).
    // Dropping it frees the GPU memory, just like a Vec.
    // Each box also holds its own Arc of the session (ctx), so a box can never outlive its session
    // (session count: 2 -> 5 after these 3).
    //
    // Naming: `_host` = CPU memory, `_dev` = GPU memory. They're separate:
    // the GPU can't read a_host, the CPU can't read a_dev.

    // from_host: reserve a box the size of the vec, put "copy the Vec into it"
    // in the queue, then WAIT until that copy is done. It has to wait:
    // it only borrows a_host for this call, and once the call returns we'd be free to change
    // or drop a_host. a_host stays on the CPU unchanged.
    // Here, we are allocating memory on the GPU and filling it with data from a_host and b_host
    // What from_host does:
    // 1. Reserves a box in GPU memory for the data
    // 2. Puts a job in the stream saying "copy a_host into that box"
    // 3. Waits until the GPU has done the job
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).expect("Failed to make box a on the GPU");
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).expect("Failed to make box b on the GPU");

    // zeroed: reserve a box for N numbers and queue "fill with zeros".
    // Unlike from_host it does NOT wait: there's no CPU data to protect.
    // No data to copy in, so Rust can't guess the type, ::<f32> tells it.
    // Why zero? Safe Rust never hands out memory with leftover garbage,
    // and if the kernel ever skipped an element, we would see 0.0, not junk.
    // `mut` because the launch hands it to the kernel to write into.
    // Here, we are allocating memory which will eventually store the result.
    // Rather than it being empty, we are just allocating zeros.
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).expect("Failed to make box c on the GPU");

    // Can't look inside the boxes from the CPU; that needs the copy back (Part 3).
    // Getting here without an expect firing means they exist
    println!(
        "\nMade 3 boxes on the GPU, {} bytes each", // f32 = 4 bytes
        N * std::mem::size_of::<f32>()
    );

    // Load the kernel -----
    //
    // kernels::load (generated by #[cuda_module]) finds the GPU code packed 
    // inside our program and hands it to the driver, which gets it ready to run on THIS card.
    // Like the boxes, the loaded module belongs to the session.
    // Happens once; the module is reused for every launch.
    let module = kernels::load(&ctx).expect("Failed to load the kernel");
    // The kernel itself never mentions ctx: its just a recipe, compiled before any session exists.
    // THIS line above ties it to our session: it loads the GPU code into CTX, and the returned module
    // holds its own Arc of ctx (like the stream and the boxes), so it belongs to, and keeps alive,
    // this session.

    println!("Kernel loaded.");

    // Part 2: Run - launch the kernel (can repeat)

    // The launch numbers -----
    //
    // This where "4 blocks of 256 threads" is decided: by the CPU, at launch.
    // The kernel never knows in advance; it reads these numbers back from the GPU while running (index_1d).
    //
    // 256 threads per block: a common default. It's a multiple of 32 (one warp), so no warp is left half-empty.
    // Tuning this is a later topic (stages 3 and 7).
    const THREADS_PER_BLOCK: u32 = 256;

    // Enough blocks to cover N, rounded UP: 1000 / 256 = 3.9 -> 4 blocks.
    // Rounding down (3 blocks = 768 threads) would leave 232 elements unprocessed;
    // rounding up gives 1024 threads, 24 idle.
    let blocks = (N as u32).div_ceil(THREADS_PER_BLOCK);

    // Eeach dimension is (x, y, z). Ours are 1D: y and z are 1. This MUST match
    // the kernel's index_1d, which checks for a 1D launch.
    // shared_mem_bytes: per-block scratch memory (why blocks exist); we don't use any yet.
    // Shortcut: LaunchConfig::for_num_elems(N as U32) gives the same result
    // (256 threads per block, blocks rounded up, no shared memory). We write it out
    // by hand here so the numbers are visible
    let config = LaunchConfig {
        grid_dim: (blocks, 1, 1), // how many blocks
        block_dim: (THREADS_PER_BLOCK, 1, 1), // how many threads per block
        shared_mem_bytes: 0,
    };

    println!(
        "\nLaunching {} blocks x {} threads = {} threads for {} elements.",
        blocks,
        THREADS_PER_BLOCK,
        blocks * THREADS_PER_BLOCK,
        N,
    );

    // The raw launch ---
    //
    // module.vecadd was generated by #[cuda_module]. The compiler checks the
    // ARGUMENTS against the kernel's parameters:
    //  a: &[f32]   <- &DeviceBuffer<f32> (&a_dev)
    //  b: &[f32]   <- &DeviceBuffer<f32> (&b_dev)
    //  c: DisjointSlice<f32>   <- &mut DeviceBuffer<f32> (&mut c_dev)
    // Pass the wrong type, or forget one, and it won't compile.
    //
    // What the compiler can NOT check is the launch SHAPE: a LaunchConfig is just numbers
    // and doesn't know which kernel it's for. So this "raw" launch is `unsafe`: we promise the numbers
    // fit the kernel.
    // "Fit" means the launch has the shape the kernel was written for:
    // ours numbers its thread as one flat row (index_1d), so the launch
    // must be a flat row of blocks too (y = z = 1). With say, a 4 x 2 grid,
    // two blocks would compute the same seat numbers and two thread would write the same element of c.
    //
    // The prepared launch, our next version, makes the library check it.
    //
    // It only QUEUES the kernel and returns at once. Ok means "queued",
    // not "finished". If the kernel fails while running, the error shows up at the next wait.
    //
    // SAFETY:
    // - The launch is 1D (y = z = 1), matching the kernel's index_1d.
    // - a_dev, b_dev, and c_dev all hold N elements, so every seat that get_mut
    //   lets through (i < c's length) is also in range for a, b.
    // - The kernel uses no shared memory, and we request none.
    unsafe { module.vecadd(&stream, config, &a_dev, &b_dev, &mut c_dev) }.expect("Failed to queue the kernel");

    println!("Kernel queued (it may still be running).");

    // Part 3: Get the Result Back - waits for the queue to finish -----

    // to_host_vec does: copy c back into a Vec on the CPU.
    //
    // When we reach this line, the kernel may still be running (the launch only queued it).
    // to_host_vec puts "copy c back" in the queue BEHIND the kernel, then the CPU WAITS here until:
    //  1. The kernel has finished, and then
    //  2. box c has been copied back into the Vec.
    // Only then does it hand us the Vec, so c_host always holds the finished result.
    //
    // This is also where the kernel errors show up: if the kernel had crashed, the wait
    // here would return an Err and this expect would stop us.
    //
    // It only reads c_dev, so the box still exists afterwards.
    let c_host = c_dev.to_host_vec(&stream).expect("Failed to copy c back (or the kernel failed)");

    // c[i] should be a[i] + b[i] = i + 2i = 3i
    println!("\nResult (first 5):");
    println!("  c = {:?}", &c_host[0..5]);
    // The last element is the one the bounds check had to get right:
    // seat 999 must write, seat 1000 must not.
    println!("  c[{}] = {}", N - 1, c_host[N-1]); // 3 * 999 = 2997

    // Check every element -----
    //
    // Compare the GPU's answer with the same sum done on the CPU.
    // Checking a GPU result against a simple CPU version (a "reference") is the standard weay to test
    // kernels; later stages do the same.
    //
    // Why a tolerance (1e-5) instead of ==? Floating point maths can round slightly differently on 
    // the CPU and the GPU. Our numbers are small whole numbers that store f32 directly, so here ==
    // would work too, but the tolerance is the habit that stays correct in later stages.
    let mut errors = 0;
    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        if c_host[i] - expected.abs() > 1e-5 {
            if errors < 5 {
                // Only print the first few, so a broken kernel doesn't flood the terminal
                eprintln!(" Error at [{}]: expected {}, got {}", i, expected, c_host[i])
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("\nSUCCESS: all {} elements correct!", N);
    } else {
        println!("\nFAILED: {} of {} elements wrong", errors, N);
        // A non-zero exit code tells scripts (and CI) the run failed.
        std::process::exit(1);
    }
}