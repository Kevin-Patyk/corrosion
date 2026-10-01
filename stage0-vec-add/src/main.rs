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

// How many numbers we add. Deliberately not a multiple of 256 (the block size we 
// will launch with), so the last block has leftover threads with nothing to do,
// and the bounds check in the kernel actually matters.
const N: usize = 1000;


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
    let ctx = CudaContext::new(0).expect("Failed to open a session with GPU 0");
  
    // Get the session's ready-made queue ("stream"). Every GPU job we hand over
    // (fill a box, run the kernel, copy back) goes in the queue, and the GPU does them
    // one at a time, strictly in the order given.
    //
    // Arc detail: an Arc is a pointer to ONE shared object plus a counter.
    // Cloning the Arc makes another pointer (count + 1), never a copy of the
    // object. default_stream takes `self: &Arc<Self>` (our ctx pointer, not the base context)
    // so that `self.clone()` can make a second pointer, which the stream stores in its field.
    // Now, the session stays alive for as long as the stream exists; the stream can never 
    // outlive the session.
    // counts now: session = 2 (ctx + the stream's), stream = 1.
    let stream = ctx.default_stream();

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
    // Arc rather than Rc becayse a session can be used from several CPU
    // thrreads, and Arc keeps its counter correct when that happens.

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
}