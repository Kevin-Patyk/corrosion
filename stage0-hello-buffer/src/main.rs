//! Stage 0 - Hello Buffer
//! 
//! ## What is a GPU?
//! 
//! A CPU has a few powerful cores (8-32), each built to run one complicated sequence of instructions as fast as possible: big caches,
//! branch prediction, out-of-order-execution.
//! 
//! A GPU makes the opposite trade. It has thousands of small, simple cores that all run the SAME program
//! at the same time, each on a different piece of data. Any single GPU core is much slower than a CPU core,
//! but if the work is "do this one thing to a million elements", the GPU wins by sheer width. It also has its
//! own memory with far higher bandwidth than system RAM, because feeding thousands of cores is mostly a memory problem.
//! 
//! Two consequences of everything in this roadmap:
//!     1. The GPU is a separate device with separate memory. Data has to be copied over 
//!         to it and results copied back. Nothing is shared.
//!     2. The CPU does not call GPU functions. It records commands, submits them to
//!         a queue, and the GPU runs them asynchronously later.
//! 
//! Bandwidth: how many bytes per second can move between the memory and the cores, measured in GB/s.
//! System RAM manages rougly 50-100 GB/s; VRAM on a modern NVIDIA card manages 500-1000+. Doubling a million
//! floats is 8mb in and 8mb out and almost no arithmetic, so the time it takes is set by bandwidth, not the core count.
//! 
//! ## Why are we using it?
//! 
//! For data-parallel work: the same operation applied independently to many elements. 
//! The word "independently" is doing the work here. Computing element `i` must not need
//! the result of any other element, so the order doesn't matter and everything can happen at once.
//! Image filters, physics, sorting, matrix multiply, and neural networks all have this shape. 
//! The program that runs oncer per element is called a compute shader, or a kernel.
//! 
//! NOTE: A useful test for any loop: could the iterations run in random order and still 
//! give the same result? If yes, it's data parallel and maps straight onto a GPU. Stage 0
//! is deliberately the easiest case, so that the plumbing is only the hard part.
//! 
//! We use wgpu (the WebGPU API in Rust) with shaders written in WGSL. It runs on any GPU vendor,
//! and its concepts are explicit without being as verbose as raw Vulkan. CUDA comes in Stage 7, and the
//! ideas carry over.
//! 
//! ## Terminology
//! 
//! On a CPU you'd write a loop and put the per-element work in its body. 
//! On the GPU you write ONLY the body, as a small function, and tell the GPU how many times to run it.
//! The GPU launches that many copies at once, each told which element it's responsible for. That
//! function has two names, depending on who you ask:
//! 
//!     - "compute shader": the WebGPU/WGSL/graphics term. "Shader" is historical:
//!       these programs originally computed pixel shading and the name stuck even after
//!       they stopped having anything to do with pixels. This is what we write in the roadmap's wgpu stages.
//!     - "kernel": the CUDA/scientific-computing term for the same thing. This is what PMPP and the CUDA stages call it.
//! 
//! One copy of a function running one element is an "invocation" (WGSL) or a "thread" (CUDA). The shader
//! doesn't know how many copies exist. It only knows its own index and does its own tiny job.
//! 
//! ## How the CPU talks to the GPU
//! 
//! The CPU never calls a GPU function. The GPU is a separate processor across the PCIe bus, and reaching
//! it has a fixed overhead, so work is sent in batches instead of one call at a time:
//! 
//!     1. Record: write commands into a command encoder. Nothing runs just yet.
//!     2. Submit: finish() the encoder into a command buffer and hand it to the queue.
//!        This returns almost immediately.
//!     3. Execute: the GPU runs the queued commands in order, on its own schedule, while the CPU carries on.
//!     4. Sync: when the CPU needs a result, it must explicitly wait, which in wgpu means mapping a buffer.
//! 
//! Consequences to keep in mind while reading the code below: 
//!     - A line like dispatch_workgroups() does not run the shader. It records an instruction
//!       to run it later. Timing that line on the CPU measures nothing (Stage 3 uses GPU timestamp queries instead).
//!     - Readback takes two steps. The buffer the shader writes to lives in VRAM and the CPU can't read it, so
//!       we record a copy into a second mappable "staging" buffer, submit, then map that one and wait.
//!     - Errors can show up at submit time or later, and not at the line that caused them.
//! 
//! ## What are we achieving in this file?
//! 
//! The smallest GPU program: take an array of f32 and double every element on the GPU. 
//! The arithmetic is trivial on purpose. The poin is the plumbing, which every later stage reuses:
//! 
//!     1. Instance -> Adapter -> Device + Queue (get a handle on the GPU)
//!     2. Buffers (allocate GPU memory, upload)
//!     3. Shader module (the WGSL) kernel
//!     4. Bind group layout + group (wire buffers to the shader)
//!     5. Compute pipeline (shader + layout, compiled)
//!     6. Command encoder -> compute pass -> dispatch -> submit
//!     7. Readback (copy to a mappable buffer, map it, read on the CPU)
//! 
//! Everything after this stager is a variation on these seven steps.
//! 
//! The round trip looks like this:
//! 
//!   CPU array --copy--> GPU buffer --shader--> GPU buffer --copy--> CPU array
//!                                   (parallel)
//! 
//! NOTE: these two copies are where GPU programming stops being free.
//! Doubling six numbers on the GPU is far slower than doing it on the CPU,
//! because the copies and the setup cost more than the arithmetic. The GPU
//! only wins once the work in the middle is big enough to pay for the transfers,
//! which is why the later stages care so much about keeping data on the GPU between passes
//! instead of round-tripping it every time. Stage 5 is the clearest example: the particle
//! positions never leave the GPU, the compute pass updates them and the render pass draws them
//! from the same buffer.

// ---------------------------------------------------------------------------
// A note on async and .await
//
// The BOX ANALOGY
// Some functions take a while to answer, because they wait on something
// outside of our program: a web server, a databse, or (here) the GPU driver.
// A normal function would freeze our program until the answer arrives. An async function doesn't.
// Calling it returns immediately with a "future": think of it as a labelled empty box that will be filled later.
//
// A future box has two states:
//  - pending: the work is still going, the box is empty
//  - ready: the work is finished, the box holds the result (or an error)
//
// `.await` means "wait here until the box is filled, then hand me what's inside." The code
// after the `.await` does not run until then.
//
// "But isn't that blocking?" It depends what you look at:
//  - the TASK (this piece of code) is stuck. It cannot continue until the box is filled. That's true of every `.await`.
//  - the THREAD is not stuck. Think of a thread as a worker and a task as a to-do item. A blocking wait is the working
//    standing at the mail box staring at it. An `.await` is the working writing "waiting for letter" on the item, putting
//    it back on the pile, and picking up another item. When the letter arrives, the item goes back on the pile.
//
// "Non-blocking" describes the worker, not the to-do item. Async pays off when one thread has many tasks, because the
// thread stays busy while any single task waits.
//
//In this program the thread has exactly one task. When it parks that task there is nothing else to pick up, so
// the worker is standing at the mailbox after all. That's why pollster's function is honestly named `block_on:` here,
// `.await` really is "wait, then continue."
//
// WHY THESE CALLS ARE ASYNC
// `request_adapter` asks the driver "which GPUs do you have?", and `request_device` asks it "open a connection to this one."
// Both involve loading the driver, talking to the hardware, and possibly initializing it. That's the outside work we're
// waiting on. On a desktop it takes a few milliseconds; in a browser it can include a permission prompt that takes as long as
// the user takes. wgpu uses async for both so one API fits both cases.
//
// WHO FILLS THE BOX
// A future doesn't fill itself. Something has to keep checking on it and push the work until it's done.
// That something is called an executor, or an async runtime. tokio is the big one, built for servers that
// juggle thousands of boxes at once. We have one box at a time and no
// need for any of that, so we use pollster. Its `block_on` takes a single future
// and just sits on the current thread until the box is ready, then returns its contents. From our point of view it
// turns an async call back into an ordinary blocking function call, which is all we want.
//
// WHERE ASYNC SHOWS UP IN THIS FILE
// Only in setup. Once we hold a device and queue, creating buffers,
// recording commands and sumitting them are plain synchronous calls. The one other place
// we wait is reading the result back at the end and wgpu does that with a callback plus device.poll() instead 
// of a future. We'll cover that when we get there.
// ---------------------------------------------------------------------------

fn main() {
    println!("Hello, world!");
}
