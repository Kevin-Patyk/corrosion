//! filler-ch4: device query + ch. 4 arithmetic for my own card
//! 
//! What are we achieving in this file? (Part 1)
//! Ask the GPU, through the driver, what it is: how many SMs it has,
//! how big the four pools are that an SM hands out to blocks (thread seats, 
//! block seats, registers, shared memory). PMPP 4.8 does this with cudaGetDeviceProperties;
//! here it is cuda-core methods on CudaContext.
//! 
//! Refresher: the GPU hierarchy
//! 
//! Four levels, each one CONTAINS and SCHEDULES the one below it:
//! 
//! GPU runs the GRID.
//!     The grid is all the blocks of one launch. The GPU hands blocks to SMs
//!     as room frees up; a block goes to one SM, whole, and never spans SMs.
//!     Blocks never wait for each other.
//! 
//! SM runs BLOCKS.
//!     One SM = control + cores + on-chip memory (registers, shared memory).
//!     It holds several blocks at once, as many as fit in its four pools: thread
//!     seats, block seats, registers, shared memory. The first pool to run out stops
//!     the next block landing. The pools' sizes are what this file prints. 
//! 
//! PROCESSING BLOCK runs WARPS.
//!     Once a block lands, the hardware cuts it into warps of 32 threads 
//!     (consecutive threadIdx). Each warp is assigned to one of the SM's N processing blocks:
//!     a dispatcher plus its cores. The dispatcher fetches one instruction at once;
//!     all 32 threads execute it on their own data (SIMD). N processing blocks = 
//!     N warps issuing at the same instant; the rest are resident, waiting for their
//!     turn (latency hiding).
//! 
//! CORES run THREADS.
//!     Inside the processing block, one thread per core per round.
//!     Fewer cores than 32 -> the warp runs in rounds (8 cores = 4 rounds),
//!     still as once instruction.
//! 
//! Where this file sits: it talks to the top 2 levels only. The GPU (SM count)
//! and the SM (its four pools) are what the driver can tell us; warps and cores show up in
//! Part 2, when we do the arithmetic. 

use cuda_core::CudaContext;
use cuda_core::{sys, IntoResult};

/// Ask the driver one numbered attribute of the GPU.
/// `fn name(args) -> type` defines a function; `CudaContext` borrows the
/// context (we only read it). `sys` = cuda_core's re-export of the raw
/// C bindings; CUdevice_attribute is the type of the attribute numbers. 
fn attribute(ctx: &CudaContext, attr: sys::CUdevice_attribute) -> Result<i32, cuda_core::DriverError> {
    // C files the answer in through a pointer, so make a slot first.
    let mut value: i32 = 0;
    // unsafe = "I promise &mut value is a valid place to write an i32".
    // cuDeviceGetAttribute (where to write, which attribute, which device)
    // returns a C error code; .result() (trait IntoResult) turns it into 
    // Ok(()) or Err(DriverError) and ? passes the error up.
    unsafe {
        sys::cuDeviceGetAttribute(&mut value, attr, ctx.cu_device()).result()?;
    }

    Ok(value)
}

/// The card's limits, read once, passed around as one value.
/// `struct` = a named bundle of fields; each field has a type.
struct Card {
    sm_count: u32,
    warp_size: u32,
    threads_per_sm: u32,
    blocks_per_sm: u32,
    regs_per_sm: u32,
    smem_per_sm: u32,
}

/// Print the ch. 4 launch arithmetic for `n` elements in blocks of `block`.
///
/// What is being calculated here?
/// Everything the host fixes at launch time, before the GPU sees anything:
/// how many blocks and warps the launch creates, how many threads have no data
/// (because threads come in whole blocks but data comes in any size).
/// and how the data edge cuts through the warps. None of this depends on the
/// card except the warp size.
/// 
/// `&Card` borrows the card (read only). No return value: it prints.
/// The block size is our choice at launch (LaunchConfig2D::new()) in Stage 2, for example.
fn launch_report(card: &Card, n: u32, block: u32) {
    // 1. Launch shape
    // Threads come in whole blocks, so the grid is rounded UP:
    // blocks = div_ceil(n, block).
    // threads_launched = blocks x block >= n; the gap is the extras, the
    // threads that `if i < n` will switch off. 
    let blocks = n.div_ceil(block);
    let threads = blocks * block;
    let extras = threads - n;

    // 2. Warps
    // Once on an SM, each block is cut into warps of 32 threads.
    // A block size not multiple of 32 gets a last warp padded with
    // inactive threads, hence div_ceil again. Total warps = blocks x warps per block;
    // this is the count the SM's dispatchers work through.
    // Those padded threads take seats in every pool (thread seats, registers) and cores in every pass,
    // in every block of the launch and do nothing.
    // Padding happens in every block and comes from the block-size choice; extras from div_ceil happen
    // in the last block only and come from the data size. The first is always avoidable, the second usually isn't.

    // Why a block size is not a multiple of 32 is a mistake (e.g. 48):
    // The hardware cuts the block into warps of 32: warp 0 = 32 real threads,
    // warp 1 = 16 real + 16 padding. Nothing breaks; results are correct.
    // The cost is waste, charged twice, in EVERY block of the launch:
    // 1. Cores: each instruction of warp 1 goes to 32 cores, 16 do nothing.
    // 2. Pools: the SM charges the block 64 thread seats (and registers), not 48.
    //      The pools fill with padded threads, so fewer USEFUL threads are resident per SM
    //      (my card: 24 blocks x 48 = 1152 of 1526 seats useful),
    //      and with bigger sizes (480 vs 512), fewer blocks fit as well.
    // Fix: A multiple of 32 (128, 256, 512). Same cost, no padding.

    // As a note, processing blocks per SM x SMs = warps issuing at once. 
    // The warps resident, taking turns on those dispatchers, is the bigger number from the pools.
    let warps_per_block = block.div_ceil(card.warp_size);
    let total_warps = blocks * warps_per_block;

    // 3. Data edge
    // Only the LAST block can contain the edge. `n % block` (% = remainder)
    // = how many of its threads have data. 0 means n is a multiple of block:
    // the edge sits on a block border, every warp is all-yes, no divergence. 
    let used_in_last = n % block;
    
    // Straddling warp: the one warp with both yes and no threads; it costs an extra
    // pass with idle cores. It exists only when the used count is not a whole number of warps.
    // `if` expression here: it yields 1 or 0. 
    let straddling = if used_in_last % card.warp_size != 0 { 1 } else { 0 };

    // Fully idle warps: whole warps past the edge inside the last block.
    // Every thread fails the check -> one path, NOT divergence, just cores
    // doing nothing. (block - used) = extras in the last block; divide down to whole
    // warps.
    let idle = if used_in_last == 0 { 0 } else { (block - used_in_last) / card.warp_size };

    println!("n = {n}, block = {block}");
    println!("  blocks launched: {blocks}  threads: {threads}  extras: {extras}");
    println!("  warps per block: {warps_per_block}  total warps: {total_warps}");
    println!("  edge: {straddling} straddling warp(s), {idle} fully idle warp(s)");
}

/// Print the ch. 4 card arithmetic: how many blocks of `block` fit on one SM,
/// which pool limits it, waves, and occupancy.
/// 
/// What is being calculated here?
/// A block lands on an SM only if there is room in ALL four pools at once. Each pool,
/// on its own, allows some number of blocks; the smallest of those is what actually fits. 
/// From that: blocks per wave, waves, occupancy. `regs_per_thread` and `smem_per_block` are what
/// the compiler / kernel would decide; here we choose them to see their effect.
fn card_report(card: &Card, n: u32, block: u32, regs_per_thread: u32, smem_per_block: u32) {
    // 1. one quotient per pool: how many blocks this pool alone would allow.
    //  "how many blocks would fit on the SM if THIS pool were the ONLY limit?"
    let by_threads = card.threads_per_sm / block;
    // Block seats alone: 24, whatever the block size.
    let by_blocks = card.blocks_per_sm; 
    // Registers alone: each block takes (block x regs per thread) registers.
    // "how many blocks would fit on the sm if THIS pool were the ONLY limit?"
    let by_regs = card.regs_per_sm / (block * regs_per_thread);
    // Shared memory alone. A kernel with no shared memory has no limit from this pool.: u32::MAX
    // (the biggest u32) so it nevers wins the min. Diving by 0 would panic.
    // "how many blocks would fit on the sm if THIS pool were the ONLY limit?"
    let by_smem = if smem_per_block == 0 { u32::MAX } else { card.smem_per_sm / smem_per_block };

    // 2. The verdict: the smallest opinion is what actually fits, because a block needs room
    // in all four pools at once. `.min(x)` = smaller of the two.
    // The winning pool has zero leftover; the other three keep leftover nobody can use.
    let fits = by_threads.min(by_blocks).min(by_regs).min(by_smem);

    // Name the winner for the printout. `match` compares `fits` against each 
    // arm in order and picks the first that is equal; `_` = anything else.
    let limiter = match fits {
        f if f == by_threads => "thread seats",
        f if f == by_blocks => "block seats",
        f if f == by_regs => "registers",
        _ => "shared memory",
    };

    // 3. Waves: blocks per wave = fits x SMs; waves = div_ceil (a partly
    // full last wave still costs a whole wave).
    let blocks = n.div_ceil(block); // We determine the number of block by ceil dividing n by the size of the blocks we chose
    let per_wave = fits * card.sm_count; // We determine the number of blocks per wave by multiplying how many blocks fit by SM count
    let waves = blocks.div_ceil(per_wave); // We determine the number of waves by ceil diving the number of blocks we chose by how many blocks can run per wave
    let last_wave_fill = 100.0 * (blocks - (waves - 1) * per_wave) as f32 / per_wave as f32;

    // 4. Occupancy: resident threads / thread seats. Counting warps gives the same
    // ratio since the 32 cancels. `as f32` turns integers into floats so the division
    // keeps its fraction; {:.0} prints 0 decimals.
    let occupancy = 100.0 * (fits * block) as f32 / card.threads_per_sm as f32;
    // We are getting the number of threads that are fitting in the SM at once by multiplying how many blocks fit by the block size
    // Then we divide that result by how many threads the SM can actually hold

    println!("block = {block}, {regs_per_thread} regs/thread, {smem_per_block} B smem/block");
    println!("  per SM: threads allow {by_threads}, block seats {by_blocks}, registers {by_regs}, smem {}",
        if smem_per_block == 0 { "no limit".to_string() } else { by_smem.to_string() });
    println!("  -> {fits} blocks per SM, limited by {limiter}; occupancy {occupancy:.0}%");
    println!("  -> {per_wave} blocks per wave; {blocks} blocks = {waves} waves (last {last_wave_fill:.0}% full)")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Open the GPU. Same call as Stages 0-2: device 0, primary context.
    // No stream needed yet: a query is question to the driver, not work for the queue.
    let ctx = CudaContext::new(0)?;

    // 2. Card-wide facts.
    // compute_capability() returns a tuple (major, minor); let (a, b) = ...
    // unpacks it.
    let (major, minor) = ctx.compute_capability()?;
    println!("device: {} (compute capability {major}.{minor})", ctx.device_name()?);
    // SM count: how many SMs the GPU hands blocks to. This is the
    // "more SMs = faster, never different" number from 4.2.
    let sm_count = ctx.multiprocessor_count()?;
    println!("SMs: {sm_count}");

    // 3. Per-SM pool: thread seats.
    // The most threads one SM can hold RESIDENT at once, all blocks together.
    // The seats are shared by every block sitting on that SM, not a per-block allowance.
    // Occupancy's denominator, in threads (divide by 32 for warps).
    let threads_per_sm = ctx.max_threads_per_multiprocessor()?;
    println!("thread seats per SM: {threads_per_sm} (= {} warps)", threads_per_sm / 32);

    // 4. Per-BLOCK limits (not pools): what one block may look like.
    // launch_limits() returns a struct DeviceLaunchLimits; its fields are
    // private, so values come out through getter methods: limits.max_thread_per_block().
    // These are the limits prepare_<kernel> checks against the launch contract.
    let limits = ctx.launch_limits()?;
    println!("max threads per block: {}", limits.max_threads_per_block());
    println!("max block dims (x, y, z): {:?}", limits.max_block_dim());
    println!("max shared memory per block: {} bytes", limits.max_shared_memory_per_block());

    // Results:
    // device: NVIDIA GeForce RTX 5060 Ti (compute capability 12.0)
    // SMs: 36
    // thread seats per SM: 1536 (= 48 warps)
    // max threads per block: 1024
    // max block dims (x, y, z): (1024, 1024, 64)
    // max shared memory per block: 49152 bytes

    // The results show that my card can have 36 x 1,536 = 55,296 threads (1,728 warps)
    // resident at once (sitting on the SMs, registers in place), not how many running at once. 
    // To calculate the number of threads running at instant = one warp per processing block. 
    // We first need to find how many processing blocks we have per SM to see how many threads
    // we can run at once.
    
    // Example wave calculation: if we have 256-thread blocks, we could have 1536 / 256 = 6 blocks 
    // per SM. With 36 SMs, that is 6 x 36 = 216 blocks per wave. If we needed 2,960 blocks of 256,
    // then we would have 13.7 -> 14 waves, the last one 70% full.

    // 3b. The other three pools + warp size, via the raw driver.
    // &ctx: the ctx is an Arc<CudaContext> (shared handle); &ctx hands the
    // function a plain &CudaContext (Rust unwraps the Arc for us)
    let warp_size = attribute(&ctx, sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE)?;
    let blocks_per_sm = attribute(&ctx, sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)?;
    let regs_per_sm = attribute(&ctx, sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)?;
    let smem_per_sm = attribute(&ctx, sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?;
    println!("warp size: {warp_size}");
    println!("block seats per SM: {blocks_per_sm}");
    println!("registers per SM: {regs_per_sm}  (32-bit each)");
    println!("shared memory per SM: {smem_per_sm} bytes");

    // Results:
    // warp size: 32
    // block seats per SM: 24
    // registers per SM: 65536  (32-bit each)
    // shared memory per SM: 102400 bytes

    // Thread seats / block seats = 1536 / 24 = 64: a block smaller than 64 threads 
    // uses up the block seats before the thread seats. Each landing block takes 1 block seat
    // and (block size) thread seats; with 32-thread blocks, the 24th block takes the last block seat while
    // only 768 of 1537 thread seats are used -> 50% occupancy. At exactly 64, both pools run out on the same block.

    // 5. Bundle the pools Part 2.
    // Card { field: value, ... } builds a struct; when a local has the same
    // name as the field, `field: field` can be written just `field`.
    let card = Card {
        sm_count,
        warp_size: warp_size as u32,
        threads_per_sm,
        blocks_per_sm: blocks_per_sm as u32,
        regs_per_sm: regs_per_sm as u32,
        smem_per_sm: smem_per_sm as u32,
    };

    // 6. Launch arithmetic.
    // Checks:
    // 1) 1,003 / 64 -> 16 blocks, 32 warps, 1 straddling, 0 idle
    // 2) 756,480/256 -> 2,955 blocks, 23,640 warps, 0 extras. (grayscale as a
    // flat 1D launch; the real 2D launch was 2,960 blocks because the grid covers whole
    // 16-row block rows: 5 blocks' worth of spare threads, cut across 80 warps).
    launch_report(&card, 1003, 64);
    launch_report(&card, 756_480, 256);

    // As a note, when we do a wave calculation, we look at how many blocks the SMs
    // can hold at once, all SMs together. It is how many can be resident at once,
    // not how many get handed out in one go. The GPU hands out blocks one at a time
    // as seats free up. 
    // And the number of blocks assigned to an SM depends on which of the 4 items from
    // the pool allows the fewest blocks.
    // All SMs across the GPU will assign the same amount of blocks because
    // every SM has the same four pools and every block of a kernel has the same size,
    // register count, and shared memory, so the fit is the same number on every SM. 

    // 7. Card arithmetic on the grayscale launch. Expected: 42 regs -> thread seats win,
    // 6 per SM, 216 per wave, 14 waves, 100%; 64 regs -> registers win, 4 per SM, 144 per wave,
    // 21 waves, 67%.
    card_report(&card, 756_480, 256, 42, 0);
    card_report(&card, 756_480, 256, 64, 0);

    // Example results (1):
    // block = 256, 42 regs/thread, 0 B smem/block
    // per SM: threads allow 6, block seats 24, registers 6, smem no limit
    // -> 6 blocks per SM, limited by thread seats; occupancy 100%
    // -> 216 blocks per wave; 2955 blocks = 14 waves (last 68% full)

    // For the one above, only 6 blocks fit. Each block is 256 threads, so 6 x 1536 are resident.
    // We have 100% occupancy since 100 * (1536 / 1536) = 100.
    // With 36 SMs on the card, we have 36 x 6 = 216 blocks per wave. 
    // We have 756_480 / 256 (n / block size) = 2955 blocks.
    // If we have 216 blocks per wave, that means we have 2955 / 216 = 13.68 = 14 waves.
    // In the above example, thread seats filled up first. At 42 registers per thread, the four opinions are:
    // thread seats 6, block seats 24, registers 65,536 / (256 x 42) = 6.09, shared memory no limit.

    // As a refresher, every block landing on an SM takes a slice of all 4 pools at once (thread
    // seats, block seats, registers, shared memory), so the number of blocks that fit is the smallest 
    // any one pool allows. That pool is the limiter; it has no leftover, while the other three
    // keep the leftover nobody can use.

    // 8. Block-size sweep at 32 regs/thread: 32 -> block seats win (50%);
    //    64 -> thread seats and block seats tie (100%); 256 and 768 divide 1536
    //    (100%); 1024 doesn't (1 fits, 67%). Then the register cliff from 4.7:
    //    512-thread blocks at 32 regs -> 3 fit (100%); at 33 -> 3.87 -> 3? check.
    for block in [32, 64, 256, 768, 1024] {
        card_report(&card, 756_480, block, 32, 0);
    }
    card_report(&card, 756_480, 512, 32, 0);
    // For the above example, with a block size of 512, N of 756_480, 32 regs per block, and 0 smem per block:
    // By threads = 1536 / 512 = 3 blocks fit
    // By blocks = 24 blocks fit
    // By regs = 65536 / (512 x 32) = 65536 / 16384 = 4 blocks fit
    // 0B smem per block.
    //
    // Only 3 blocks fit, so 3 x 512 = 1536 threads resident.
    // The occupancy is 100% since 100 * (1536/1536) = 100.
    // With 36 SMs on the card, we have 36 x 3 = 108 blocks per wave.
    // With an N of 756480 and block size of 512, we have 756480 / 512 = 1477.5 = 1478 blocks.
    // If we have 108 blocks per wave, that means we have 1478 / 108 = 13.69 = 14 waves.
    // We will have 1478 x 512 = 756736 totals threads, so we will have 756480 - 756736 = 256 extra threads.
    // Thread seats filled up first.

    card_report(&card, 756_480, 512, 33, 0);

    Ok(())
}
