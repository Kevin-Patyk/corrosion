//! Stage 2 - Image Filters (grayscale, then blur)
//! 
//! PMPP: Chapter 3 (multidimensional grids and data): the colour-to-grayscale and blue examples. Chapter 7 (convolution) is 
//! an optional background for the blur. This stage reuses everything from stage1-matmaul (mapping, bounds checks
//! linearizing) on real data: a picture loaded from a file.
//! 
//! ## What changes from Stage 1
//! 
//! The kernel pattern is the same: one thread PER output element, here one per output pixel.
//! Three things are new:
//!     1. The data is not square. A picture has a width (columns) and a height (rows), and they usually differ.
//!     2. A colour pixel is not one value, but three: R, G, B.
//!     3. The size is not a constant. It comes from a file, so we only know it when the program runs, not when it builds.
//! 
//! The CPU side gains two steps: load the picture before Part 1, and save the result as a PNG after Part 3 -
//! the `image` crate does both.
//! 
//! ## What is grayscale?
//! 
//! A colour needs three numbers (how much red, green blue). A shade of gray needs only one: its bightness,
//! from 0 (black) to 255 (white). Gratscale mixes each pixel's three values into one:
//! 
//!     gray = 0.21 * R + 0.72 * G + 0.07 * B
//! 
//! The weights aren't equal because our eyes are most sensitive to green. Each output pixel
//! needs only ITS OWN input pixel, so: one thread per pixel, reading 3 values and writing 1.
//! No pixel depends on another.
//! 
//! ## What is blur? (TODO: fill in when we get there)
//! 
//! Each output pixel is the average of its input pixel AND its neighbours (e.g. a 3x3 patch). Still
//! one thread per output pixel, but now each thread reads several input pixels, and pixels at the edge have
//! fewer neighbours. The result goes into a separate output, never back into the input.
//! 
//! ## How a colour picture sits in memory
//! 
//! Like a matrix: one long line, stored row-major. Pixel (row, col) is:
//!     pixel = row * width + col
//! 
//! But each pixel is 3 values side by side ("interleaved"): R G B R G B ...
//! So the spot of one value is
//! 
//! spot = (row * width + col) * 3 + channel (R = 0, G = 1, B = 2)
//! 
//! `width` is always the number of PIXELS across, not the values. The `* 3` comes after finding the pixel.
//! One row of the input is width * 3 values; the whole input is width * height * 3.
//! 
//! The gray output has 1 value per pixel, so it's just row * width + col. Input and out have the same
//! (row, col) for a pixel, but different formulas for where it lives in memory.
//! 
//! ## What each thread does
//!     1. Map: which outpit pixel is mine? -> (row, col)
//!     2. Check: col < width and row < height, otherwise stop
//!     3. Linearize: where is my output spot, and where are the inputs I need?
//!     4. Read + calculate
//!     5. Write the result to my output spot
//! 
//! Mapping is just arithmetic: every thread gets a (row, col), even if that pixel doesn't exist.
//! The check (step 2) stops those threads before they touch memory. Same order as matmul: map -> check -> linearize.
//! 
//! ## Launch shape for a non-square picture
//! 
//! Blocks stay 16 x 16 = 256 threads. The grid rounds up SEPARATELY for each direction:
//! 
//!     blocks_x = width.div_ceil(16) (across: columns)
//!     blocks_y = height.div_ceil(16) (down: rows)
//! 
//! In matmul both were WIDTH, so a mix-up wouldn't show. Here it would. The bounds check
//! uses both limits too: col < width, row < height.
//! 
//! ## Size known only at run time
//! 
//! In matmul, WIDTH was a constant, baked into the kernel with index_2d::<WIDTH>().
//! A picture's width comes from the file, so the kernel has to receive it at launch instead.
//! (TODO: name the cuda-oxide API once verified from source; likely index_2d_runtime + RowWidth)
//! 
//! ## What are we acheiving this file?
//! 
//!     Part 0: Load the picture from images/ and turn it into a flat RGB list
//!     Part 1: Context + stream, copy the picture to the GPU, make the box for the gray output, load the kernels
//!     Part 2: The grayscale kernel, a 2D launch config sized to the picture, prepare, launch
//!     Part 3: Copy the gray picture back, check it against the same formula on the CPU, save it as images/gray.png
//!     (Later: same again for the blur.)
//! 

// Imports -----

// RowWidth: wraps an output buffer + its row width, for kernels whose output 
// is a DisJointSlice<_, Runtime2DIndex> (the width is only known at run time).
use cuda_host::{RowWidth, cuda_module};

// `image` loads and saves picture files.
use image::GrayImage;

// cuda_device is the GPU side: thread positions, DisjointSlice, and the macros that make a kernel
// and declare its launch shape
// A DisjoitSlice is cuda-oxide's type for an output list that many threads write to at the same time
// "Disjoint" means not overlapping: each thread may only write to its own spot 
// Thousands of threads run the kernel at once. If two of them wrote to the same spot,
// the result would depend on which one happened to finish last, so you would get a different answer run to run
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};

// cuda_core is the HOST side: open the GPU, make buffers, launch, copy back.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};

// Where the pictures live, relative to the crate folder.
// `cargo oxide run` runs from the crate folder, so these paths work as-is.
//
// `const` = a fixed value, known when the program BUILDS, never changes.
// It has no memory spot of its own: the compiler pastes the value wherever
// it's used. Names are UPPER_CASE convention, and the type must be written.
//
// `&str` = borrowed text: a view onto text stored somewhere else (here, inside the
// program file itself). `&` means "borrowed, not owned". 
// The owned version, which can grow, is `String`.
const INPUT: &str = "images/piperi_sleeping.jpg";

// Where the result goes. .png picks the PNG format (from the extension).
const OUTPUT: &str = "images/gray.png";

// Threads per block in each direction (16 x 16 = 256). Must match the 
// `block = (16, 16, 1)` in the kernel's launch_contract: prepare checks it.
const BLOCK: u32 = 16;

// #[...] above an item is an ATTRIBUTE. The ones below are macros: at BUILD
// time they read the code under them and rewrite or add code.

// #[cuda_module] marks a module that holds kernels. At build time it:
// - compiles every #[kernel] inside to PTX (GPU code) and packs it into our program file;
// - generates kernels::load(&ctx), which hands that GPU code to the driver at run time;
// - generates, per kernel, a prepare_<name>(config) method (checks the launch shape) and a
//      <name>(&stream, &prepared, args...) launcher, type-checked like any Rust call.
#[cuda_module]
mod kernels {
    // `mod` = module, a named box for code. Code inside can't see the files imports by default;
    // `use super::*` pulls in everything from the level above (`super`), `*` = all of it.
    use super::*;

    // #[kernel] marks the function that runs on the GPU, once per thread.
    // It also swaps calls like thread::index_2d_runtime (stubs on the CPU) for the real GPU versions.
    // Kernels can't return a value: results go into an output like `gray`.
    #[kernel]
    // #[launch_bounds(256)]: a promise to the compiler that a block never has more than 256
    // threads (16 x 16), so it can plan the GPU's resources.
    #[launch_bounds(256)]
    // #[launch_contract(...)]: the launch shape this kernel expects.
    // domain = 2 -> must be launched with a 2D config (LaunchConfig2D); checked at build time.
    // block = (16, 16, 1) -> blocks must be exactly 16 x 16 (x 1 deep); checked at RUN time by prepare_grayscale.
    // Not checked: whether the grid covers the whole picture. That's on us.
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn grayscale(
        // Plain numbers: every thread gets the same copy. They come from the file, so they're
        // arguments, not constants like matmul's WIDTH.
        width: u32,
        height: u32,
        // Input: the color picture, 3 bytes per pixel. Read-only, so a 
        // plain slice: many thrads could read the same byte safely.
        rgb: &[u8],
        // Output: 1 byte per pixel. Runtime2DIndex = "2D index whose row width" is given at launch" (not baked in like Index2D<WIDTH>)
        // The host binds that width with RowWidth::new(&mut buf, width).
        mut gray: DisjointSlice<u8, thread::Runtime2DIndex>,
        // Runtime2DIndex means threads write to it with a (row, col) ticket, which
        // get_mut turns into a row * width + col
        // `mut` means we are allowed to change it: `get_mut` hands out spots to write into,
        // and Rust only allows that on something marked `mut`
        //
        // mut gray is a parameter: the host passes it in at launch as RowWidth::new(&mut gray_dev, width),
        // the buffer plus its row width. Runtime2DIndex = 2D from a thread's point of view, with
        // that width carried inside `gray`. index_2d_runtime(&gray) reads it from there to check
        // that the thread's col < width. Height is NOT stored, so we pass it separately and
        // check row ourselves

        // The thread thinks in 2D (row, col) but memory is one flat line,
        // so every read and write is translated 2D -> flat:
        // - input (read): we do it ourselves, (row * width + col) * 3 + channel
        // - output (write): `get_mut` does it for us, row * width + col, from the ticket index_2d_runtime gave us 
    ) {
        // 1. Map: which output pixel is mine? (global index formula)
        // Every thread gets an answer, even off the picture.
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        // 2a. Check col: index_2d_runtime reads the row width bound to 
        // `gray` and returns None if col >= width. It DOES NOT check row.
        // On Some, gray_idx is this thread's "one spot per thread" ticket.
        // 
        // Check col. `if let Some(x)` = value { ... }`: if value is Some, take out what's inside,
        // name it x, and run the block; if None, skip the block (this thread does nothing).
        // index_2d_runtime gives Some(ticket) only if col < width (the width bound to `gray`).
        // It DOES NOT check row. The ticket holds this thread's (row, col); get mut turns it into 
        // row * width + col later.
        // It is checking if this thread is within the limits of the output or not; if it is not, do nothing (skip).
        if let Some(gray_idx) = thread::index_2d_runtime(&gray) {
            // index_2d_runtime(&gray) works out the thread's row and col itself, from the built-in values
            // (block * block size + position), not from our `row` / `col` variables. It checks col < width
            // (gray's width) but does NOT linearize: the ticket holds the plain (row, col). The flat spot
            // row * width + col is only made later during writing, in gray.get_mut(ticket), using gray's own width.

            // 2b. Check the row ourselves. `as usize`: row is usize, height u32
            if row < height as usize {
                // 3. Linearize the input: pixel number, then * 3 + channel
                // We need to linearize since the thread thinks in (row, col), but the 
                // input picture is stored flat in memory.
                // The line below gives which pixel from the input the thread needs
                let pixel = row * width as usize + col;
                // `as f32`: bytes 0..=255 become decimals so we can mix them
                let r = rgb[pixel * 3] as f32; // channel 0
                let g = rgb[pixel * 3 + 1] as f32; // channel 1
                let b = rgb[pixel * 3 + 2] as f32; // channel 2

                // 4. Calculate: weighted mix into one brightness
                // Weights add up to 1.0, so the result stays in 0..=255
                let value = 0.21 * r + 0.72 * g + 0.07 * b;

                // 5. Write: get_mut returns the ticket into this thread's spot
                // (row * width + col; 1 value per pixel, so no * 3).
                // `as u8` cuts off the decimals (117.5 -> 117).

                // Linearizing happens whenever the thread touches memory:
                // read (step 3): we find R G B in the flat color buffer ourselves
                // write (step 5): get_mut finds the spot in the flat gray buffer, then *out = value puts the result there;
                //      finding which spot the thread is writing to in memory
                if let Some(out) = gray.get_mut(gray_idx) {
                    // `out` is a &mut u8: an address pointing into the gray buffer.
                    // `*out` = the value at that address (dereference),
                    // so `*out = ...` writes into the buffer itself.
                    *out = value as u8;
                }
                // This checks col < width again and that the spot is inside the buffer; if not
                // None, and the write is skipped. With correct host code (buffer = width * height,
                // same width bound) it's always Some: a safety net, so a host bug gives missing pixels
                // instead of corrupted memory.
            }
        }
    }
}

// 1. if let Some(gray_idx) = thread::index_2d_runtime(&gray): if the thread's col < width, take out the ticket and call it gray_idx.
// 2. if row < height: a plain check, our own 
// 3. if let Some(out) = gray.get_mut(gray_idx): if the ticket's spot (row * width + col) is inside gray_dev (below its length
//      and col < width checked once more), take out a reference to that spot and call it out, then write there.

// "Ticket" a value only index_2d_runtime can make. It holds this thread's (row, col) and proves
// the thread passed the col < width check. get_mut only accepts a ticket, not a plain number
// so a thread can only ever write to its own spot. 

// The whole kernel in short:
// - The host passes gray_dev (flat) bound to `width`, plus width and height.
//      Input and output are the same picture size, so one width fits both.
// - Every thread gets a (row, col), even off the picture.
// - index_2d_runtime checks col < width; we check row < height ourselves.
//      Off-picture threads stop here.
// - The thread linearizes to READ its R, G, B from the flat color buffer, mixes them into one value
//      then get_mut linearizes to find its spot in the flat gray_dev, and *out = value WRITEs it there.

// main returns a Result: Ok(()) if everything worked, or an error.
// `Box<dyn std::error::Error` = "any kind of error":
// - std:error::Error is a trait (a set of abilities) that every error type has, e.g. it can describe itself in a message
// - `dyn` = the exact type is only known when the program runs (an image error, a GPU error...).
// - `Box` = put it in its own box of memory, because different error types have different sizes and Rust needs one fixed size to return.
// That's why `?` works on both image errors and GPU errors in main
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the picture and turn it into a flat list of bytes
    // 
    // image::open reads the file and decodes it. The format comes from the file extension (.png, .jpg). It
    // can fail (missing file, unsupported format), so it returns a Result and `?` passes any error out of main.
    let picture = image::open(INPUT)?;

    // Files come in many shapes: RGB, RGBA (with a 4th transparency value), gray, 16 bits per value...
    // rgb8() converts whatever we got into one fixed shape: 3 values per pixel (R, G, B) each a u8 (0..=255).
    // Transparency is dropped. Now every pixel is exactly 3 bytes.
    let rgb = picture.to_rgb8();

    // Thw size comes from the file, so we only know it NOW, at run time.
    // That's why the kernel can't use a constant like matmul's WIDTH.
    let width = rgb.width(); // pixels across = number of columns (u32)
    let height = rgb.height(); // pixels down = number of rows (u32)

    // into_raw() gives up the picture wrapper and hands us the bytes inside:
    // one flat Vec<u8>, row-major, R G B R G B ...
    let rgb_host: Vec<u8> = rgb.into_raw();

    // Sanity check of that layout: 3 values per pixel, width * height pixels.
    // `as usize`: width/height are u32, Vec lengths are usize, so convert.
    assert_eq!(rgb_host.len(), width as usize * height as usize * 3);

    println!("Loaded {INPUT}: {width} x {height} pixels, {} values", rgb_host.len());
    // Pixel (0, 0) is spots 0, 1, 2 (its R, G, B).
    println!("Pixel (0, 0): R = {}, G = {}, B = {}", rgb_host[0], rgb_host[1], rgb_host[2]);

    // 2. Open the GPU + get the queue 
    //
    // Context = our connection to one GPU. 0 = the first GPU on the machine.
    // All setup (making buffers, loading kernels) goes through it.
    let ctx = CudaContext::new(0)?;
    // The queue that comes with the context. Copies and kernel launches are
    // put on it and the GPU runs them IN ORDER, while the CPU carries on.
    let stream = ctx.default_stream();

    // 3. Copy the picture to the GPU, make the gray output buffer
    //
    // The GPU has its own memory, separate from the CPU's. Anything the kernel reads
    // must be copied over first; a DeviceBuffer is a list that lives in GPU memory.
    // We can't read it directly from the CPU.
    //
    // from_host: reserves GPU memory (right away), queues the copy, then WAITS until
    // the bytes are on the GPU. &rgb_host lends the Vec, so we keep it for the CPU check layer.
    // Type: DeviceBuffer<u8>.
    let rgb_dev = DeviceBuffer::from_host(&stream, &rgb_host)?;

    // zeroed: reserves GPU memory (right away) and queues filling it with 0.
    // Does NOT wait. One byte per pixel, so width * height of them.
    // `mut`: the kernel will write into it.
    // (Limitation: a pixel the kernel forgets ro write stays 0 = block, so the
    // CPU check can't tell a missing write from a truly black pixel).
    let pixel_count = width as usize * height as usize;
    let mut gray_dev: DeviceBuffer<u8> = DeviceBuffer::zeroed(&stream, pixel_count)?;

    // 4. Load the kernels
    //
    // At build time, #[cuda_module] compiled our kernel to PTX (GPU code) and 
    // packed it into this program. `load` handes that code to the GPU driver,
    // which turns it into instructions for this exact GPU.
    // `unsafe`: we promise the packed code really comes from this module;
    // Rust can't check that itself. `module` holds the generated 
    // `prepare_grayscale` and grayscale methods.
    let module = unsafe { kernels::load(&ctx)? };
    // `unsafe`: the packed GPU code (PTX) and the generated Rust methods are
    // made separately at build time; they must match (same arguments, same order).
    // Rust can't look inside of PTX to check, so we promise it.
    // cargo oxide builds both from the same source, so in practice it holds.

    // 5. Launch: a grid sized to the PICTURE, then queue the kernel
    //
    // The kernel's attributes say what's ALLOWED; this config is what ACTUALLY
    // happens this run. Round UP, separately per direction:
    // blocks across cover the width (columns), blocks down the height (rows).
    // For 1280 x 591: 80 x 37 blocks; the bottom blocks row hangs one over.
    let blocks_x = width.div_ceil(BLOCK);
    let blocks_y = height.div_ceil(BLOCK);
    // LaunchConfig2D::new(grid, block, shared_mem):
    // grid = (blocks across, blocks down) from the picture's size
    // block = (threads across, threads down) per block; z is 1
    // shared_mem = bytes of shared memory per block (on-chip block memory); we use none, so 0.
    let config = LaunchConfig2D::new((blocks_x, blocks_y), (BLOCK, BLOCK), 0);
    println!("Grid: {blocks_x} x {blocks_y} blocks of {BLOCK} x {BLOCK} threads");

    // prepare_grayscale (generated by #[cuda_module]) checks the config against the kernel's 
    // launch_contract: 2-dimensional launch with blocks exactly 16 x 16 x 1, and within the GPU's limits. 
    // Errors come back here, before any launch.
    // NOT checked: does this grid cover the picture? That's on us.
    let prepared = module.prepare_grayscale(config)?;

    // grayscale (generated) = the launcher. First the queue and the prepared shape,
    // then the kernel's arguments in the same order as its parameters:
    //      width, height -> plain u32s, every thread gets the same copy
    //      &rgb_dev -> the kernel's `rgb: &[u8]` (borrowed, read only)
    //      RowWidth::new -> the kernel's `gray`: the buffer (&mut: it gets written) + its row
    //          width, so index_2d_runtime and get_mut know the width.
    // It also checks the queue belongs to the same GPU as the module.
    // This only QUEUES the kernel and returns at once: the GPU runs it later.
    // A crash inside the kernel shows up at the next waiting call.
    module.grayscale(&stream, &prepared, width, height, &rgb_dev, RowWidth::new(&mut gray_dev, width))?;
    println!("Kernel queued");

    // gray_dev vs width: different things
    //      width = pixels in ONE row (1280), used for col < width check and linearizing
    //      gray_dev.len() = values in the WHOLE gray bufer, 1 per pixel
    // The buffer is flat and only knows its total length. width says
    // where each row starts, so the col < width check and get_mut's row * width + col need it.
    // RowWidth::new(&mut gray_dev, width) joins thw two for the kernel. The length is only a safety net
    // (get_mut checks the spot < gray_dev.len()).

    // 6. Copy the gray picture back
    //
    // to_host_vec queues the copy back on the same queue, then WAITS for everything before it
    // to finish, including our kernel. So this is the first line where the kernel has really run,
    // and where a crash inside it would show up as an error.
    let gray_host: Vec<u8> = gray_dev.to_host_vec(&stream)?;

    // 7. Check against the same formula on the CPU
    //
    // A plain loop over every pixel: no threads, no (row, col) needed,
    // because pixel i's R, G, B sit at i * 3, i * 3 + 1, i * 3 + 2.
    // We allow a difference of 1: the GPU can round the mix slightly differently 
    // (it may do a multiply and add in one step), and 
    // `as u8` cutting off decimals can then land 1 lower on one side.
    let mut mismatches = 0;
    for i in 0..pixel_count {
        let r = rgb_host[i * 3] as f32;
        let g = rgb_host[i * 3 + 1] as f32;
        let b = rgb_host[i * 3 + 2] as f32;
        let expected = (0.21 * r + 0.72 * g + 0.07 * b) as u8;

        // abs_diff: the distance between two numbers, never negative
        // (u8 can't hold negatives, so a - b could fail; this can't).
        if gray_host[i].abs_diff(expected) > 1 {
            // Print only the first 5, so a total failure doesn't flood
            // the terminal. i % width = col, i / width = row.
            if mismatches < 5 {
                let (row, col) = (i / width as usize, i % width as usize);
                println!(
                    "Mismatch at pixel ({row}, {col}): GPU {}, CPU {expected}",
                    gray_host[i]
                );
            }
            mismatches += 1;
        }
    }

    if mismatches == 0 {
        println!("SUCCESS: all {pixel_count} pixels match");
    } else {
        println!("FAILED: {mismatches} of {pixel_count} pixels differ");
    }

    // 8. Save the gray picture
    //
    // from_raw wraps our flat bytes as a picture: width x height, 1 byte per pixel,
    // row-major (the same layout the kernel wrote). It returns None if the Vec is too
    // short for the width * height, so we turn that into an
    // error with ok_or and `?` passes it out ouf main.
    let gray_image = GrayImage::from_raw(width, height, gray_host)
        .ok_or("gray buffer has the wrong size")?;
    gray_image.save(OUTPUT)?;
    println!("Saved {OUTPUT}");

    Ok(())
}
