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
// It has non memory spot of its own: the compiler pastes the value wherever
// it's used. Names are UPPER_CASE convention, and the type must be written.
//
// `&str` = borrowed text: a view onto text stored somewhere else (here, inside the
// program file itself). `&` means "borrowed, not owned". 
// The owned version, which can grow, is `String`.
const INPUT: &str = "images/piperi_sleeping.jpg";

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
    // Not checked: whether the grid covers the whole picture. That's on use.
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn grayscale(
        // Plain numbers: every thread gets the same copy. They come from the file, so they're
        // arguments, not constants like matmul's WIDTH.
        width: u32,
        height: u32,
        // Input: the color picture, 3 bytes per pixel. Read-only, so a 
        // plain slice: many thrads could read the same byte safely.
        rgb: &[u8],
        // Output: 1 byte per pixel. Runtime2DIndex = "2D index whose row width"
        // is given at launch" (not baked in like Index2D<WIDTH>)/
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
        if let Some(gray_idx) = thread::index_2d_runtime(&gray) {
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

    Ok(())
}
