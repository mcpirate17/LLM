use std::fs;
use std::path::Path;

fn main() {
    println!(
        "cargo:rustc-link-search=native=/home/tim/Projects/LLM/research/runtime/native/build"
    );
    println!("cargo:rustc-link-lib=static=aria_native_runtime_static");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=gomp");

    // Link the scipy-bundled OpenBLAS discovered in the active local venv.
    let scipy_libs = "/home/tim/venvs/llm/lib/python3.12/site-packages/scipy.libs";
    println!("cargo:rustc-link-search=native={}", scipy_libs);
    let openblas_name = find_scipy_openblas_name(scipy_libs)
        .expect("could not find libscipy_openblas*.so in scipy.libs");
    println!("cargo:rustc-link-lib=dylib={}", openblas_name);

    // Use RPATH to ensure the shared library can be found at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", scipy_libs);
}

fn find_scipy_openblas_name(scipy_libs: &str) -> Option<String> {
    let dir = Path::new(scipy_libs);
    let mut candidates: Vec<String> = fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.starts_with("libscipy_openblas") && name.ends_with(".so"))
        .collect();
    candidates.sort_unstable();
    let filename = candidates.into_iter().next()?;
    Some(
        filename
            .trim_start_matches("lib")
            .trim_end_matches(".so")
            .to_string(),
    )
}
