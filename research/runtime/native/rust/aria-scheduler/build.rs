fn main() {
    println!("cargo:rustc-link-search=native=/home/tim/Projects/LLM/research/runtime/native/build");
    println!("cargo:rustc-link-lib=static=aria_native_runtime_static");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=gomp");
    
    // Link OpenBLAS from scipy.libs
    let scipy_libs = "/home/tim/venvs/llm/lib/python3.12/site-packages/scipy.libs";
    println!("cargo:rustc-link-search=native={}", scipy_libs);
    println!("cargo:rustc-link-lib=dylib=scipy_openblas-b75cc656");
    
    // Use RPATH to ensure the shared library can be found at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", scipy_libs);
}
