.PHONY: all build clean c-kernels rust cython test bench

BUILD_DIR := build
RUST_DIR := rust/aria-scheduler
CYTHON_DIR := cython

all: c-kernels rust cython

# Backward-compatible alias used by older scripts/CI.
build: c-kernels

# C/C++ kernels via CMake
c-kernels:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make -j$$(nproc)

# Rust scheduler
rust:
	cd $(RUST_DIR) && cargo build --release

# Rust with Python bindings
rust-python:
	cd $(RUST_DIR) && cargo build --release --features python

# Cython bridge
cython:
	cd $(CYTHON_DIR) && python setup.py build_ext --inplace

# Run all tests
test: test-c test-rust test-cython

test-c: c-kernels
	cd $(BUILD_DIR) && ctest --output-on-failure

test-rust:
	cd $(RUST_DIR) && cargo test

test-cython: cython
	cd $(CYTHON_DIR) && python -m pytest test_bridge.py -v

# Benchmarks
bench: c-kernels rust
	cd $(BUILD_DIR) && ./bench_kernels 2>/dev/null || echo "No C benchmarks yet"
	cd $(RUST_DIR) && cargo bench 2>/dev/null || echo "No Rust benchmarks yet"

clean:
	rm -rf $(BUILD_DIR)
	cd $(RUST_DIR) && cargo clean
	rm -f $(CYTHON_DIR)/*.so $(CYTHON_DIR)/*.c
