# aria_designer/runtime/Makefile
# Compiles the high-performance C++ execution backend by pulling sources from aria_core.

CC = g++
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra -I../../aria_core/include
LDFLAGS = -shared

# Source directories
CORE_SRC_DIR = ../../aria_core/src/cpu
LIB_DIR = lib
OBJ_DIR = obj

# Pull sources from aria_core (exclude graph_executor which needs torch)
SRCS = $(CORE_SRC_DIR)/kernels.cpp $(CORE_SRC_DIR)/graph_validator.cpp \
       $(CORE_SRC_DIR)/shape_inference.cpp $(CORE_SRC_DIR)/clifford.cpp \
       $(CORE_SRC_DIR)/hyperbolic.cpp
OBJS = $(patsubst $(CORE_SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))
TARGET = $(LIB_DIR)/libaria_runtime.so

.PHONY: all build clean test

all: build

build: $(TARGET)

$(TARGET): $(OBJS) | $(LIB_DIR)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

$(OBJ_DIR)/%.o: $(CORE_SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)

# Basic sanity test for the shared library
test: build
	$(CC) $(CFLAGS) tests/test_runtime.cpp -L$(LIB_DIR) -laria_runtime -Wl,-rpath,$(CURDIR)/$(LIB_DIR) -o tests/run_runtime_tests -lm
	./tests/run_runtime_tests
	$(CC) $(CFLAGS) tests/test_kernels.cpp -L$(LIB_DIR) -laria_runtime -Wl,-rpath,$(CURDIR)/$(LIB_DIR) -o tests/run_kernel_tests -lm
	./tests/run_kernel_tests
	rm tests/run_runtime_tests tests/run_kernel_tests
