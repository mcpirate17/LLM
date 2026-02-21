CC = gcc
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra
LDFLAGS = -shared

SRC_DIR = src
LIB_DIR = lib
OBJ_DIR = obj

# Library sources (exclude test files from shared lib)
SRCS = $(filter-out $(SRC_DIR)/test_%.c, $(wildcard $(SRC_DIR)/*.c))
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
TARGET = $(LIB_DIR)/libaria_runtime.so

.PHONY: all build clean test

all: build

build: $(TARGET)

$(TARGET): $(OBJS) | $(LIB_DIR)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)

# Basic sanity test for the shared library
# Includes kernel coverage for native data I/O helpers (CSV loader, binary reader, text writer).
test: build
	$(CC) $(CFLAGS) tests/test_runtime.c -L$(LIB_DIR) -laria_runtime -Wl,-rpath,$(CURDIR)/$(LIB_DIR) -o tests/run_runtime_tests -lm
	./tests/run_runtime_tests
	$(CC) $(CFLAGS) tests/test_kernels.c -L$(LIB_DIR) -laria_runtime -Wl,-rpath,$(CURDIR)/$(LIB_DIR) -o tests/run_kernel_tests -lm
	./tests/run_kernel_tests
	rm tests/run_runtime_tests tests/run_kernel_tests
