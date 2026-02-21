# Makefile for ARM NEON LM Head implementation
# Target: ARM Cortex-A53 (Xilinx KV260)

# ============================================================================
# Cross-compilation settings
# ============================================================================

# For native compilation on ARM (run directly on KV260)
CXX_NATIVE = g++

# Aggressive optimization flags for ARM Cortex-A53
# -O3: Maximum optimization level
# -march=armv8-a+simd: Enable ARMv8-A with NEON SIMD
# -mtune=cortex-a53: Tune specifically for Cortex-A53
# -mcpu=cortex-a53: Generate code for Cortex-A53
# -ffast-math: Aggressive floating-point optimizations (non-IEEE)
# -fno-math-errno: Don't set errno for math functions
# -funsafe-math-optimizations: Allow unsafe FP optimizations
# -ffinite-math-only: Assume no NaN/Inf
# -fno-signed-zeros: Ignore sign of zero
# -fno-trapping-math: Assume FP ops don't trap
# -freciprocal-math: Allow x/y -> x*(1/y)
# -funroll-loops: Unroll loops for better pipelining
# -fomit-frame-pointer: Free up a register
# -ftree-vectorize: Enable auto-vectorization
# -ftree-loop-vectorize: Vectorize loops
# -fvect-cost-model=unlimited: Don't limit vectorization by cost
# -fprefetch-loop-arrays: Prefetch loop arrays
# -flto: Link-time optimization
# -fuse-linker-plugin: Use linker plugin for LTO
# -fipa-pta: Interprocedural pointer analysis
# -fno-plt: Avoid PLT for faster function calls
# -fno-semantic-interposition: Assume no interposition
# -fgcse-after-reload: Global CSE after reload
# -fipa-cp-clone: Clone functions for interprocedural constant propagation
# -ftree-partial-pre: Partial redundancy elimination
# -fschedule-insns: Instruction scheduling
# -fschedule-insns2: Post-register-allocation scheduling
# -fpeel-loops: Peel loops for better optimization
# -ftracer: Tail duplication for superblock formation
# -DNDEBUG: Disable assertions

CXXFLAGS_AGGRESSIVE = -O3 -march=armv8-a+simd -mtune=cortex-a53 -mcpu=cortex-a53 \
                      -ffast-math -fno-math-errno -funsafe-math-optimizations \
                      -ffinite-math-only -fno-signed-zeros -fno-trapping-math \
                      -freciprocal-math -funroll-loops -fomit-frame-pointer \
                      -ftree-vectorize -ftree-loop-vectorize -fvect-cost-model=unlimited \
                      -fprefetch-loop-arrays -flto -fuse-linker-plugin \
                      -fipa-pta -fno-plt -fno-semantic-interposition \
                      -fgcse-after-reload -fipa-cp-clone -ftree-partial-pre \
                      -fschedule-insns -fschedule-insns2 -fpeel-loops -ftracer \
                      -DNDEBUG -std=c++14

# Standard optimization (for debugging/comparison)
CXXFLAGS_STANDARD = -O3 -march=armv8-a -mtune=cortex-a53 \
                    -ffast-math -funroll-loops -fomit-frame-pointer \
                    -Wall -Wextra -std=c++14

# Select optimization level (default: aggressive)
CXXFLAGS_NATIVE = $(CXXFLAGS_AGGRESSIVE)

# For cross-compilation from x86 host (using Xilinx SDK or aarch64 toolchain)
CROSS_COMPILE ?= aarch64-linux-gnu-
CXX_CROSS = $(CROSS_COMPILE)g++
CXXFLAGS_CROSS = $(CXXFLAGS_AGGRESSIVE)

# Linker flags (with LTO support)
LDFLAGS = -lpthread -flto -fuse-linker-plugin

# PyBind11 settings (for native build on KV260)
PYTHON_VERSION ?= python3
PYBIND_INCLUDE := $(shell $(PYTHON_VERSION) -m pybind11 --includes)
PYTHON_EXT := $(shell $(PYTHON_VERSION)-config --extension-suffix)
# Use aggressive optimization for pybind as well, but without LTO (may cause issues)
PYBIND_CXXFLAGS = -O3 -march=armv8-a+simd -mtune=cortex-a53 -mcpu=cortex-a53 \
                  -ffast-math -fno-math-errno -funsafe-math-optimizations \
                  -ffinite-math-only -fno-signed-zeros -fno-trapping-math \
                  -freciprocal-math -funroll-loops -fomit-frame-pointer \
                  -ftree-vectorize -ftree-loop-vectorize -fvect-cost-model=unlimited \
                  -fprefetch-loop-arrays -fipa-pta -fno-plt \
                  -fgcse-after-reload -fipa-cp-clone -ftree-partial-pre \
                  -fschedule-insns -fschedule-insns2 -fpeel-loops -ftracer \
                  -DNDEBUG -std=c++14 \
                  -fPIC -shared -fvisibility=hidden $(PYBIND_INCLUDE)
PYBIND_LDFLAGS = -lpthread

# ============================================================================
# Source files
# ============================================================================

HEADERS = lm_head_neon.hpp lm_head_neon_mt.hpp
SOURCES = test_lm_head_neon.cpp
PYBIND_SOURCES = lm_head_pybind.cpp
TARGET = test_lm_head_neon
PYBIND_TARGET = lm_head_neon$(PYTHON_EXT)

# ============================================================================
# Build targets
# ============================================================================

.PHONY: all native cross clean help pybind

# Default: native compilation (for building on KV260 directly)
all: native

# Native build (run on ARM target) - Aggressive optimization
native: $(SOURCES) $(HEADERS)
	@echo "Building for native ARM execution (aggressive optimization)..."
	@echo "Flags: $(CXXFLAGS_NATIVE)"
	$(CXX_NATIVE) $(CXXFLAGS_NATIVE) -o $(TARGET) $(SOURCES) $(LDFLAGS)
	@echo "Built: $(TARGET)"

# Native build with standard optimization (for comparison)
native-std: $(SOURCES) $(HEADERS)
	@echo "Building for native ARM execution (standard optimization)..."
	$(CXX_NATIVE) $(CXXFLAGS_STANDARD) -o $(TARGET)_std $(SOURCES) -lpthread
	@echo "Built: $(TARGET)_std"

# PyBind11 module (native build only, requires pybind11 installed)
pybind: $(PYBIND_SOURCES) $(HEADERS)
	@echo "Building PyBind11 module (aggressive optimization)..."
	@echo "Python: $(PYTHON_VERSION)"
	@echo "Extension: $(PYTHON_EXT)"
	$(CXX_NATIVE) $(PYBIND_CXXFLAGS) -o $(PYBIND_TARGET) $(PYBIND_SOURCES) $(PYBIND_LDFLAGS)
	@echo "Built: $(PYBIND_TARGET)"
	@echo "Usage: python3 -c 'import lm_head_neon; print(lm_head_neon.HIDDEN_SIZE)'"

# Cross-compilation (run on x86 host, deploy to ARM)
cross: $(SOURCES) $(HEADERS)
	@echo "Cross-compiling for ARM Cortex-A53..."
	$(CXX_CROSS) $(CXXFLAGS_CROSS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
	@echo "Built: $(TARGET)"

# Debug build (native)
debug: $(SOURCES) $(HEADERS)
	@echo "Building debug version..."
	$(CXX_NATIVE) -g -O0 -march=armv8-a -std=c++14 -o $(TARGET)_debug $(SOURCES) $(LDFLAGS)
	@echo "Built: $(TARGET)_debug"

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET)_std $(TARGET)_debug *.o *.bin $(PYBIND_TARGET) lm_head_neon*.so

# Help
help:
	@echo "NEON LM Head Makefile (Aggressive Optimization)"
	@echo ""
	@echo "Targets:"
	@echo "  native     - Build with aggressive optimization (default)"
	@echo "  native-std - Build with standard optimization (for comparison)"
	@echo "  pybind     - Build PyBind11 Python module"
	@echo "  cross      - Cross-compile from x86 host"
	@echo "  debug      - Build with debug symbols"
	@echo "  clean      - Remove build artifacts"
	@echo ""
	@echo "Optimization Flags (aggressive):"
	@echo "  -O3, -flto, -ffast-math, -march=armv8-a+simd, -mcpu=cortex-a53"
	@echo "  -funroll-loops, -ftree-vectorize, -fprefetch-loop-arrays, etc."
	@echo ""
	@echo "Variables:"
	@echo "  CROSS_COMPILE   - Cross-compiler prefix (default: aarch64-linux-gnu-)"
	@echo "  PYTHON_VERSION  - Python interpreter (default: python3)"
	@echo ""
	@echo "Example usage:"
	@echo "  make native                          # Aggressive optimization"
	@echo "  make native-std                      # Standard optimization"
	@echo "  make pybind                          # Build Python module"
	@echo ""
	@echo "Run C++ test:"
	@echo "  ./test_lm_head_neon                              # Random test data"
	@echo "  ./test_lm_head_neon ../weight_bin_5 input.bin    # With weight files"
	@echo ""
	@echo "Run Python test:"
	@echo "  python3 test_lm_head.py                          # Random test data"
	@echo "  python3 test_lm_head.py ../weight_bin_5 input.bin # With data files"
	@echo "Targets:"
	@echo "  native  - Build for native ARM execution (default)"
	@echo "  pybind  - Build PyBind11 Python module"
	@echo "  cross   - Cross-compile from x86 host"
	@echo "  debug   - Build with debug symbols"
	@echo "  clean   - Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  CROSS_COMPILE   - Cross-compiler prefix (default: aarch64-linux-gnu-)"
	@echo "  PYTHON_VERSION  - Python interpreter (default: python3)"
	@echo ""
	@echo "Example usage:"
	@echo "  make native                          # Build test binary on KV260"
	@echo "  make pybind                          # Build Python module"
	@echo "  make pybind PYTHON_VERSION=python3.8 # Build with specific Python"
	@echo "  make cross CROSS_COMPILE=aarch64-xilinx-linux-  # Cross-compile with Xilinx SDK"
	@echo ""
	@echo "Run C++ test:"
	@echo "  ./test_lm_head_neon                              # Random test data"
	@echo "  ./test_lm_head_neon ../weight_bin_5 input.bin    # With weight files"
	@echo ""
	@echo "Run Python test:"
	@echo "  python3 test_lm_head.py                          # Random test data"
	@echo "  python3 test_lm_head.py --data-dir ./test_data   # With data files"
