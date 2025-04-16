# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++17 -fopenmp
NVCCFLAGS = -Xcompiler "-Wall -Wextra" -std=c++17 -arch=sm_52

# Source files
CPPSRCS = serial.cpp utils.cpp
SHARED_SRCS = shared.cpp utils.cpp
CU_SRCS = gpu.cu utils.cpp

# Object files
CPPOBJS = $(CPPSRCS:.cpp=.o)
CUOBJS = $(CU_SRCS:.cu=.o)

# Output binary
SERIAL_TARGET = serial
SHARED_TARGET = shared
GPU_TARGET = gpu

# Default rule
all: $(SERIAL_TARGET) $(GPU_TARGET) $(SHARED_TARGET)

# Serial executable
$(SERIAL_TARGET): serial.o utils.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Shared executable
$(SHARED_TARGET): shared.o utils.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# GPU executable
$(GPU_TARGET): gpu.o utils.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile C++ source
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f *.o $(SERIAL_TARGET) $(GPU_TARGET) $(SHARED_TARGET)
