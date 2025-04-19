# Compiler
CXX = g++
NVCC = nvcc
MPICXX = mpicxx

# Compiler flags
CXXFLAGS = -std=c++17 -fopenmp -fPIC
NVCCFLAGS = -Xcompiler -fPIC -std=c++17 -arch=sm_52
MPICXXFLAGS = -std=c++17 -fopenmp -fPIC

# Source files
CPPSRCS = serial.cpp utils.cpp
SHARED_SRCS = shared.cpp utils.cpp
CU_SRCS = gpu.cu
SHARED_GPU_CPPSRCS = shared-gpu.cpp utils.cpp
MPI_SRCS = distributed.cpp utils.cpp
DISTRIBUTED_GPU_SRCS = distributed-gpu.cpp utils.cpp

# Object files
CPPOBJS = $(CPPSRCS:.cpp=.o)
SHARED_GPU_CPPOBJS = $(SHARED_GPU_CPPSRCS:.cpp=.o)
CUOBJS = $(CU_SRCS:.cu=.o)
MPIOBJS = $(MPI_SRCS:.cpp=.o)
DISTRIBUTED_GPU_OBJS = $(DISTRIBUTED_GPU_SRCS:.cpp=.o)

# Output binary
SERIAL_TARGET = serial
SHARED_TARGET = shared
SHARED_GPU_TARGET = shared-gpu
DISTRIBUTED_TARGET = distributed
DISTRIBUTED_GPU_TARGET = distributed-gpu

# Default rule
all: $(SERIAL_TARGET) $(SHARED_GPU_TARGET) $(SHARED_TARGET) $(DISTRIBUTED_TARGET)

# Serial executable
$(SERIAL_TARGET): serial.o utils.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Shared executable
$(SHARED_TARGET): shared.o utils.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# GPU executable
$(SHARED_GPU_TARGET): gpu.o utils.o shared-gpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lcudart -lstdc++

# MPI executable
$(DISTRIBUTED_TARGET): distributed.o utils.o
	$(MPICXX) $(MPICXXFLAGS) -o $@ $^

# GPU MPI executable
$(DISTRIBUTED_GPU_TARGET): distributed-gpu.o utils.o gpu.o
	$(MPICXX) $(MPICXXFLAGS) -o $@ $^ -lcudart -lstdc++

# Compile MPI source
distributed.o: distributed.cpp
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@
distributed-gpu.o: distributed-gpu.cpp
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

# Compile C++ source
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f *.o $(SERIAL_TARGET) $(SHARED_GPU_TARGET) $(SHARED_TARGET) $(DISTRIBUTED_TARGET) $(DISTRIBUTED_GPU_TARGET)
