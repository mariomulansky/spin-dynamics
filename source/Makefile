ARCH = sm_13

CUDA_ROOT = /usr/local/cuda

CC = gcc
CXX = g++
NVCC = nvcc

LDLIBS = -lcudart -lgomp
LDFLAGS = -L$(CUDA_ROOT)/lib64
INCLUDES = -I${BOOST_ROOT}

NVCCFLAGS = -O3 $(INCLUDES) -arch $(ARCH) --use_fast_math 
#-Xcompiler -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP
#--compiler-bindir=/usr/bin/g++-4.4

%.co : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

spin_dynamics.co: spin_dynamics.cu
spin_dynamics: spin_dynamics.co
	$(CC) -o spin_dynamics $(INCLUDES) $(LDLIBS) $(LDFLAGS) spin_dynamics.co

spin_relax.co: spin_relax.cu
spin_relax: spin_relax.co
	$(CC) -o spin_relax $(INCLUDES) $(LDLIBS) $(LDFLAGS) spin_relax.co

short_time.co: short_time.cu
short_time: short_time.co
	$(CC) -o short_time $(INCLUDES) $(LDLIBS) $(LDFLAGS) short_time.co
