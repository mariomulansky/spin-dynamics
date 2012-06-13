ARCH = sm_13

NVCC = nvcc

LDLIBS = -lcudart -lgomp
LDFLAGS = -L$(CUDA_ROOT)/lib64

NVCCFLAGS = -O3 $(INCLUDES) -arch $(ARCH)  -Xcompiler -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP
#--compiler-bindir=/usr/bin/g++-4.4

%.co : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

spin_dynamics.co: spin_dynamics.cu
spin_dynamics: spin_dynamics.co
