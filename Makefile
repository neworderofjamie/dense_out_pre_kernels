
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-lineinfo -c -x cu -arch sm_30 -std=c++11 -O3

all: kernel

kernel: kernel.o
	$(CXX) -o kernel kernel.o -L/usr/local/cuda/lib64 -lcuda -lcudart
	
kernel.o: kernel.cu
	$(NVCC) $(NVCCFLAGS)  kernel.cu

clean:
	rm -f kernel kernel.o
