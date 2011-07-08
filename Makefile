all:cpu.o
	nvcc cpu.o -o cuda

cpu.o:matrix_kernel.cu cpu.cu
	nvcc -I .,/usr/share/cuda-sdk/C/common/inc/,/usr/include/cuda -ccbin /opt/gcc-4.4/ -c cpu.cu matrix_kernel.cu -O4 -Xptxas -v --compiler-options -Wall,-fpermissive 

clean:
	rm -rf *.o cuda