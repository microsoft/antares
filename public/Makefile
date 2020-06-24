convfwd_multialgo:
	/opt/rocm/hip/bin/hipcc hip_convfwd_multialgo.cc -I/opt/rocm/include -L/opt/rocm/lib -lMIOpen -std=c++11 -O2 -o hip_convfwd_multialgo -Wno-unused-result -Wno-ignored-attributes && ./hip_convfwd_multialgo

convfwd_multialgo-cuda:
	/usr/local/cuda/bin/nvcc hip_convfwd_multialgo_cuda.cc -lcudnn -std=c++11 -O2 -o cuda_convfwd_multialgo && ./cuda_convfwd_multialgo


matmul:
	/opt/rocm/hip/bin/hipcc hip_matmul.cc -I/opt/rocm/include -L/opt/rocm/lib -lrocblas -std=c++11 -O2 -o hip_matmul -Wno-unused-result -Wno-ignored-attributes && ./hip_matmul

matmul-cuda:
	HIP_PLATFORM=nvcc /opt/rocm/hip/bin/hipcc hip_matmul.cc -lcublas -std=c++11 -O2 -o cuda_matmul && ./cuda_matmul

batch_matmul:
	/opt/rocm/hip/bin/hipcc hip_batch_matmul.cc -I/opt/rocm/include -L/opt/rocm/lib -lrocblas -std=c++11 -O2 -o hip_batch_matmul -Wno-unused-result -Wno-ignored-attributes && ./hip_batch_matmul

batch_matmul-cuda:
	HIP_PLATFORM=nvcc /opt/rocm/hip/bin/hipcc hip_batch_matmul.cc -lcublas -std=c++11 -O2 -o cuda_batch_matmul && ./cuda_batch_matmul

clean:
	rm -rf hip_matmul hip_batch_matmul
