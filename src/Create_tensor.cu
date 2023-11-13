#include "Create_tensor.cuh"

__global__ void process(const unsigned char* srcData, float* tgtData, const int h, const int w)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * w;
    int idx3 = idx * 3;

    if (ix < w && iy < h)
    {
        tgtData[idx] = static_cast<float>(srcData[idx3 + 2]) / 255.0f;
        tgtData[idx + h * w] = static_cast<float>(srcData[idx3 + 1]) / 255.0f;
        tgtData[idx + h * w * 2] = static_cast<float>(srcData[idx3]) / 255.0f;
    }
}

// 创建张量
Ort::Value create_tensor(unsigned char* aImg,
	const std::vector<int64_t>& tensor_dims,
	const Ort::MemoryInfo& memory_info_handler,
	std::vector<float>& tensor_value_handler)
	throw(std::runtime_error)
{
	const unsigned int target_channel = tensor_dims.at(1);
	const unsigned int target_height = tensor_dims.at(2);
	const unsigned int target_width = tensor_dims.at(3);
	const unsigned int target_tensor_size = target_channel * target_height * target_width;
	tensor_value_handler.resize(target_tensor_size);
	// 在GPU上执行toCHW操作
	float* dstDevData;
	cudaMalloc((void**)&dstDevData, sizeof(float) * target_tensor_size);
	unsigned char* srcDevData;
	cudaMalloc((void**)&srcDevData, sizeof(unsigned char) * target_tensor_size);
	cudaMemcpy(srcDevData, aImg, sizeof(unsigned char) * target_tensor_size, cudaMemcpyHostToDevice);
	dim3 blockSize(32, 32);
    	dim3 gridSize((target_width + blockSize.x - 1) / blockSize.x, (target_height + blockSize.y - 1) / blockSize.y);
	// hwc to chw / bgr to rgb
	process<<<gridSize, blockSize>>>(srcDevData, dstDevData, target_height, target_width);
	cudaDeviceSynchronize();  
	cudaMemcpy(tensor_value_handler.data(), dstDevData, sizeof(float) * target_tensor_size, cudaMemcpyDeviceToHost);
	
	cudaFree(srcDevData);
	cudaFree(dstDevData);

	return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
		target_tensor_size, tensor_dims.data(),
		tensor_dims.size());

}
