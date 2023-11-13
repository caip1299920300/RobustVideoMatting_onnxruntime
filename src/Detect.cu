#include "Detect.h"
#include "Transform.h"
#include "Update_context.h"

__global__ void process(unsigned char* srcData, const float* tgtData, const int h, const int w)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * w;
    int idx3 = idx * 3;

    if (ix < w && iy < h)
    {
        srcData[idx3 + 0] *=  tgtData[idx];
        srcData[idx3 + 1] *= tgtData[idx];
        srcData[idx3 + 2] *=  tgtData[idx];
    }
}

RVM::RVM(char* sModelPath) //: session(env, sModelPath, session_options) 
{
	// 添加CUDAExecutionProvider到会话选项中
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));   //CUDA加速开启
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC); //设置图优化类型
	session_options.SetIntraOpNumThreads(nThreadNum); // 设置线程
	session = new Ort::Session(env, sModelPath, session_options);    // 创建会话，把模型加载到内存
	Ort::AllocatorWithDefaultOptions allocator;
	
}


void RVM::detect(unsigned char* aImg,unsigned char* aResultImg, int nWeigh, int nHeight)
{
	// 在运行时设置dsr
	dynamic_dsr_value_handler.at(0) = downsample_ratio;

	// 图像预处理，输入张量src, rxi, dsr
	std::vector<Ort::Value> input_tensors = transform(aImg,
		nHeight,
		nWeigh,
		dynamic_input_node_dims,
		dynamic_src_value_handler,
		dynamic_r1i_value_handler,
		dynamic_r2i_value_handler,
		dynamic_r3i_value_handler,
		dynamic_r4i_value_handler,
		dynamic_dsr_value_handler
	);
	// 前向推理
	auto output_tensors = session->Run(Ort::RunOptions{ nullptr },
		input_node_names.data(),
		input_tensors.data(), num_inputs, output_node_names.data(),
		num_outputs
	);
	
	// 推理的结果解码
	Ort::Value& fgr = output_tensors.at(0); // fgr (1,3,h,w) 0.~1.
	Ort::Value& pha = output_tensors.at(1); // pha (1,1,h,w) 0.~1.
	// 自动获取维度数量
	auto fgr_dims = fgr.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto pha_dims = pha.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	const unsigned int height = fgr_dims.at(2); // output height
	const unsigned int width = fgr_dims.at(3); // output width
	const unsigned int target_tensor_size = height * width;
	
	unsigned char* imgResult = new unsigned char[height * width*3];
	
	
	
	// 在GPU上执行背景黑化操作
	float* dstDevData;
	cudaMalloc((void**)&dstDevData, sizeof(float) * target_tensor_size);
	cudaMemcpy(dstDevData, pha.GetTensorData<float>(), sizeof(float) * target_tensor_size, cudaMemcpyHostToDevice);
	unsigned char* srcDevData;
	cudaMalloc((void**)&srcDevData, sizeof(unsigned char) * target_tensor_size*3);
	cudaMemcpy(srcDevData, aImg, sizeof(unsigned char) * target_tensor_size*3, cudaMemcpyHostToDevice);
	dim3 blockSize(32, 32);
    	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	// hwc to chw / bgr to rgb
	process<<<gridSize, blockSize>>>(srcDevData, dstDevData, height, width);
	cudaDeviceSynchronize();  
	cudaMemcpy(imgResult, srcDevData, sizeof(unsigned char) * target_tensor_size*3, cudaMemcpyDeviceToHost);
	
	cudaFree(srcDevData);
	cudaFree(dstDevData);
	
	// 计算输出张量的大小
	size_t nimg_size = sizeof(unsigned char) * height * width*3;
	std::memcpy(aResultImg, imgResult, nimg_size);
	


	// 更新循环记忆上下文、动态维度更新
	update_context(output_tensors,
		dynamic_input_node_dims,
		dynamic_src_value_handler,
		dynamic_r1i_value_handler,
		dynamic_r2i_value_handler,
		dynamic_r3i_value_handler,
		dynamic_r4i_value_handler,
		context_is_update
	);
}
