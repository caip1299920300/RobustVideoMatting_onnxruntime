#include "Transform.h"
#include "Value_size.h"
#include "Create_tensor.cuh"

//将输入的图片转换为模型输入需要的格式
std::vector<Ort::Value> transform(unsigned char* aImg,
	int img_height,
	int img_width, 
	std::vector<std::vector<int64_t>>& dynamic_input_node_dims,
	std::vector<float> & dynamic_src_value_handler,
	std::vector<float>& dynamic_r1i_value_handler,
	std::vector<float>& dynamic_r2i_value_handler,
	std::vector<float>& dynamic_r3i_value_handler,
	std::vector<float>& dynamic_r4i_value_handler,
	std::vector<float>& dynamic_dsr_value_handler
	)
{
	std::vector<int64_t>& src_dims = dynamic_input_node_dims.at(0); // (1,3,h,w)
	// 更新src的宽和高
	src_dims.at(2) = img_height;
	src_dims.at(3) = img_width;
	// assume that rxi's dims and value_handler was updated by last step in a while loop.
	std::vector<int64_t>& r1i_dims = dynamic_input_node_dims.at(1); // (1,?,?h,?w)
	std::vector<int64_t>& r2i_dims = dynamic_input_node_dims.at(2); // (1,?,?h,?w)
	std::vector<int64_t>& r3i_dims = dynamic_input_node_dims.at(3); // (1,?,?h,?w)
	std::vector<int64_t>& r4i_dims = dynamic_input_node_dims.at(4); // (1,?,?h,?w)
	std::vector<int64_t>& dsr_dims = dynamic_input_node_dims.at(5); // (1)
	int64_t src_value_size = value_size_of(src_dims); // (1*3*h*w)
	int64_t r1i_value_size = value_size_of(r1i_dims); // (1*?*?h*?w)
	int64_t r2i_value_size = value_size_of(r2i_dims); // (1*?*?h*?w)
	int64_t r3i_value_size = value_size_of(r3i_dims); // (1*?*?h*?w)
	int64_t r4i_value_size = value_size_of(r4i_dims); // (1*?*?h*?w)
	int64_t dsr_value_size = value_size_of(dsr_dims); // 1
	dynamic_src_value_handler.resize(src_value_size);
	

	Ort::AllocatorWithDefaultOptions allocator;
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault);
	// 声明一个输入的张量的容器
	std::vector<Ort::Value> input_tensors;

	input_tensors.emplace_back(create_tensor(
		aImg, src_dims, memory_info_handler, dynamic_src_value_handler));
	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, dynamic_r1i_value_handler.data(),
		r1i_value_size, r1i_dims.data(), r1i_dims.size()
		));

	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, dynamic_r2i_value_handler.data(),
		r2i_value_size, r2i_dims.data(), r2i_dims.size()
		));
	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, dynamic_r3i_value_handler.data(),
		r3i_value_size, r3i_dims.data(), r3i_dims.size()
		));
	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, dynamic_r4i_value_handler.data(),
		r4i_value_size, r4i_dims.data(), r4i_dims.size()
		));
	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, dynamic_dsr_value_handler.data(),
		dsr_value_size, dsr_dims.data(), dsr_dims.size()
		));


	return input_tensors;
}
