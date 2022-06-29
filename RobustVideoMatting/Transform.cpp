#include "Transform.h"
#include "Value_size.h"
#include "Create_tensor.h"


//将输入的图片转换为模型输入需要的格式
std::vector<Ort::Value> transform(const cv::Mat& mat, 
	std::vector<std::vector<int64_t>>& dynamic_input_node_dims,
	std::vector<float> & dynamic_src_value_handler,
	std::vector<float>& dynamic_r1i_value_handler,
	std::vector<float>& dynamic_r2i_value_handler,
	std::vector<float>& dynamic_r3i_value_handler,
	std::vector<float>& dynamic_r4i_value_handler,
	std::vector<float>& dynamic_dsr_value_handler
	)
{
	cv::Mat src = mat.clone(); // 使用src复制mat矩阵
	const unsigned int img_height = mat.rows;
	const unsigned int img_width = mat.cols;
	std::vector<int64_t>& src_dims = dynamic_input_node_dims.at(0); // (1,3,h,w)
	// update src height and width
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

	// normalize & RGB 归一化
	cv::cvtColor(src, src, cv::COLOR_BGR2RGB); // (h,w,3)
	src.convertTo(src, CV_32FC3, 1.0f / 255.0f, 0.f); // 0.~1.

	Ort::AllocatorWithDefaultOptions allocator;
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault);


	// 声明一个输入的张量的容器
	std::vector<Ort::Value> input_tensors;

	// 在input_tensors容器后面，分别放入以下几个张量
	input_tensors.emplace_back(create_tensor(
		src, src_dims, memory_info_handler, dynamic_src_value_handler, "CHW"));


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