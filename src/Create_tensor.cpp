#include "Create_tensor.h"

// 创建张量
Ort::Value create_tensor(const cv::Mat& mat,
	const std::vector<int64_t>& tensor_dims,
	const Ort::MemoryInfo& memory_info_handler,
	std::vector<float>& tensor_value_handler,
	string data_format)
	throw(std::runtime_error)
{
	const unsigned int rows = mat.rows;
	const unsigned int cols = mat.cols;
	const unsigned int channels = mat.channels();

	cv::Mat mat_ref;
	if (mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
	else mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

	if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
	if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");

	// CXHXW
	if (data_format == "CHW")
	{
		const unsigned int target_channel = tensor_dims.at(1);
		const unsigned int target_height = tensor_dims.at(2);
		const unsigned int target_width = tensor_dims.at(3);
		const unsigned int target_tensor_size = target_channel * target_height * target_width;
		if (target_channel != channels) throw std::runtime_error("channel mismatch.");

		tensor_value_handler.resize(target_tensor_size);

		cv::Mat resize_mat_ref;
		if (target_height != rows || target_width != cols)
			cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
		else resize_mat_ref = mat_ref; // reference only. zero-time cost.

		std::vector<cv::Mat> mat_channels;
		cv::split(resize_mat_ref, mat_channels);
		// CXHXW
		for (unsigned int i = 0; i < channels; ++i)
			std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
				mat_channels.at(i).data, target_height * target_width * sizeof(float));

		return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
			target_tensor_size, tensor_dims.data(),
			tensor_dims.size());
	}

	// HXWXC
	const unsigned int target_channel = tensor_dims.at(3);
	const unsigned int target_height = tensor_dims.at(1);
	const unsigned int target_width = tensor_dims.at(2);
	const unsigned int target_tensor_size = target_channel * target_height * target_width;
	if (target_channel != channels) throw std::runtime_error("channel mismatch!");
	tensor_value_handler.resize(target_tensor_size);

	cv::Mat resize_mat_ref;
	if (target_height != rows || target_width != cols)
		cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
	else resize_mat_ref = mat_ref; // reference only. zero-time cost.

	std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, target_tensor_size * sizeof(float));

	return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
		target_tensor_size, tensor_dims.data(),
		tensor_dims.size());
}