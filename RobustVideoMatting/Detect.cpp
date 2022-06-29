#include "Detect.h"
#include "Transform.h"
#include "Update_context.h"

// 抠图片中的人像返回mask
cv::Mat detect(const cv::Mat& img, float downsample_ratio, Ort::Session& session,
	std::vector<std::vector<int64_t>>& dynamic_input_node_dims,
	std::vector<float>& dynamic_src_value_handler,
	std::vector<float>& dynamic_r1i_value_handler,
	std::vector<float>& dynamic_r2i_value_handler,
	std::vector<float>& dynamic_r3i_value_handler,
	std::vector<float>& dynamic_r4i_value_handler, 
	std::vector<float>& dynamic_dsr_value_handler,
	std::vector<const char*>& input_node_names,
	std::vector<const char*>& output_node_names,
	bool& context_is_update,
	unsigned int& num_inputs,
	unsigned int& num_outputs
	)
{
	// 在运行时设置dsr
	dynamic_dsr_value_handler.at(0) = downsample_ratio;

	// make input tensors, src, rxi, dsr
	// 图像预处理，输入张量src, rxi, dsr
	std::vector<Ort::Value> input_tensors = transform(img,
		dynamic_input_node_dims,
		dynamic_src_value_handler,
		dynamic_r1i_value_handler,
		dynamic_r2i_value_handler,
		dynamic_r3i_value_handler,
		dynamic_r4i_value_handler,
		dynamic_dsr_value_handler
	);


	// 前向推理
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
		input_node_names.data(),
		input_tensors.data(), num_inputs, output_node_names.data(),
		num_outputs
	);
	// ===================================== generate_matting ============================================
	// 推理的结果解码
	Ort::Value& fgr = output_tensors.at(0); // fgr (1,3,h,w) 0.~1.
	Ort::Value& pha = output_tensors.at(1); // pha (1,1,h,w) 0.~1.
	// 自动获取维度数量
	auto fgr_dims = fgr.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto pha_dims = pha.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	const unsigned int height = fgr_dims.at(2); // output height
	const unsigned int width = fgr_dims.at(3); // output width
	//const unsigned int channel_step = height * width;
	float* pha_ptr = pha.GetTensorMutableData<float>();
	cv::Mat pmat(height, width, CV_32FC1, pha_ptr); // ref only, zero copies.
	pmat *= 255.;

	pmat.convertTo(pmat, CV_8UC1);

	return pmat;
}