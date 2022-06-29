#include "Detect.h"
#include "Transform.h"
#include "Update_context.h"

// 切换背景
cv::Mat detect_Change_background(const cv::Mat& img, const string background_path, float downsample_ratio, Ort::Session& session,
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

	// 获取推理的结果
	Ort::Value& fgr = output_tensors.at(0); // fgr (1,3,h,w) 0.~1.
	Ort::Value& pha = output_tensors.at(1); // pha (1,1,h,w) 0.~1.
	// 自动获取维度数量
	auto fgr_dims = fgr.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto pha_dims = pha.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	const unsigned int height = fgr_dims.at(2); // output height
	const unsigned int width = fgr_dims.at(3); // output width
	const unsigned int channel_step = height * width;
	// cv::merge -> assign & channel transpose(CHW->HWC).
	// 张量维度换位
	float* fgr_ptr = fgr.GetTensorMutableData<float>();
	float* pha_ptr = pha.GetTensorMutableData<float>();
	cv::Mat rmat(height, width, CV_32FC1, fgr_ptr); // ref only, zero copies.
	cv::Mat gmat(height, width, CV_32FC1, fgr_ptr + channel_step); // ref only, zero copies.
	cv::Mat bmat(height, width, CV_32FC1, fgr_ptr + 2 * channel_step); // ref only, zero copies.
	cv::Mat pmat(height, width, CV_32FC1, pha_ptr); // ref only, zero copies.
	rmat *= 255.;
	bmat *= 255.;
	gmat *= 255.;

	// 换背景
	cv::Mat rest = 1. - pmat;
	
	/*Mat test = cv::imread("C:/Users/95160/Desktop/feng.jpg");
	cv::resize(background, background, cv::Size(width, height));
	background.convertTo(background, CV_32F);
	vector<Mat>mv;
	split(background, mv);
	cv::Mat mbmat = mv[0].mul(rest) + bmat.mul(pmat);
	cv::Mat mgmat = mv[1].mul(rest) + gmat.mul(pmat);
	cv::Mat mrmat = mv[2].mul(rest) + rmat.mul(pmat);*/

	Mat background = cv::imread(background_path);
	cv::resize(background, background, cv::Size(width, height));
	background.convertTo(background, CV_32F);
	vector<Mat>mv;
	split(background, mv);
	cv::Mat mbmat = mv[0].mul(rest) + bmat.mul(pmat);
	cv::Mat mgmat = mv[1].mul(rest) + gmat.mul(pmat);
	cv::Mat mrmat = mv[2].mul(rest) + rmat.mul(pmat);

	// 把三个通道的变量放入vector容器中
	std::vector<cv::Mat> merge_channel_mats;
	merge_channel_mats.push_back(mbmat);
	merge_channel_mats.push_back(mgmat);
	merge_channel_mats.push_back(mrmat);

	// 创建空画布
	cv::Mat merge_mat;
	// 通道融合
	cv::merge(merge_channel_mats, merge_mat);

	// 图片深度转换
	merge_mat.convertTo(merge_mat, CV_8UC3);

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

	return merge_mat;
}