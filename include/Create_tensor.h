
#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <onnxruntime_c_api.h>

using namespace cv;     //当定义这一行后，cv::imread可以直接写成imread
using namespace std;
using namespace Ort;

// 
// @brief:创建张量
// @params:
//	input_filename		需要遍历文件的路径
//	downsample_ratio	下采样的比例
//	session				Ort::Session				
//	dynamic_input_node_dims		输入节点的dims
//	dynamic_src_value_handler	输入图片的值
//  r1i-r4i 对应的值
//	dynamic_r1i_value_handler
//	dynamic_r2i_value_handler
//	dynamic_r3i_value_handler
//	dynamic_r4i_value_handler
//	dynamic_dsr_value_handler
//	input_node_names			输入节点的名字
//	output_node_names			输出节点的名字
//	context_is_update			张量是否更新
//	num_inputs					输入节点数
//	num_outputs					输出节点数
// @ret:返回网络输入需要的张量
// @birth:created by wucaipeng on 20220406
//
Ort::Value create_tensor(const cv::Mat& mat,
	const std::vector<int64_t>& tensor_dims,
	const Ort::MemoryInfo& memory_info_handler,
	std::vector<float>& tensor_value_handler,
	string data_format)
	throw(std::runtime_error);
