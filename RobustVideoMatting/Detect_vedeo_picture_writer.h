#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include<io.h>
#include<string>

#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


//
// @brief:遍历输入的文件，对视频和图片进行抠图然后再保存
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
// @ret:void
// @birth:created by wucaipeng on 20220406
//
void Detect_video_picture_main(string input_filename, float downsample_ratio, Ort::Session& session,
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
);

