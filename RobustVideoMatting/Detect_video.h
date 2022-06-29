#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>
#include "conio.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>

#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>

using namespace cv;     //当定义这一行后，cv::imread可以直接写成imread
using namespace std;
using namespace Ort;


//
// @brief:调用摄像头或者读取本地视频文件，进行抠图效果展示
// @params:
//	downsample_ratio	下采样的比例
//	session				Ort::Session			
//  choose						使用的是调用摄像头还是本地视频文件	
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
void detect_video(float downsample_ratio, Ort::Session& session, string choose,
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


// 调用摄像头或者读取本地视频文件，进行更换背景效果展示
void detect_background_video(const string background_path, float downsample_ratio, Ort::Session& session, string choose,
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