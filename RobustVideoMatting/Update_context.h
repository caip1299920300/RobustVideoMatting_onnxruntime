#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>

#include "conio.h"

#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>

using namespace cv;     //当定义这一行后，cv::imread可以直接写成imread
using namespace std;
using namespace Ort;


// 
// @brief:	更新循环记忆上下文、动态维度更新			
// @params:
//	mat		输入的图片张量
//	dynamic_input_node_dims			输入张量的维度
//	dynamic_src_value_handler		各个的张量值				
//	dynamic_r1i_value_handler						
//	dynamic_r2i_value_handler					
//	dynamic_r3i_value_handler					
//	dynamic_r4i_value_handler					
//	dynamic_dsr_value_handler 	
//	context_is_update		        是否已经更新		
// @ret:void
// @birth:created by wucaipeng on 20220406
//
void update_context(std::vector<Ort::Value>& output_tensors,
	std::vector<std::vector<int64_t>>& dynamic_input_node_dims,
	std::vector<float>& dynamic_src_value_handler,
	std::vector<float>& dynamic_r1i_value_handler,
	std::vector<float>& dynamic_r2i_value_handler,
	std::vector<float>& dynamic_r3i_value_handler,
	std::vector<float>& dynamic_r4i_value_handler,
	bool& context_is_update
);