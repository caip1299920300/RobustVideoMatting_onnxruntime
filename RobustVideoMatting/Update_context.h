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

using namespace cv;     //��������һ�к�cv::imread����ֱ��д��imread
using namespace std;
using namespace Ort;


// 
// @brief:	����ѭ�����������ġ���̬ά�ȸ���			
// @params:
//	mat		�����ͼƬ����
//	dynamic_input_node_dims			����������ά��
//	dynamic_src_value_handler		����������ֵ				
//	dynamic_r1i_value_handler						
//	dynamic_r2i_value_handler					
//	dynamic_r3i_value_handler					
//	dynamic_r4i_value_handler					
//	dynamic_dsr_value_handler 	
//	context_is_update		        �Ƿ��Ѿ�����		
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