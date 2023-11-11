#pragma once
#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <onnxruntime_c_api.h>

using namespace cv;     //��������һ�к�cv::imread����ֱ��д��imread
using namespace std;
using namespace Ort;




//
// @brief:��ͼƬ�е����񷵻�mask
// @params:
//	img					�����ͼƬ����
//	downsample_ratio	�²����ı���
//	session				Ort::Session				
//	dynamic_input_node_dims		����ڵ��dims
//	dynamic_src_value_handler	����ͼƬ��ֵ
//  r1i-r4i ��Ӧ��ֵ
//	dynamic_r1i_value_handler
//	dynamic_r2i_value_handler
//	dynamic_r3i_value_handler
//	dynamic_r4i_value_handler
//	dynamic_dsr_value_handler
//	input_node_names			����ڵ������
//	output_node_names			����ڵ������
//	context_is_update			�����Ƿ����
//	num_inputs					����ڵ���
//	num_outputs					����ڵ���
// @ret:mask����
// @birth:created by wucaipeng on 20220406
//
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
);



//
// @brief:��ͼͼƬ�е�����
// @params:
//	img					�����ͼƬ����
//	downsample_ratio	�²����ı���
//	session				Ort::Session				
//	dynamic_input_node_dims		����ڵ��dims
//	dynamic_src_value_handler	����ͼƬ��ֵ
//  r1i-r4i ��Ӧ��ֵ
//	dynamic_r1i_value_handler
//	dynamic_r2i_value_handler
//	dynamic_r3i_value_handler
//	dynamic_r4i_value_handler
//	dynamic_dsr_value_handler
//	input_node_names			����ڵ������
//	output_node_names			����ڵ������
//	context_is_update			�����Ƿ����
//	num_inputs					����ڵ���
//	num_outputs					����ڵ���
// @ret:�����mat����
// @birth:created by wucaipeng on 20220406
//
cv::Mat detect_human(const cv::Mat& img, float downsample_ratio, Ort::Session& session,
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


// �л�����
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
);
