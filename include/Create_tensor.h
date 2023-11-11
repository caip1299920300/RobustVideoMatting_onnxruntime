
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
// @brief:��������
// @params:
//	input_filename		��Ҫ�����ļ���·��
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
// @ret:��������������Ҫ������
// @birth:created by wucaipeng on 20220406
//
Ort::Value create_tensor(const cv::Mat& mat,
	const std::vector<int64_t>& tensor_dims,
	const Ort::MemoryInfo& memory_info_handler,
	std::vector<float>& tensor_value_handler,
	string data_format)
	throw(std::runtime_error);
