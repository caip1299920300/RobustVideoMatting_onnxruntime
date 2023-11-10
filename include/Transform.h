#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>

#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <onnxruntime_c_api.h>

using namespace std;
using namespace Ort;

// 
// @brief:	�������ͼƬת��Ϊģ��������Ҫ�ĸ�ʽ			
// @params:
//	mat		�����ͼƬ����
//	dynamic_input_node_dims			����������ά��
//	dynamic_src_value_handler		����������ֵ				
//	dynamic_r1i_value_handler						
//	dynamic_r2i_value_handler					
//	dynamic_r3i_value_handler					
//	dynamic_r4i_value_handler					
//	dynamic_dsr_value_handler 					
// @ret:ģ��������Ҫ������
// @birth:created by wucaipeng on 20220406
//
std::vector<Ort::Value> transform(unsigned char* aImg,
	int img_height,
	int img_width, 
	std::vector<std::vector<int64_t>>& dynamic_input_node_dims,
	std::vector<float>& dynamic_src_value_handler,
	std::vector<float>& dynamic_r1i_value_handler,
	std::vector<float>& dynamic_r2i_value_handler,
	std::vector<float>& dynamic_r3i_value_handler,
	std::vector<float>& dynamic_r4i_value_handler,
	std::vector<float>& dynamic_dsr_value_handler
);