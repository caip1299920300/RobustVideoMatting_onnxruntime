#ifndef DETECT_H
#define DETECT_H

// ϵͳͷ�ļ�
#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>
// onnxruntime��Ҫ��ͷ�ļ�
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <onnxruntime_c_api.h>
// ��׼���

using namespace std;
using namespace Ort;

class RVM
{
private:
	// ����ڵ����
	unsigned int num_inputs = 6;
	// ����ڵ������
	std::vector<const char*> input_node_names = {
	 "src",
	 "r1i",
	 "r2i",
	 "r3i",
	 "r4i",
	 "downsample_ratio"
	};

	// ��ʼ����̬����dims
	std::vector<std::vector<int64_t>> dynamic_input_node_dims = {
	 {1, 3, 1280, 720}, // src  (b=1,c,h,w)
	 {1, 1, 1,    1}, // r1i
	 {1, 1, 1,    1}, // r2i
	 {1, 1, 1,    1}, // r3i
	 {1, 1, 1,    1}, // r4i
	 {1} // �²������� dsr
	};
	// ����Ľڵ����
	unsigned int num_outputs = 6;
	// ����ڵ������
	std::vector<const char*> output_node_names = {
	 "fgr",
	 "pha",
	 "r1o",
	 "r2o",
	 "r3o",
	 "r4o"
	};
	// ����ֵ���� & ��ʼ��
	std::vector<float> dynamic_src_value_handler;
	std::vector<float> dynamic_r1i_value_handler = { 0.0f }; // init 0. with shape (1,1,1,1)
	std::vector<float> dynamic_r2i_value_handler = { 0.0f };
	std::vector<float> dynamic_r3i_value_handler = { 0.0f };
	std::vector<float> dynamic_r4i_value_handler = { 0.0f };
	std::vector<float> dynamic_dsr_value_handler = { 0.25f }; // downsample_ratio with shape (1)

	// �Ƿ����ѭ�����������ġ���̬ά�ȸ���		
	bool context_is_update = false;
	// ����ΪVERBOSE���������̨���ʱ������ʹ����cpu����gpuִ��
	//Ort::Env env{ ORT_LOGGING_LEVEL_VERBOSE, "Onnxruntime" };
	//Ort::SessionOptions session_options;
	//Ort::Session session{ nullptr };
	
	Ort::Env env = Env(ORT_LOGGING_LEVEL_ERROR, "NVM");
	Ort::Session *session = nullptr;
	Ort::SessionOptions session_options = Ort::SessionOptions();

public:
	/* �ɵ�����-�²�������
		 �ֱ���	        ���� �� ȫ��
		<= 512x512	1	1
		1280x720	0.375	0.6
		1920x1080	0.25	0.4
		3840x2160	0.125	0.2
	*/
	float downsample_ratio = 0.375;
	/* ģ�������߳��� */
	int nThreadNum = 8;
	/* ����RGB��BGR��ͼƬ�����ض�Ӧ��RGB��BGR����ͼΪ��ɫ��ͼƬaResultImg */
	void detect(unsigned char* aImg,unsigned char* aResultImg, int nWeigh, int nHeight);

	/* ���캯�� */
	RVM(char* sModelPath);
	
};

#endif DETECT_H
