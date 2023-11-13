#ifndef DETECT_H
#define DETECT_H

// 系统头文件
#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>
// onnxruntime需要的头文件
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <onnxruntime_c_api.h>
// 标准输出

using namespace std;
using namespace Ort;

class RVM
{
private:
	// 输入节点个数
	unsigned int num_inputs = 6;
	// 输入节点的名字
	std::vector<const char*> input_node_names = {
	 "src",
	 "r1i",
	 "r2i",
	 "r3i",
	 "r4i",
	 "downsample_ratio"
	};

	// 初始化动态输入dims
	std::vector<std::vector<int64_t>> dynamic_input_node_dims = {
	 {1, 3, 1280, 720}, // src  (b=1,c,h,w)
	 {1, 1, 1,    1}, // r1i
	 {1, 1, 1,    1}, // r2i
	 {1, 1, 1,    1}, // r3i
	 {1, 1, 1,    1}, // r4i
	 {1} // 下采样比例 dsr
	};
	// 输出的节点个数
	unsigned int num_outputs = 6;
	// 输出节点的名字
	std::vector<const char*> output_node_names = {
	 "fgr",
	 "pha",
	 "r1o",
	 "r2o",
	 "r3o",
	 "r4o"
	};
	// 输入值处理 & 初始化
	std::vector<float> dynamic_src_value_handler;
	std::vector<float> dynamic_r1i_value_handler = { 0.0f }; // init 0. with shape (1,1,1,1)
	std::vector<float> dynamic_r2i_value_handler = { 0.0f };
	std::vector<float> dynamic_r3i_value_handler = { 0.0f };
	std::vector<float> dynamic_r4i_value_handler = { 0.0f };
	std::vector<float> dynamic_dsr_value_handler = { 0.25f }; // downsample_ratio with shape (1)

	// 是否更新循环记忆上下文、动态维度更新		
	bool context_is_update = false;
	// 设置为VERBOSE，方便控制台输出时看到是使用了cpu还是gpu执行
	//Ort::Env env{ ORT_LOGGING_LEVEL_VERBOSE, "Onnxruntime" };
	//Ort::SessionOptions session_options;
	//Ort::Session session{ nullptr };
	
	Ort::Env env = Env(ORT_LOGGING_LEVEL_ERROR, "NVM");
	Ort::Session *session = nullptr;
	Ort::SessionOptions session_options = Ort::SessionOptions();

public:
	/* 可调参数-下采样比率
		 分辨率	        人像 或 全身
		<= 512x512	1	1
		1280x720	0.375	0.6
		1920x1080	0.25	0.4
		3840x2160	0.125	0.2
	*/
	float downsample_ratio = 0.375;
	/* 模型推理线程数 */
	int nThreadNum = 8;
	/* 输入RGB或BGR的图片，返回对应的RGB或BGR背景图为黑色的图片aResultImg */
	void detect(unsigned char* aImg,unsigned char* aResultImg, int nWeigh, int nHeight);

	/* 构造函数 */
	RVM(char* sModelPath);
	
};

#endif DETECT_H
