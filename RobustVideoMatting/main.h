#pragma once
//Circle_queue 头文件名
#ifndef main_H //就是头文件名（全大写后加个_H

#define main_H


// 系统头文件
#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>
#include<stdlib.h> // 菜单中的system用到
#include "conio.h"

// opencv需要的头文件
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>

// onnxruntime需要的头文件
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>

// 标准输出
using namespace cv;     //当定义这一行后，cv::imread可以直接写成imread
using namespace std;
using namespace Ort;


// 测试图片位置
string img_path = "image/test1.png";

// hardcode input node names
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

// init dynamic input dims
// 初始化动态输入dims
std::vector<std::vector<int64_t>> dynamic_input_node_dims = {
 {1, 3, 1280, 720}, // src  (b=1,c,h,w)
 {1, 1, 1,    1}, // r1i
 {1, 1, 1,    1}, // r2i
 {1, 1, 1,    1}, // r3i
 {1, 1, 1,    1}, // r4i
 {1} // 下采样比例 dsr
}; // (1, 16, ?h, ?w) for inner loop rxi


// hardcode output node names
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


#endif