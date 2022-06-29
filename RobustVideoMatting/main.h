#pragma once
//Circle_queue ͷ�ļ���
#ifndef main_H //����ͷ�ļ�����ȫ��д��Ӹ�_H

#define main_H


// ϵͳͷ�ļ�
#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>
#include<stdlib.h> // �˵��е�system�õ�
#include "conio.h"

// opencv��Ҫ��ͷ�ļ�
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>

// onnxruntime��Ҫ��ͷ�ļ�
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>

// ��׼���
using namespace cv;     //��������һ�к�cv::imread����ֱ��д��imread
using namespace std;
using namespace Ort;


// ����ͼƬλ��
string img_path = "image/test1.png";

// hardcode input node names
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

// init dynamic input dims
// ��ʼ����̬����dims
std::vector<std::vector<int64_t>> dynamic_input_node_dims = {
 {1, 3, 1280, 720}, // src  (b=1,c,h,w)
 {1, 1, 1,    1}, // r1i
 {1, 1, 1,    1}, // r2i
 {1, 1, 1,    1}, // r3i
 {1, 1, 1,    1}, // r4i
 {1} // �²������� dsr
}; // (1, 16, ?h, ?w) for inner loop rxi


// hardcode output node names
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


#endif