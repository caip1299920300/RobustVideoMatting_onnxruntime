#include "Detect.h" 
#include <chrono>


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

// 主函数
int main(int argc, char* argv[])
{
	/* 1、初始化抠图类的对象 */
	RVM demo("model/model.onnx");
	// 可调参数
	
	string img_path;// 声明一个字符串，用于接受输入的图片或视频等
	cv::Mat img;
	unsigned char* aResultImg;        // 声明一个图片矩阵
	while (1)
	{
		// 输入测试图片并显示
		std::cout << "input_ImgPath:" << endl;
		cin >> img_path;
		img = cv::imread(img_path);
		//cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		
		auto start = std::chrono::steady_clock::now();
		try // 使用try，防止输入的图片地址为错误的情况
		{
			int height = img.rows;
			int width = img.cols;
			aResultImg = new unsigned char[height*width*3];
			
			/* 抠图函数调用 */
			demo.detect(img.data,aResultImg,width,height);
			
			cv::Mat pmat(height, width, CV_8UC3, aResultImg); 
			cv::imwrite("result.jpg",pmat);
			cout << "succeed" << endl;
		}
		catch (const std::exception&)
		{
			cout << "图片输入地址有误!" << endl;
		}
		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "函数调用耗时：" << duration.count() << "毫秒" << std::endl;
		
		delete[] aResultImg;

	}
	return 0;
}

