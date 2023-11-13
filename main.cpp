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
	// 可调参数(Detect.h,有说明，这个值越小推理时间越快)
	demo.downsample_ratio = 0.25;
	// 启用推理的线程数
	demo.nThreadNum = 8;
	
	string img_path;
	cv::Mat img;
	unsigned char* aResultImg;    
	
	while(true)
	{
		img_path = "./test.jpg";
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
			cout << "succeed!" << endl;
		}
		catch (const std::exception&)
		{
			cout << "Input Img fail!" << endl;
		}
		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Main_Time：" << duration.count() << " ms" << std::endl;
		
	}

	delete[] aResultImg;
	return 0;
}

