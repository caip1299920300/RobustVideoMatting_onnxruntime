#include "Detect.h" 
#include <chrono>


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

// ������
int main(int argc, char* argv[])
{
	/* 1����ʼ����ͼ��Ķ��� */
	RVM demo("model/model.onnx");
	// �ɵ�����
	
	string img_path;// ����һ���ַ��������ڽ��������ͼƬ����Ƶ��
	cv::Mat img;
	unsigned char* aResultImg;        // ����һ��ͼƬ����
	while (1)
	{
		// �������ͼƬ����ʾ
		std::cout << "input_ImgPath:" << endl;
		cin >> img_path;
		img = cv::imread(img_path);
		//cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		
		auto start = std::chrono::steady_clock::now();
		try // ʹ��try����ֹ�����ͼƬ��ַΪ��������
		{
			int height = img.rows;
			int width = img.cols;
			aResultImg = new unsigned char[height*width*3];
			
			/* ��ͼ�������� */
			demo.detect(img.data,aResultImg,width,height);
			
			cv::Mat pmat(height, width, CV_8UC3, aResultImg); 
			cv::imwrite("result.jpg",pmat);
			cout << "succeed" << endl;
		}
		catch (const std::exception&)
		{
			cout << "ͼƬ�����ַ����!" << endl;
		}
		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "�������ú�ʱ��" << duration.count() << "����" << std::endl;
		
		delete[] aResultImg;

	}
	return 0;
}

