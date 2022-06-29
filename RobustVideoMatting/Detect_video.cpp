
#include "Detect_video.h"
#include "Detect.h"

// ��������ͷ���߶�ȡ������Ƶ�ļ������п�ͼЧ��չʾ
void detect_video(float downsample_ratio, Ort::Session& session, string choose,
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
)
{

	double inferenceTime = 0.0;

	cv::VideoCapture video_capture;
	// ѡ����Ƶ��ַ��ͼ��������ͷ��ͼ
	if (choose == "movie") {
		cout << "��Ƶ��ַ��" << endl;
		cin >> choose;
		video_capture.open(choose);
	}
	else
	{
		cout << "�������������ͷ����ţ�" << endl;
		int num = 0;
		cin >> num;
		video_capture.open(num);
	}

	if (!video_capture.isOpened())
	{
		std::cout << "Can not open video" << "\n";
		return;
	}
	// 2. matting loop
	cv::Mat mat;

	while (video_capture.read(mat))
	{
		//��ʼʱ��
		double t1 = static_cast<double>(cv::getTickCount());

		cv::imshow("ԭͼ", mat);
		// �����ͼ
		mat = detect_human(mat, downsample_ratio, session,
			dynamic_input_node_dims,
			dynamic_src_value_handler,
			dynamic_r1i_value_handler,
			dynamic_r2i_value_handler,
			dynamic_r3i_value_handler,
			dynamic_r4i_value_handler,
			dynamic_dsr_value_handler,
			input_node_names,
			output_node_names,
			context_is_update,
			num_inputs,
			num_outputs
		);

		// �����ͼ��ʾmask
		/*mat = detect(mat,true, downsample_ratio, session,
			dynamic_input_node_dims,
			dynamic_src_value_handler,
			dynamic_r1i_value_handler,
			dynamic_r2i_value_handler,
			dynamic_r3i_value_handler,
			dynamic_r4i_value_handler,
			dynamic_dsr_value_handler,
			input_node_names,
			output_node_names,
			context_is_update,
			num_inputs,
			num_outputs
		);*/

		//����ʱ�䣬����fps
		double t2 = static_cast<double>(cv::getTickCount());
		inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
		std::stringstream fpsSs;
		fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
		// ��ÿ֡д��fps
		cv::putText(mat, fpsSs.str(), cv::Point(16, 32),
			cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));

		cv::imshow("��ͼ", mat);
		cv::waitKey(1);
		// 4. check context states.
		if (!context_is_update) break;

		if (_kbhit()) // ����а���������
		{
			if (_getch() == 'q') //���������q��������ѭ��
			{
				break;
			}

		}
	}
	// 5. release
	video_capture.release();
	destroyAllWindows();
}