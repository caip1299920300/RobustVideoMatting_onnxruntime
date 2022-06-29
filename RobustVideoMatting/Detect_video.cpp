
#include "Detect_video.h"
#include "Detect.h"

// 调用摄像头或者读取本地视频文件，进行抠图效果展示
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
	// 选择视频地址抠图或者摄像头抠图
	if (choose == "movie") {
		cout << "视频地址：" << endl;
		cin >> choose;
		video_capture.open(choose);
	}
	else
	{
		cout << "请输入调用摄像头的序号：" << endl;
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
		//开始时间
		double t1 = static_cast<double>(cv::getTickCount());

		cv::imshow("原图", mat);
		// 人像抠图
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

		// 人像抠图显示mask
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

		//结束时间，计算fps
		double t2 = static_cast<double>(cv::getTickCount());
		inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
		std::stringstream fpsSs;
		fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
		// 向每帧写入fps
		cv::putText(mat, fpsSs.str(), cv::Point(16, 32),
			cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));

		cv::imshow("抠图", mat);
		cv::waitKey(1);
		// 4. check context states.
		if (!context_is_update) break;

		if (_kbhit()) // 如果有按键被按下
		{
			if (_getch() == 'q') //如果按下了q键则跳出循环
			{
				break;
			}

		}
	}
	// 5. release
	video_capture.release();
	destroyAllWindows();
}