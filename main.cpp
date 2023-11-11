#include "main.h" 
#include "Detect.h" 
// 主函数
int main(int argc, char* argv[])
{

	//设置为VERBOSE，方便控制台输出时看到是使用了cpu还是gpu执行
	Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "Onnxruntime");
	Ort::SessionOptions session_options;
	// 使用1个线程执行op,提升速度
	session_options.SetIntraOpNumThreads(8);
	// 设置图形优化级别
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	// 加载模型
	Ort::Session session(env, "model/model.onnx", session_options);
	std::cout << "onnxruntime loading onnx model..." << std::endl;
	Ort::AllocatorWithDefaultOptions allocator;
	

	/* ==================================== 菜单栏 ========================================*/ 
	int select; 
	string img_path;// 声明一个字符串，用于接受输入的图片或视频等
	Mat img;        // 声明一个图片矩阵
	while (1)
	{
		// 输入测试图片并显示
		std::cout << "input_ImgPath:" << endl;
		cin >> img_path;
		img = imread(img_path);
		try // 使用try，防止输入的图片地址为错误的情况
		{
			// 这里调用的是人像分割函数
			img = detect(img, 0.25f, session,
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
			cv::imshow("img", img);
			cv::waitKey(0);
			cout << "图片窗口已关闭!" << endl;
		}
		catch (const std::exception&)
		{
			cout << "图片输入地址有误!" << endl;
		}

	// =================配置初始化回原来的样子==========================
		// 输入值处理 & 初始化;
		dynamic_r1i_value_handler = { 0.0f }; // init 0. with shape (1,1,1,1)
		dynamic_r2i_value_handler = { 0.0f };
		dynamic_r3i_value_handler = { 0.0f };
		dynamic_r4i_value_handler = { 0.0f };
		dynamic_dsr_value_handler = { 0.25f }; // downsample_ratio with shape (1)
		// 初始化动态输入dims
		dynamic_input_node_dims = {
		 {1, 3, 1280, 720}, // src  (b=1,c,h,w)
		 {1, 1, 1,    1}, // r1i
		 {1, 1, 1,    1}, // r2i
		 {1, 1, 1,    1}, // r3i
		 {1, 1, 1,    1}, // r4i
		 {1} // 下采样比例 dsr
		}; // (1, 16, ?h, ?w) for inner loop rxi


		system("cls"); // 清空屏幕
	}
	
	return 0;
}

