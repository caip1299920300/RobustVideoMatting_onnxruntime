
#include "main.h" 
#include "Detect.h" 
#include "Detect_video.h"
#include "Detect_vedeo_picture_writer.h"

// ����˵�����
void showmenu();

// ������
int main(int argc, char* argv[])
{

	//����ΪVERBOSE���������̨���ʱ������ʹ����cpu����gpuִ��
	Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "Onnxruntime");
	Ort::SessionOptions session_options;
	// ʹ��1���߳�ִ��op,�����ٶ�
	session_options.SetIntraOpNumThreads(8);
	// ����ͼ���Ż�����
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	// ����ģ��
	Ort::Session session(env, L"model/model.onnx", session_options);
	std::cout << "onnxruntime loading onnx model..." << std::endl;
	Ort::AllocatorWithDefaultOptions allocator;
	

	/* ==================================== �˵��� ========================================*/ 
	int select; 
	string img_path;// ����һ���ַ��������ڽ��������ͼƬ����Ƶ��
	Mat img;        // ����һ��ͼƬ����
	while (1)
	{
		showmenu(); // ���ò˵���ʾ
		cin >> select;
		// �������Ĳ���ѡ���ڣ���Ĭ��ѡ��1
		if (!(select < 6 && select>=0))select = 1;
		
		switch (select)
		{
			case 1: 
				cout << "����������ָ��ļ��У�" << endl;
				cin >> img_path;
				// ���������ļ��зָ��
				Detect_video_picture_main(img_path, 0.25f, session,
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
				cout << "�ļ�����ָ���Ӧ�ļ��Ѿ�����!" << endl;
				system("pause");
				break;
			case 2:
				// �������ͼƬ����ʾ
				cout << "������ͼƬ��ַ:" << endl;
				cin >> img_path;
				img = imread(img_path);
				try // ʹ��try����ֹ�����ͼƬ��ַΪ��������
				{
					// ������õ�������ָ��
					img = detect_human(img, 0.25f, session,
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
					cout << "ͼƬ�����ѹر�!" << endl;
				}
				catch (const std::exception&)
				{
					cout << "ͼƬ�����ַ����!" << endl;
				}
				system("pause");
				break;
			case 3:
				// ��Ƶ��ͼ��������ʾ
				detect_video(0.25f, session, "movie",
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
				cout << "��Ƶչʾ����!" << endl;
				system("pause");
				break;
			case 4:
				// ����ͷ��ͼ
				detect_video(0.25f, session, "camera",
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
				cout << "����ͷ�ѹر�!" << endl;
				system("pause");
				break;

			case 5:
				// ���뱳��ͼƬ����ʾ
				cout << "�����뱳��ͼƬ��ַ:" << endl;
				cin >> img_path;
				if (imread(img_path).empty()) {
					cout << "����ͼƬ��ַ����!" << endl;
					break;
				}
				try { // ʹ��try����ֹ�����ͼƬ��ַΪ��������
					// ����ͷ����
					detect_background_video(img_path, 0.25f, session, "camera",
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
				}
				catch (const std::exception&)
				{
					cout << "ͼƬ�����ַ����!" << endl;
				}
				cv::destroyAllWindows();
				cout << "����ͷ�ѹر�!" << endl;
				system("pause");
				break;
			case 0:
				exit(0);
		}

	// =================���ó�ʼ����ԭ��������==========================
		// ����ֵ���� & ��ʼ��;
		dynamic_r1i_value_handler = { 0.0f }; // init 0. with shape (1,1,1,1)
		dynamic_r2i_value_handler = { 0.0f };
		dynamic_r3i_value_handler = { 0.0f };
		dynamic_r4i_value_handler = { 0.0f };
		dynamic_dsr_value_handler = { 0.25f }; // downsample_ratio with shape (1)
		// ��ʼ����̬����dims
		dynamic_input_node_dims = {
		 {1, 3, 1280, 720}, // src  (b=1,c,h,w)
		 {1, 1, 1,    1}, // r1i
		 {1, 1, 1,    1}, // r2i
		 {1, 1, 1,    1}, // r3i
		 {1, 1, 1,    1}, // r4i
		 {1} // �²������� dsr
		}; // (1, 16, ?h, ?w) for inner loop rxi


		system("cls"); // �����Ļ
	}
	return 0;
}


void showmenu()
{
	printf("----�˵�----\n");
	printf("1-�����ļ��п�ͼ�������Ӧ��ͼ�ļ���Ĭ��ѡ�\n");
	printf("2-ͼƬ��ͼչʾ�������棩\n");
	printf("3-��Ƶ��ͼչʾ�������棩\n");
	printf("4-����ͷ��ͼչʾ�������棩\n");
	printf("5-����ͷ������չʾ�������棩\n");
	printf("0-�˳�����\n");
}

