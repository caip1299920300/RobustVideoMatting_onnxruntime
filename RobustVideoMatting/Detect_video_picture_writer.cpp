#include "Detect_vedeo_picture_writer.h"
#include "Detect.h"

// ��Ƶ�ļ��Ķ�ȡ����ͼ�������ͼ�Ľ��
void detect_video_writer(string path_, float downsample_ratio, Ort::Session& session,
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

	// ���ļ�
	VideoCapture capture;
	capture.open(path_);
	if (!capture.isOpened()) {
		printf("could not read this video file...\n");
		return;
	}
	Size video_size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH),
		(int)capture.get(CAP_PROP_FRAME_HEIGHT));
	int fps = capture.get(CAP_PROP_FPS);

	//printf("current fps : %d \n", fps);
	//��ȡ�ļ���׺
	string out_path = path_.substr(0, path_.find_last_of('.'));
	out_path.append("_result");
	out_path.append(path_.substr(path_.find_last_of('.')));

	VideoWriter writer(out_path, CAP_OPENCV_MJPEG, fps, video_size, true);

	Mat frame;
	while (capture.read(frame)) {
		//frame = detect_human(frame, downsample_ratio, session,
		frame = detect(frame, downsample_ratio, session,
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
		// 4. check context states.
		//if (!context_is_update) break;
		writer.write(frame);
	}
	capture.release();
	writer.release();
}

// ͼƬ�ļ���ȡ����ͼ������
void detect_picture_writer(string path_, float downsample_ratio, Ort::Session& session,
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
	//��ȡ�ļ���׺
	string out_path = path_.substr(0, path_.find_last_of('.'));
	out_path.append("_result");
	out_path.append(path_.substr(path_.find_last_of('.')));
	cout << out_path << endl;

	Mat img = imread(path_);
	// �����ͼ
	//img = detect_human(img, 0.25f, session,
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
	imwrite(out_path,img);
}



// ��б��ת˫б�ܺ���
void Double2(string& s) {
	string::size_type pos = 0;
	while ((pos = s.find('\\', pos)) != string::npos) {
		s.insert(pos, "\\");
		pos = pos + 2;
	}
}

// ��ȡ�ļ�������·�����������ļ�����
void getFiles1(string path, vector<string>& files)
{
	//�ļ����  
	//long hFile = 0;  //win7
	intptr_t hFile = 0;   //win10
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		// "\\*"��ָ��ȡ�ļ����µ��������͵��ļ��������ȡ�ض����͵��ļ�����pngΪ�������á�\\*.png��
	{
		do
		{
			//�����Ŀ¼,����֮  
			//�������,�����б�  
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles1(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(path + "\\" + fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


// ����������ļ�������Ƶ��ͼƬ���п�ͼȻ���ٱ���
void Detect_video_picture_main(string input_filename, float downsample_ratio, Ort::Session& session,
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
	vector<string> files;

	//��ȡ��·���µ������ļ�·��  
	getFiles1(input_filename, files);

	const char* imgname[10] = { "jpg","jpeg","png","tiff" };
	const char* videoname[10] = { "avi","mp4",",mpeg","asf" };
	string name;

	// ����ļ�����û��ͼƬ������Ƶ�Ļ�����ֱ�������ú���
	if (files.size() == 0) {
		cout<<"�����ļ�û����Ƶ��ͼƬ!"<<endl;
		return;
	}

	// �������е�ͼƬ����Ƶ·��
	for (int i = 0; i < files.size(); i++)
	{
		name = files[i].c_str();

		//��ȡ�ļ���׺
		string str2 = name.substr(name.find_last_of('.') + 1);
		// cout << str2 << endl;
		
		// ���������б��ж���ͼƬ������Ƶ
		for (int j = 0; j < 4; j++)
		{
			if (str2 == imgname[j])
			{
				// ͼƬ��ͼ������
				detect_picture_writer(name, 0.25f, session,
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
			else if(str2==videoname[j]){
				// ��Ƶ��ͼ������
				detect_video_writer(name, 0.25f, session,
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

		}
	}

}