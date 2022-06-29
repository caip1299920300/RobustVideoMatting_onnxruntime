#include "Detect_vedeo_picture_writer.h"
#include "Detect.h"

// 视频文件的读取、抠图并保存抠图的结果
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

	// 打开文件
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
	//获取文件后缀
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

// 图片文件读取、抠图并保存
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
	//获取文件后缀
	string out_path = path_.substr(0, path_.find_last_of('.'));
	out_path.append("_result");
	out_path.append(path_.substr(path_.find_last_of('.')));
	cout << out_path << endl;

	Mat img = imread(path_);
	// 人像抠图
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



// 单斜杠转双斜杠函数
void Double2(string& s) {
	string::size_type pos = 0;
	while ((pos = s.find('\\', pos)) != string::npos) {
		s.insert(pos, "\\");
		pos = pos + 2;
	}
}

// 获取文件下所有路径，包括子文件函数
void getFiles1(string path, vector<string>& files)
{
	//文件句柄  
	//long hFile = 0;  //win7
	intptr_t hFile = 0;   //win10
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		// "\\*"是指读取文件夹下的所有类型的文件，若想读取特定类型的文件，以png为例，则用“\\*.png”
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
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


// 遍历输入的文件，对视频和图片进行抠图然后再保存
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

	//获取该路径下的所有文件路径  
	getFiles1(input_filename, files);

	const char* imgname[10] = { "jpg","jpeg","png","tiff" };
	const char* videoname[10] = { "avi","mp4",",mpeg","asf" };
	string name;

	// 如果文件里面没有图片或者视频的话，则直接跳出该函数
	if (files.size() == 0) {
		cout<<"输入文件没有视频和图片!"<<endl;
		return;
	}

	// 遍历所有的图片和视频路径
	for (int i = 0; i < files.size(); i++)
	{
		name = files[i].c_str();

		//获取文件后缀
		string str2 = name.substr(name.find_last_of('.') + 1);
		// cout << str2 << endl;
		
		// 遍历所有列表，判断是图片还是视频
		for (int j = 0; j < 4; j++)
		{
			if (str2 == imgname[j])
			{
				// 图片抠图并保存
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
				// 视频抠图并保存
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