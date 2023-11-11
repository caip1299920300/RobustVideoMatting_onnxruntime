#include "Update_context.h"
#include "Value_size.h"

// ����ѭ�����������ġ���̬ά�ȸ��£�������LSTMЧ����
void update_context(std::vector<Ort::Value>& output_tensors,
	std::vector<std::vector<int64_t>>& dynamic_input_node_dims,
	std::vector<float>& dynamic_src_value_handler,
	std::vector<float>& dynamic_r1i_value_handler,
	std::vector<float>& dynamic_r2i_value_handler,
	std::vector<float>& dynamic_r3i_value_handler,
	std::vector<float>& dynamic_r4i_value_handler,
	bool& context_is_update
)
{
	// 0. update context for video matting.
	// 0. ����������ϵ��Ƶ����Ĵ�С
	Ort::Value& r1o = output_tensors.at(2); // fgr (1,?,?h,?w)
	Ort::Value& r2o = output_tensors.at(3); // pha (1,?,?h,?w)
	Ort::Value& r3o = output_tensors.at(4); // pha (1,?,?h,?w)
	Ort::Value& r4o = output_tensors.at(5); // pha (1,?,?h,?w)
	auto r1o_dims = r1o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto r2o_dims = r2o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto r3o_dims = r3o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto r4o_dims = r4o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	// 1. update rxi's shape according to last rxo
	// 1. ͨ������rxo������rxiϵ�е���״
	dynamic_input_node_dims.at(1) = r1o_dims;
	dynamic_input_node_dims.at(2) = r2o_dims;
	dynamic_input_node_dims.at(3) = r3o_dims;
	dynamic_input_node_dims.at(4) = r4o_dims;
	// 2. update rxi's value according to last rxo
	// 2. ͨ������rxo������rxiϵ�е�ֵ
	int64_t new_r1i_value_size = value_size_of(r1o_dims); // (1*?*?h*?w)
	int64_t new_r2i_value_size = value_size_of(r2o_dims); // (1*?*?h*?w)
	int64_t new_r3i_value_size = value_size_of(r3o_dims); // (1*?*?h*?w)
	int64_t new_r4i_value_size = value_size_of(r4o_dims); // (1*?*?h*?w)
	dynamic_r1i_value_handler.resize(new_r1i_value_size);
	dynamic_r2i_value_handler.resize(new_r2i_value_size);
	dynamic_r3i_value_handler.resize(new_r3i_value_size);
	dynamic_r4i_value_handler.resize(new_r4i_value_size);
	float* new_r1i_value_ptr = r1o.GetTensorMutableData<float>();
	float* new_r2i_value_ptr = r2o.GetTensorMutableData<float>();
	float* new_r3i_value_ptr = r3o.GetTensorMutableData<float>();
	float* new_r4i_value_ptr = r4o.GetTensorMutableData<float>();
	// ����һ��dynamic_rxi_value_handler������
	std::memcpy(dynamic_r1i_value_handler.data(), new_r1i_value_ptr, new_r1i_value_size * sizeof(float));
	std::memcpy(dynamic_r2i_value_handler.data(), new_r2i_value_ptr, new_r2i_value_size * sizeof(float));
	std::memcpy(dynamic_r3i_value_handler.data(), new_r3i_value_ptr, new_r3i_value_size * sizeof(float));
	std::memcpy(dynamic_r4i_value_handler.data(), new_r4i_value_ptr, new_r4i_value_size * sizeof(float));
	// �Ѿ�����
	context_is_update = true;
}