#include "Value_size.h"

int64_t value_size_of(const std::vector<int64_t>& dims)
{
	if (dims.empty()) return 0;
	int64_t value_size = 1;
	for (const auto& size : dims) value_size *= size;
	return value_size;
}