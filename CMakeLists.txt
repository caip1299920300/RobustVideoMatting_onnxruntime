cmake_minimum_required(VERSION 3.0)
project(NVM)

# 设置C++编译标准
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)   
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(LITE_ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.9.0)
set(LITE_ONNXRUNTIME_INCLUDE_DIR ${LITE_ONNXRUNTIME_DIR}/include)
set(LITE_ONNXRUNTIME_LIBRARY_DIR ${LITE_ONNXRUNTIME_DIR}/lib)
include_directories(${LITE_ONNXRUNTIME_INCLUDE_DIR})
link_directories(${LITE_ONNXRUNTIME_LIBRARY_DIR})

find_package(OpenCV REQUIRED COMPONENTS imgproc core highgui)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


#源文件
set (SRC_PATH
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/src/
)
foreach(item ${SRC_PATH})
    aux_source_directory (${item} SRC_LIST)
endforeach()
#头文件
set (STREAM_INCLUDE
    ${CMAKE_CURRENT_LIST_DIR}/include/
)
foreach(item ${STREAM_INCLUDE})
    include_directories ( ${item} ) 
endforeach()


add_compile_options(-Wall)

#添加库
set (LIBS pthread stdc++)
set (LOCAL_LIBS 
	onnxruntime
	onnxruntime_providers_cuda
 )
 
cuda_add_executable(NVM ${SRC_LIST})
target_link_libraries (NVM  ${LIBS} ${LOCAL_LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_LIBRARIES})
