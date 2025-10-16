cmake_minimum_required(VERSION 3.22)

project(benchsp CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include) 
include_directories(/usr/local/cuda-12.4/include) 

include(FetchContent)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_FLAGS " -O3 -Wall")
set(NVCC_FLAGS  -O3 -ccbin /usr/bin/gcc -m64 -gencode arch=compute_80,code=sm_80)


FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

set(CMAKE_CUDA_ARCHITECTURES "80")
file(GLOB_RECURSE mains RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp")
file(GLOB_RECURSE SRC_DIR_LIST "src/*.cpp" "src/*.cu" "src/*.c")	#遍历获取source/路径下所有的*.cpp和*.c文件列表

foreach(mainfile IN LISTS mains)
    # Get file name without directory
    get_filename_component(mainname ${mainfile} NAME_WE)
    # add_executable( ${mainfile})
    add_executable(${mainname}
    ./utils/utils.hpp
    ${mainfile}       # C++ 源文件
    ${SRC_DIR_LIST}
    )
    target_link_libraries(${mainname} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY}  fast_matrix_market::fast_matrix_market)
    target_compile_options(${mainname} PRIVATE -O3  $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)

endforeach()
# 创建可执行文件，并添加所有源文件



