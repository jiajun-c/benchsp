cmake_minimum_required(VERSION 3.22)

project(benchsp CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include) 
include(FetchContent)

find_package(CUDAToolkit REQUIRED)

FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

set(CMAKE_CUDA_ARCHITECTURES "80")

# 创建可执行文件，并添加所有源文件
add_executable(benchsp
    main.cpp       # C++ 源文件
    amgT.cu        # CUDA 源文件
    ref.cpp
    cusparse.cu    # CUDA 源文件
)

target_link_libraries(benchsp ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY}  fast_matrix_market::fast_matrix_market)

