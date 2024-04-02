cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(Emile LANGUAGES CXX CUDA)


set(PROJECT_NAME Emile)
project(${PROJECT_NAME})

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBDPDK REQUIRED libdpdk)

# 指定源文件
set(SOURCE_FILES ./src/kernel.cu  ./src/main.cpp  ./src/pipeline.cpp  ./src/utils.cpp)

find_package(CUDA 12.2 REQUIRED)
# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# 添加 DPDK 的编译选项
link_directories(/usr/local/cuda-12.2/lib64)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBDPDK_INCLUDE_DIRS} ./include  /usr/local/cuda-12.2/include)
target_compile_options(${PROJECT_NAME} PRIVATE ${LIBDPDK_CFLAGS_OTHER})

# 链接 DPDK 库
target_link_libraries(${PROJECT_NAME} ${LIBDPDK_LIBRARIES} /usr/local/cuda-12.2/lib64/libcudart.so)
