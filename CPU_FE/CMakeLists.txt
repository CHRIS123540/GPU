cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME basicfwd)
project(${PROJECT_NAME}  LANGUAGES C CUDA)
# 寻找 PkgConfig 并检查 DPDK 包
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBDPDK REQUIRED libdpdk)

# 定义 CUDA 和 TensorRT 的路径
set(CUDA_PATH "/usr/local/cuda")
#set(TENSORRT_PATH "/home/hlx/packet/TensorRT-8.5.1.7")

# 指定源文件
set(SOURCE_FILES basicfwd.c)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# 包含 DPDK, CUDA, 和 TensorRT 的头文件目录
target_include_directories(${PROJECT_NAME} PRIVATE ${LIBDPDK_INCLUDE_DIRS}
    "${CUDA_PATH}/include"
)

# 设置编译器选项
target_compile_options(${PROJECT_NAME} PRIVATE ${LIBDPDK_STATIC_CFLAGS} -w)

# 链接库目录
target_link_directories(${PROJECT_NAME} PRIVATE
    "${CUDA_PATH}/lib64"
)

# 链接 DPDK, CUDA, 和 TensorRT 库
target_link_libraries(${PROJECT_NAME} ${LIBDPDK_STATIC_LDFLAGS}
    cudart
    cudnn
)
