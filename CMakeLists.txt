cmake_minimum_required(VERSION 3.14)
project(ImageCalibrationLib 
        VERSION 1.0.0
        DESCRIPTION "Advanced image calibration library using OpenCV"
        LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
option(IMAGE_CALIB_BUILD_SHARED "Build shared library" ON)
option(IMAGE_CALIB_BUILD_EXAMPLES "Build example applications" OFF)
option(IMAGE_CALIB_BUILD_TESTS "Build test applications" OFF)
option(IMAGE_CALIB_USE_TBB "Use Intel TBB for parallelism" OFF)  # Change to OFF
option(IMAGE_CALIB_USE_SIMD "Enable SIMD optimizations" ON)
option(IMAGE_CALIB_USE_OPENCL "Enable OpenCL GPU support" ON)

# 查找依赖项
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

if(IMAGE_CALIB_USE_TBB)
    find_package(TBB REQUIRED)
    add_compile_definitions(USE_TBB)
endif()

# 检查 SIMD 支持
if(IMAGE_CALIB_USE_SIMD)
    include(CheckCXXCompilerFlag)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
            add_compile_definitions(USE_SIMD)
            add_compile_definitions(__AVX2__)
        endif()
    elseif(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
            add_compile_definitions(USE_SIMD)
            add_compile_definitions(__AVX2__)
        endif()
    endif()
endif()

include_directories(include)

# 添加子目录
add_subdirectory(src)

if(IMAGE_CALIB_BUILD_EXAMPLES)
   # add_subdirectory(examples)
endif()

if(IMAGE_CALIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 安装目标 - 头文件
install(DIRECTORY include/ DESTINATION include)

# 导出包配置
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ImageCalibrationLibConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ImageCalibrationLibConfig.cmake
    INSTALL_DESTINATION lib/cmake/ImageCalibrationLib
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ImageCalibrationLibConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ImageCalibrationLibConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ImageCalibrationLibConfigVersion.cmake
    DESTINATION lib/cmake/ImageCalibrationLib
)

install(EXPORT ImageCalibrationLibTargets
    FILE ImageCalibrationLibTargets.cmake
    NAMESPACE ImageCalibrationLib::
    DESTINATION lib/cmake/ImageCalibrationLib
)