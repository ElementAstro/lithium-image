# 收集所有源文件
set(SOURCES
    BatchProcessing.cpp
    CameraCalibrator.cpp
    ImageCorrection.cpp
    OptimizedCorrection.cpp
)

# 定义库类型
if(IMAGE_CALIB_BUILD_SHARED)
    add_library(ImageCalibrationLib SHARED ${SOURCES})
    set_target_properties(ImageCalibrationLib PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
    )
    target_compile_definitions(ImageCalibrationLib PRIVATE -DIMAGE_CALIB_EXPORTS)
    target_compile_definitions(ImageCalibrationLib PUBLIC -DIMAGE_CALIB_SHARED)
else()
    add_library(ImageCalibrationLib STATIC ${SOURCES})
endif()

# 添加包含目录
target_include_directories(ImageCalibrationLib 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接依赖库
target_link_libraries(ImageCalibrationLib 
    PUBLIC 
        ${OpenCV_LIBS}
    PRIVATE
        spdlog::spdlog
)

if(IMAGE_CALIB_USE_TBB)
    target_link_libraries(ImageCalibrationLib PRIVATE TBB::tbb)
endif()

# 设置编译选项，启用警告
if (MSVC)
    target_compile_options(ImageCalibrationLib PRIVATE /W4)
else()
    target_compile_options(ImageCalibrationLib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 生成导出头文件
include(GenerateExportHeader)
generate_export_header(ImageCalibrationLib
    EXPORT_FILE_NAME ${CMAKE_SOURCE_DIR}/include/calibration/Export.hpp
)

# 安装库文件
install(TARGETS ImageCalibrationLib
    EXPORT ImageCalibrationLibTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)