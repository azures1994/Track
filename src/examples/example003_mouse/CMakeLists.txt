cmake_minimum_required(VERSION 3.16.3)
project(example003_mouse VERSION 0.1.0 LANGUAGES C CXX)

message("Project: ${PROJECT_NAME}")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message("CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

set(CMAKE_BUILD_TYPE Debug)

set(OpenCV_DIR /e/QFile/QPackages/Linux/qhome2/OpenCV/opencv-2.4.13.6_install_ubuntu2004_gtk/install_ubuntu2004_gtk/)
include_directories(${OpenCV_DIR}/include/)
include_directories(${OpenCV_DIR}/include/opencv/)
link_directories(${OpenCV_DIR}/lib/)
set(OpenCV_LIBS libopencv_calib3d.so
                libopencv_contrib.so
                libopencv_core.so
                libopencv_features2d.so
                libopencv_flann.so
                libopencv_gpu.so
                libopencv_highgui.so
                libopencv_imgproc.so
                libopencv_legacy.so
                libopencv_ml.so
                libopencv_nonfree.so
                libopencv_objdetect.so
                libopencv_ocl.so
                libopencv_photo.so
                libopencv_stitching.so
                libopencv_superres.so
                libopencv_video.so
                libopencv_videostab.so)

# set(OpenCV_DIR /e/QFile/QPackages/Linux/qhome2/OpenCV/opencv-2.4.13.6_install_ubuntu2004_gtk/install_ubuntu2004_gtk/share/OpenCV/)
# find_package(OpenCV REQUIRED)
# if(NOT OpenCV_FOUND)
#     message("OpenCV not found")
# else()
#     message("OpenCV has found")
# endif()
# message("OpenCV_LIBS: ${OpenCV_LIBS}")

set(LIB_LIST ${OpenCV_LIBS})

set(SRC_LIST mouse.cpp)

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
