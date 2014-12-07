CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )
PROJECT ( AlloVolumeRendering )

SET ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )

SET ( EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" )
SET ( LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" )

IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
  SET ( CUDA_HOST_COMPILER clang )
ENDIF ( )

FIND_PACKAGE ( PkgConfig REQUIRED )

PKG_SEARCH_MODULE ( GLFW REQUIRED glfw3 )
PKG_SEARCH_MODULE ( ZeroMQ REQUIRED libzmq )
PKG_SEARCH_MODULE ( YAMLCPP REQUIRED yaml-cpp )

FIND_PACKAGE ( CUDA REQUIRED )
FIND_PACKAGE ( HDF5 REQUIRED )
FIND_PACKAGE ( OpenGL REQUIRED )
FIND_PACKAGE ( FreeImage REQUIRED )
FIND_PACKAGE ( Protobuf REQUIRED )

SET ( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 )

LINK_DIRECTORIES (
  ${ZeroMQ_LIBRARY_DIRS}
  ${HDF5_LIBRARY_DIRS}
  ${YAMLCPP_LIBRARY_DIRS}
  ${GLFW_LIBRARY_DIRS}
)

INCLUDE_DIRECTORIES (
  include
  ${HDF5_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIRS}
  ${YAMLCPP_INCLUDE_DIRS}
  ${FREEIMAGE_INCLUDE_PATH}
)

ADD_LIBRARY (
  dataset
    src/dataset/dataset.cpp
    src/dataset/dataset_flash.cpp
)

TARGET_LINK_LIBRARIES (
  dataset
    ${HDF5_LIBRARIES}
)

CUDA_ADD_LIBRARY (
  renderer_cuda
    src/renderer/renderer_cuda.cu
    src/renderer/cuda_common.cu
    src/renderer/lens.cu
    src/renderer/image_codec.cpp
    src/renderer/image.cpp

)

CUDA_ADD_LIBRARY (
  allosphere_calibration
    src/allosphere/allosphere_calibration.cpp
    src/allosphere/allosphere_lens.cu
)

TARGET_LINK_LIBRARIES (
  renderer_cuda
    ${FREEIMAGE_LIBRARIES}
)

ADD_EXECUTABLE (
  test
    src/test.cpp
)

ADD_EXECUTABLE ( dataset_converter
    src/dataset_converter.cpp
)

TARGET_LINK_LIBRARIES (
  dataset_converter
    dataset
)

ADD_EXECUTABLE ( allovolume_renderer
    src/allovolume_renderer.cpp
    src/renderer.pb.cc
    src/allosync.cpp
)

ADD_EXECUTABLE ( allovolume_controller
    src/allovolume_controller.cpp
    src/renderer.pb.cc
    src/allosync.cpp
)

TARGET_LINK_LIBRARIES (
  allovolume_renderer
    dataset
    renderer_cuda
    allosphere_calibration
    ${ZeroMQ_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${PROTOBUF_LIBRARY}
    ${GLFW_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

TARGET_LINK_LIBRARIES (
  allovolume_controller
    dataset
    ${ZeroMQ_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${PROTOBUF_LIBRARY}
)

TARGET_LINK_LIBRARIES (
  test
    dataset
    renderer_cuda
    allosphere_calibration
    ${GLFW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
)

# IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
#     FIND_PACKAGE ( GLFW REQUIRED )
#     TARGET_LINK_LIBRARIES ( test
#       ${GLFW_LIBRARY}
#     )
#     TARGET_LINK_LIBRARIES ( allovolume_renderer
#       ${GLFW_LIBRARY}
#     )
# ENDIF ( )

# IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )
#     TARGET_LINK_LIBRARIES ( test
#       ${GLFW_STATIC_LIBRARIES}
#     )
#     TARGET_LINK_LIBRARIES ( allovolume_renderer
#       ${GLFW_STATIC_LIBRARIES}
#     )
# ENDIF ( )

# TARGET_LINK_LIBRARIES ( test
#     dataset
#     renderer_cuda
#     ${OPENGL_LIBRARIES}
#     ${YAMLCPP_LIBRARIES}
# )

# TARGET_LINK_LIBRARIES ( allovolume_renderer
#     dataset
#     renderer_cuda
#     ${FREEIMAGE_LIBRARIES}
#     ${HDF5_LIBRARIES}
#     ${ZeroMQ_LIBRARIES}
#     ${YAMLCPP_LIBRARIES}
#     ${OPENGL_LIBRARIES}
#     ${PROTOBUF_LIBRARY}
#     readline
# )
