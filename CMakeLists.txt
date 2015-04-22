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
PKG_SEARCH_MODULE ( Protobuf REQUIRED protobuf )

FIND_PACKAGE ( CUDA REQUIRED )
FIND_PACKAGE ( HDF5 REQUIRED )
FIND_PACKAGE ( OpenGL REQUIRED )
FIND_PACKAGE ( FreeImage REQUIRED )
FIND_PACKAGE ( GLUT REQUIRED )
FIND_PACKAGE ( Boost REQUIRED COMPONENTS filesystem system )

SET ( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 )

LINK_DIRECTORIES (
  ${ZeroMQ_LIBRARY_DIRS}
  ${HDF5_LIBRARY_DIRS}
  ${YAMLCPP_LIBRARY_DIRS}
  ${GLFW_LIBRARY_DIRS}
  ${Protobuf_LIBRARY_DIRS}
  ${GLUT_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

INCLUDE_DIRECTORIES (
  include
  ${HDF5_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIRS}
  ${YAMLCPP_INCLUDE_DIRS}
  ${FREEIMAGE_INCLUDE_PATH}
  ${GLUT_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

ADD_LIBRARY (
  allovolume_dataset SHARED
    src/dataset/dataset.cpp
    src/dataset/dataset_flash.cpp
)

TARGET_LINK_LIBRARIES (
  allovolume_dataset
    ${HDF5_LIBRARIES}
)

CUDA_ADD_LIBRARY (
  allovolume_renderer SHARED
    src/renderer/renderer_cuda.cu
    src/renderer/cuda_common.cu
    src/renderer/transfer_function.cpp
    src/renderer/lens.cu
    src/renderer/image.cu
    src/renderer/image_codec.cpp
    src/timeprofiler.cpp
)

TARGET_LINK_LIBRARIES (
  allovolume_renderer
    ${FREEIMAGE_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
)

CUDA_ADD_LIBRARY (
  allovolume_allosphere_calibration SHARED
    src/allosphere/allosphere_calibration.cpp
    src/allosphere/allosphere_lens.cu
)

TARGET_LINK_LIBRARIES (
  allovolume_allosphere_calibration
    allovolume_renderer
    ${YAMLCPP_LIBRARIES}
    ${FREEIMAGE_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

ADD_EXECUTABLE ( dataset_converter
    src/dataset_converter.cpp
)

TARGET_LINK_LIBRARIES (
  dataset_converter
    allovolume_dataset
)

ADD_EXECUTABLE ( renderer
    src/allovolume_renderer.cpp
    src/allovolume_protocol.pb.cc
)

ADD_EXECUTABLE ( controller
    src/allovolume_controller.cpp
    src/allovolume_protocol.pb.cc
)

ADD_EXECUTABLE ( render_cli
    src/render_cli.cpp
    src/allovolume_protocol.pb.cc
)

ADD_LIBRARY ( allovolume_omnistereo_renderer SHARED
  src/omnistereo_renderer.cpp
  src/allovolume_protocol.pb.cc
)

TARGET_LINK_LIBRARIES (
  renderer
    allovolume_dataset
    allovolume_renderer
    allovolume_allosphere_calibration
    ${ZeroMQ_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${Protobuf_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

TARGET_LINK_LIBRARIES (
  allovolume_omnistereo_renderer
    allovolume_dataset
    allovolume_renderer
    allovolume_allosphere_calibration
    ${ZeroMQ_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${Protobuf_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

TARGET_LINK_LIBRARIES (
  controller
    allovolume_renderer
    allovolume_dataset
    readline
    ${ZeroMQ_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${Protobuf_LIBRARIES}
    ${Boost_LIBRARIES}
)

TARGET_LINK_LIBRARIES (
  render_cli
    allovolume_renderer
    allovolume_dataset
    readline
    ${YAMLCPP_LIBRARIES}
    ${Protobuf_LIBRARIES}
    ${Boost_LIBRARIES}
)

INSTALL (
  DIRECTORY
    include/allovolume
  DESTINATION include
)

# Targets.
INSTALL (
  TARGETS
    allovolume_omnistereo_renderer
    allovolume_allosphere_calibration
    allovolume_renderer
    allovolume_dataset
    renderer
    controller
    render_cli
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
