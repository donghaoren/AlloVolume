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

ADD_LIBRARY ( dataset
    src/dataset.cpp
    src/dataset_flash.cpp
)

CUDA_ADD_LIBRARY ( renderer_cuda
    src/renderer_cuda.cu
    src/cuda_common.cu
    src/lens.cu
    src/image_codec.cpp
    src/image.cpp
)

ADD_EXECUTABLE ( test
    src/test.cpp
)

ADD_EXECUTABLE ( dataset_converter
    src/dataset_converter.cpp
)

ADD_EXECUTABLE ( renderer
    src/renderer.cpp
)

IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    FIND_PACKAGE ( GLFW REQUIRED )
    TARGET_LINK_LIBRARIES ( test
      ${GLFW_LIBRARY}
    )
ENDIF ( )

IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )
    TARGET_LINK_LIBRARIES ( test
      ${GLFW_STATIC_LIBRARIES}
    )
ENDIF ( )

TARGET_LINK_LIBRARIES ( dataset_converter
  dataset
  ${HDF5_LIBRARIES}
)

TARGET_LINK_LIBRARIES ( test
    dataset
    renderer_cuda
    ${FREEIMAGE_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

TARGET_LINK_LIBRARIES ( renderer
    dataset
    renderer_cuda
    ${FREEIMAGE_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${ZeroMQ_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
)
