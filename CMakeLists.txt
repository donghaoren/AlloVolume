CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )
PROJECT ( AlloVolumeRendering )

SET ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )

SET ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" )

SET ( EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" )
SET ( LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" )

SET ( CUDA_HOST_COMPILER clang )
FIND_PACKAGE ( CUDA )
FIND_PACKAGE ( HDF5 )
FIND_PACKAGE ( GLFW REQUIRED )

INCLUDE_DIRECTORIES (
  include
  ${HDF5_INCLUDE_DIRS}
)

ADD_LIBRARY ( dataset
    src/dataset_flash.cpp
)

CUDA_ADD_LIBRARY ( renderer
    src/renderer_cuda.cu
)

ADD_EXECUTABLE ( test
    src/test.cpp
)

TARGET_LINK_LIBRARIES ( test
    dataset
    renderer
    ${HDF5_LIBRARIES}
    ${GLFW_LIBRARY}
    "-framework OpenGL"
)
