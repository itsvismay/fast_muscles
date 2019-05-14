cmake_minimum_required(VERSION 3.1)
project(elastic)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(JSON_HPP_SRC_DIR "ThirdParty/json/single_include/nlohmann")
set(SPECTRA_DIR "ThirdParty/Spectra/include")
#set(NEW_BFGSB_SRC_DIR "/home/vismay/recode/LBFGSpp/include")
set(LIBIGL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/libigl/include")
set(SUITESPARSE_HINT "/home/vismay/recode/SuiteSparse")
set(CMAKE_CXX_FLAGS "-O3 -mavx")


# libigl
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use IMGUI"           ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(SUITESPARSE REQUIRED QUIET)
find_package(OpenCV REQUIRED)

include_directories(${JSON_HPP_SRC_DIR})
include_directories(${SPECTRA_DIR})
#include_directories(${NEW_BFGSB_SRC_DIR})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB SRCFILES famu/*.cpp)

add_executable(famu "fd_famu.cpp" ${SRCFILES} ${EXTERNALS})

target_link_libraries(famu -lglfw -lumfpack -lamd igl::core igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui -fopenmp ${OpenCV_LIBS} ${SUITESPARSE_LIBRARIES})



 
