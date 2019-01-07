cmake_minimum_required(VERSION 3.1)
project(elastic)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(JSON_HPP_SRC_DIR "//u6/b/vismay/Muscles/json/single_include/nlohmann")
set(SPECTRA_DIR "/u6/b/vismay/Muscles/spectra/include")
set(SUITESPARSE_HINT "/home/vismay/recode/SuiteSparse")
set(NEW_BFGSB_SRC_DIR "/u6/b/vismay/Muscles/LBFGSpp/include")
set(CMAKE_CXX_FLAGS "-O3")


# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" ON)
option(LIBIGL_WITH_ANTTWEAKBAR      "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL             "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO           "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK             "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE           "Use Embree"         OFF)
option(LIBIGL_WITH_LIM              "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB           "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK            "Use MOSEK"          OFF)
option(LIBIGL_WITH_NANOGUI          "Use Nanogui menu"   OFF)
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_PNG              "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON           "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN           "Use Tetgen"         ON)
option(LIBIGL_WITH_TRIANGLE         "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML              "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)
find_package(OpenCV REQUIRED)

include_directories(${JSON_HPP_SRC_DIR})
include_directories(${SPECTRA_DIR})
include_directories(${NEW_BFGSB_SRC_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your project files
file(GLOB HEADFILES "*.h")
file(GLOB HEADFILES "PreProcessing/*.h")
file(GLOB SRCFILES "PreProcessing/*.cpp")
file(GLOB EXTERNALS "${SPECTRA_DIR}/*.h" "${SPECTRA_DIR}/MatOp/*.h")

add_executable(${PROJECT_NAME} "main.cpp" ${HEADFILES} ${SRCFILES} ${EXTERNALS})
target_link_libraries(${PROJECT_NAME} -lglfw igl::core igl::opengl igl::opengl_glfw -fopenmp ${OpenCV_LIBS})
 
