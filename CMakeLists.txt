cmake_minimum_required(VERSION 3.1)
project(elastic)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(JSON_HPP_SRC_DIR "ThirdParty/json/single_include/nlohmann")
set(SPECTRA_DIR "ThirdParty/Spectra/include")
set(NEW_BFGSB_SRC_DIR "ThirdParty/LBFGSpp/include")
set(LIBIGL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/libigl/include")
set(SUITESPARSE_HINT "/home/vismay/recode/SuiteSparse")
set(CMAKE_CXX_FLAGS "-O3")


# libigl
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(SUITESPARSE REQUIRED QUIET)
find_package(OpenCV REQUIRED)

include_directories(${JSON_HPP_SRC_DIR})
include_directories(${SPECTRA_DIR})
include_directories(${NEW_BFGSB_SRC_DIR})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


# Add your project files
file(GLOB HEADFILES "*.h")
file(GLOB HEADFILES "PreProcessing/*.h")
file(GLOB SRCFILES "PreProcessing/*.cpp")
file(GLOB EXTERNALS "${SPECTRA_DIR}/*.h" "${SPECTRA_DIR}/MatOp/*.h")

add_executable(redfd "redFD.cpp" ${HEADFILES} ${SRCFILES} ${EXTERNALS})
#add_executable(unreduced "unreduced.cpp" ${HEADFILES} ${SRCFILES} ${EXTERNALS})
add_executable(fd "finitedifference.cpp" ${HEADFILES} ${SRCFILES} ${EXTERNALS})
add_executable(${PROJECT_NAME} "main.cpp" ${HEADFILES} ${SRCFILES} ${EXTERNALS})
add_executable(reduced "reduced.cpp" ${HEADFILES} ${SRCFILES} ${EXTERNALS})

#target_link_libraries(unreduced -lglfw igl::core igl::opengl igl::opengl_glfw -fopenmp ${OpenCV_LIBS})
target_link_libraries(redfd -lglfw igl::core igl::opengl igl::opengl_glfw -fopenmp ${OpenCV_LIBS})
target_link_libraries(fd -lglfw igl::core igl::opengl igl::opengl_glfw -fopenmp  ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} -lglfw igl::core igl::opengl igl::opengl_glfw -fopenmp ${OpenCV_LIBS})
target_link_libraries(reduced -lglfw igl::core igl::opengl igl::opengl_glfw -fopenmp ${OpenCV_LIBS})
 
