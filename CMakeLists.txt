cmake_minimum_required(VERSION 3.1)
project(elastic)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(JSON_HPP_SRC_DIR "ThirdParty/json/single_include/nlohmann")
set(SPECTRA_DIR "ThirdParty/Spectra/include")
set(LIBIGL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/libigl/include")
#set(MKLROOT "/opt/intel/mkl")
set(SUITESPARSE_HINT "/home/vismay/recode/SuiteSparse")

set(CMAKE_CXX_FLAGS "-O3 -mavx")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DMKL_LP64 -m64")


# libigl
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_PNG              "Use PNG"            ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use IMGUI"           ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(SUITESPARSE REQUIRED QUIET)
find_package(OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#add_definitions(-DEIGEN_USE_MKL_ALL)


#set(MKL_LIBS  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_core -lgomp -lpthread -lm -ldl)

#include_directories(${MKLROOT}/include)
include_directories(${JSON_HPP_SRC_DIR})
include_directories(${SPECTRA_DIR})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(${EIGEN335_INCLUDE_DIRS})

file(GLOB SRCFILES famu/*.cpp)

add_executable(famu "fd_famu.cpp" ${SRCFILES} ${EXTERNALS})

if(APPLE)
    target_link_libraries(famu -lumfpack -lamd igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui ${SUITESPARSE_LIBRARIES})
else()
	set(LAPACK_LIB "/usr/lib/x86_64-linux-gnu/lapack/liblapack.so")
	set(BLAS_LIB "/usr/lib/x86_64-linux-gnu/blas/libblas.so")
	set(FORTRAN_LIB "/usr/lib/gcc/x86_64-linux-gnu/7.4.0/libgfortran.a")
	set(PARDISO_LIB "/u6/b/vismay/libpardiso600-GNU720-X86-64.so")
	set(QUADMATH_LIB "/usr/lib/x86_64-linux-gnu/libquadmath.so.0")
	target_link_libraries(famu ${PARDISO_LIB} ${FORTRAN_LIB} ${LAPACK_LIB} ${BLAS_LIB} ${QUADMATH_LIB} Eigen3::Eigen -fPIC -lglfw -lumfpack -lamd -lpthread -lm igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui OpenMP::OpenMP_CXX ${SUITESPARSE_LIBRARIES} ${})
endif(APPLE)
