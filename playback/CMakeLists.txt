cmake_minimum_required(VERSION 3.1)
project(elastic)

if(APPLE)
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
	set(JSON_HPP_SRC_DIR "../ThirdParty/json/single_include/nlohmann")
	set(SPECTRA_DIR "../ThirdParty/Spectra/include")
	set(LIBIGL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/libigl/include")
	set(CMAKE_CXX_FLAGS "-O3 -mavx")

	# libigl
	option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
	option(LIBIGL_WITH_PNG              "Use PNG"            ON)
	option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
	option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use IMGUI"           ON)

	find_package(LIBIGL REQUIRED QUIET)

	include_directories(${JSON_HPP_SRC_DIR})
	include_directories(${EIGEN335_INCLUDE_DIRS})

	file(GLOB SRCFILES ../famu/read_config_file.cpp)

	add_executable(playback "main.cpp" ${SRCFILES} ${EXTERNALS})

    target_link_libraries(playback igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui )
else()
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
	set(JSON_HPP_SRC_DIR "../ThirdParty/json/single_include/nlohmann")
	set(SPECTRA_DIR "../ThirdParty/Spectra/include")
	set(LIBIGL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/libigl/include")
	set(MKLROOT "/opt/intel/mkl")

	set(CMAKE_CXX_FLAGS "-O3 -mavx")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DMKL_LP64 -m64")
	set(MKL_LIBS  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_core -lgomp -lpthread -lm -ldl)


	# libigl
	option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
	option(LIBIGL_WITH_PNG              "Use PNG"            ON)
	option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
	option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use IMGUI"           ON)

	find_package(LIBIGL REQUIRED QUIET)
	find_package (Eigen3 3.3 REQUIRED NO_MODULE)
	find_package(OpenMP REQUIRED)

	include_directories(${MKLROOT}/include)
	include_directories(${JSON_HPP_SRC_DIR})
	include_directories(${EIGEN335_INCLUDE_DIRS})
	include_directories(${SPECTRA_DIR})

	file(GLOB SRCFILES ../famu/*.cpp)
	add_executable(playback "main.cpp" ${SRCFILES} ${EXTERNALS})

    target_link_libraries(playback ${MKL_LIBS} Eigen3::Eigen -lblas -llapack -lglfw -lumfpack -lamd igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::embree OpenMP::OpenMP_CXX ${SUITESPARSE_LIBRARIES} )
endif(APPLE)
