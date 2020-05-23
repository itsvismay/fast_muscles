project(elastic)

if(APPLE)
	cmake_minimum_required(VERSION 3.17)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
	set(JSON_HPP_SRC_DIR "ThirdParty/json/single_include/nlohmann")
	set(SPECTRA_DIR "ThirdParty/Spectra/include")
	set(LIBIGL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libigl/include")
	set(BARTELS_INCLUDE_DIR "ThirdParty/Bartels/include")
	set(NEW_BFGSB_SRC_DIR "ThirdParty/LBFGSpp/include")
	set(MKLROOT "/opt/intel/mkl")
	set(BLA_VENDOR Intel10_64lp)
	
	set(CMAKE_CXX_FLAGS "-O3 -mavx")
	set(CMAKE_CXX_FLAGS "-O3 -mavx")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DMKL_ILP64 -m64")

	option(LIBIGL_WITH_EMBREE		"USE EMBREE" 		ON)
	option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
	option(LIBIGL_WITH_PNG              "Use PNG"            ON)
	option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
	option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use IMGUI"           ON)

	find_package(LIBIGL REQUIRED QUIET)
	find_package (Eigen3 3.3 REQUIRED NO_MODULE)
	find_package(OpenMP REQUIRED)
	find_package(BLAS REQUIRED)

	#add_definitions(-DEIGEN_USE_MKL_ALL)
	#set(MKL_LIBS -L${MKLROOT}/lib/libmkl_intel_ilp64.a -L${MKLROOT}/lib/libmkl_intel_thread.a -L${MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl)


	#include_directories(${MKLROOT}/include)
	include_directories(${OpenMP_CXX_INCLUDE_DIRS})
	include_directories(${JSON_HPP_SRC_DIR})
	include_directories(${SPECTRA_DIR})
	include_directories(${EIGEN335_INCLUDE_DIRS})
	include_directories(${NEW_BFGSB_SRC_DIR})
	include_directories(${BARTELS_INCLUDE_DIR})



	file(GLOB SRCFILES famu/*.cpp)
	file(GLOB SRCFILES exact/*.cpp)

	file(GLOB SRCFILES exact/*.cpp)

	add_executable(exact "exact.cpp" ${SRCFILES} ${EXTERNALS})
    target_link_libraries(exact igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::embree OpenMP::OpenMP_CXX)

	#add_executable(famu "famu.cpp" ${SRCFILES} ${EXTERNALS})
    #target_link_libraries(famu  igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::embree OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})


    #add_executable(fd_famu "fd_famu.cpp" ${SRCFILES} ${EXTERNALS})
    #target_link_libraries(fd_famu  igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::embree OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})
    
else()
	cmake_minimum_required(VERSION 3.10)
	set(CMAKE_CXX_STANDARD 14)
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
	set(JSON_HPP_SRC_DIR "ThirdParty/json/single_include/nlohmann")
	set(SPECTRA_DIR "ThirdParty/Spectra/include")
	set(LIBIGL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/libigl/include")
	set(MKLROOT "/opt/intel/mkl")
	set(NEW_BFGSB_SRC_DIR "ThirdParty/LBFGSpp/include")
	 set(BARTELS_INCLUDE_DIR "ThirdParty/Bartels/include")

	set(CMAKE_CXX_FLAGS "-O3 -mavx")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DMKL_LP64 -m64")


	# libigl
	option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
	option(LIBIGL_WITH_PNG              "Use PNG"            ON)
	option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
	option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use IMGUI"           ON)

	find_package(LIBIGL REQUIRED QUIET)
	find_package(OpenMP REQUIRED)
	find_package (Eigen3 3.3 REQUIRED NO_MODULE)

	add_definitions(-DEIGEN_USE_MKL_ALL)


	set(MKL_LIBS  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_core -lgomp -lpthread -lm -ldl)


	include_directories(${MKLROOT}/include)
	include_directories(${JSON_HPP_SRC_DIR})
	include_directories(${SPECTRA_DIR})
	include_directories(${EIGEN335_INCLUDE_DIRS})
	include_directories(${NEW_BFGSB_SRC_DIR})
	include_directories(${BARTELS_INCLUDE_DIR})

	file(GLOB SRCFILES exact/*.cpp)
	add_executable(exact "exact.cpp" ${SRCFILES} ${EXTERNALS})
	target_link_libraries(exact  ${MKL_LIBS} Eigen3::Eigen -lblas -llapack -lglfw -lumfpack -lamd igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui  OpenMP::OpenMP_CXX )	
	
	#add_executable(famu "famu.cpp" ${SRCFILES} ${EXTERNALS})
    	#target_link_libraries(famu  ${MKL_LIBS} Eigen3::Eigen -lblas -llapack -lglfw -lumfpack -lamd igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::embree OpenMP::OpenMP_CXX ${SUITESPARSE_LIBRARIES} )

    	#add_executable(fd_famu "fd_famu.cpp" ${SRCFILES} ${EXTERNALS})
    	#target_link_libraries(fd_famu  ${MKL_LIBS} Eigen3::Eigen -lblas -llapack -lglfw -lumfpack -lamd igl::core igl::png igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::embree OpenMP::OpenMP_CXX ${SUITESPARSE_LIBRARIES} )

endif(APPLE)
