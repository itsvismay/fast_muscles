cmake_minimum_required (VERSION 3.5)
project(muscle_gen)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### ---- libIGL options: choose between header only and compiled static library
option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"                     ON)
option(LIBIGL_WITH_COMISO            "Use CoMiso"                   OFF)
option(LIBIGL_WITH_CORK              "Use Cork"                     OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"                   OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"                   OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"                    OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"                   ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    ON)
option(LIBIGL_WITH_PNG               "Use PNG"                      OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"                   OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 OFF)
option(LIBIGL_WITH_XML               "Use XML"                      OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"                   OFF)
option(LIBIGL_WITHOUT_COPYLEFT       "Disable Copyleft libraries"   OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/extern/libigl/cmake)
include(libigl)
### ---- End of libigl

ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/extern/TetWild)


### Add your project files
include_directories("src/" ${PROJECT_SOURCE_DIR}/extern/lf-utils/cpp/)
file(GLOB SRCFILES src/*.cpp)

add_executable(${PROJECT_NAME} ${SRCFILES})
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::cgal libTetWild)
