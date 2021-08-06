cmake_minimum_required (VERSION 3.13)

#set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project (CP_KNN LANGUAGES CUDA CXX)
add_subdirectory(pybind11)

option(BUILD_TEST "Enable building the C++ test"
       ON)

# The version number.
set (CP_KNN_VERSION_MAJOR 0)
set (CP_KNN_VERSION_MINOR 1)
set (CMAKE_CXX_STANDARD 11)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

IF (WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
ENDIF()

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

#Search for python
#find_package(Python 3.1 COMPONENTS Interpreter Development REQUIRED)

#TODO: Windows...
#list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

set(COMMON_SOURCE_LIST
    "src/kdtree.cpp"
    "src/nndistance_g.cu"
    "src/kdtree_g.cu")

set(COMMON_HEADERS_LIST
    "src/cutils.cuh"
    "src/kdtree.hpp"
    "src/nndistance.hpp")

set(EXEC_SOURCE_LIST
	"${COMMON_SOURCE_LIST}"
    "src/test_kdtree.cpp")

set(EXEC_HEADERS_LIST
    "${COMMON_HEADERS_LIST}")
    
set(INTERFACE_SOURCE_LIST
    "${COMMON_SOURCE_LIST}"
    "src/nndistance_g.cu"
    "src/nndistance.cpp"
    "src/interface.cpp")



IF(WIN32)
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
         #PREFIX "src"
         FILES ${INTERFACE_SOURCE_LIST})
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES ${EXEC_SOURCE_LIST})
    #source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
    #     PREFIX "headers"
    #     FILES ${EXEC_HEADERS_LIST})
    source_group("src" FILES "src/cutils.cuh")

ENDIF()

include(ExternalProject)
set(EIGEN_INSTALL_DIR "${PROJECT_SOURCE_DIR}/eigenlib")
ExternalProject_Add( eigen
  GIT_REPOSITORY  "https://gitlab.com/libeigen/eigen"
  SOURCE_DIR "${EIGEN_INSTALL_DIR}"
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
  )

#Old, see below
#INCLUDE_DIRECTORIES("${EIGEN_INSTALL_DIR}")

##############################################################
## Create targets and add definitions, dependencies...

if (BUILD_TEST)
    # add the executable
    add_executable(test_kdtree ${EXEC_SOURCE_LIST}) # ${EXEC_HEADERS_LIST})
endif()

#PYTHON_ADD_MODULE(oum_export_release tsitsiklis.cpp OrderedUpwindMethod.cpp utility.cpp OUMNPInterface.cpp)

#set_target_properties(test_kdtree PROPERTIES
#                            CUDA_SEPARABLE_COMPILATION ON)
pybind11_add_module(cp_knn src/interface.cpp)
add_library(tf_nndistance SHARED ${INTERFACE_SOURCE_LIST})

target_compile_definitions(tf_nndistance PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(test_kdtree cudart)
#target_link_libraries(tf_nndistance tensorflow_framework cudart)
target_link_libraries(tf_nndistance)

target_include_directories(test_kdtree PRIVATE ${EIGEN_INSTALL_DIR})
target_include_directories(tf_nndistance PRIVATE ${EIGEN_INSTALL_DIR} ${TF_INCLUDE_DIR})

#-fPIC
#set_property(tf_nndistance PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(tf_nndistance PUBLIC "$<$<CONFIG:RELEASE>:NDEBUG>")
set_target_properties(cp_knn PROPERTIES COMPILE_DEFINITIONS VERSION_INFO="${CP_KNN_VERSION_MAJOR}.${CP_KNN_VERSION_MINOR}")
#target_compile_definitions(test_kdtree PUBLIC "$<$<CONFIG:RELEASE>:NDEBUG>")

##################################
# Post-build commands
add_custom_command(TARGET tf_nndistance 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tf_nndistance> "${PROJECT_SOURCE_DIR}/..")

##################################
# Tests
enable_testing()
add_test (NAME python-tests
  COMMAND "python" #-m pytest 
  ${PROJECT_SOURCE_DIR}/../scripts/test_knn_unit.py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../"
)

#Dependencies
add_dependencies(tf_nndistance eigen)
add_dependencies(test_kdtree eigen)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS tf_nndistance)

##################################
# Debug: Print all variables
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()