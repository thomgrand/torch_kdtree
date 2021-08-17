cmake_minimum_required (VERSION 3.13)

#set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project (CP_KNN LANGUAGES CUDA CXX)
add_subdirectory(pybind11)

option(BUILD_TEST "Enable building the C++ test"
       ON)
option(BUILD_GPU "Enable the GPU build"
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

set(COMMON_CPU_SOURCE_LIST
    "src/kdtree.cpp")

set(COMMON_GPU_SOURCE_LIST
    "${COMMON_CPU_SOURCE_LIST}"
    "src/nndistance_g.cu"
    "src/kdtree_g.cu")

set(COMMON_HEADERS_LIST
    "src/cutils.cuh"
    "src/kdtree.hpp"
    "src/nndistance.hpp")

set(EXEC_SOURCE_LIST
	"${COMMON_GPU_SOURCE_LIST}" #Test only available for the GPU
    "src/test_kdtree.cpp")

set(EXEC_HEADERS_LIST
    "${COMMON_HEADERS_LIST}")
    
set(INTERFACE_CPU_SOURCE_LIST
    "${COMMON_CPU_SOURCE_LIST}"
    "src/interface.cpp")

set(INTERFACE_GPU_SOURCE_LIST
    "${COMMON_GPU_SOURCE_LIST}"
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

#Link libraries
set(COMMON_LINKS "")

#OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
    #set(COMMON_LINKS "${COMMON_LINKS}" OpenMP::OpenMP_CXX)
    #target_link_libraries(cp_knn PUBLIC OpenMP::OpenMP_CXX)
    #target_link_libraries(test_kdtree PUBLIC OpenMP::OpenMP_CXX)
endif()


##############################################################
## Create targets and add definitions, dependencies...

if (BUILD_TEST AND BUILD_GPU)
    # add the executable
    add_executable(test_kdtree ${EXEC_SOURCE_LIST}) # ${EXEC_HEADERS_LIST})
    target_link_libraries(test_kdtree cudart ${COMMON_LINKS})
    target_include_directories(test_kdtree PRIVATE ${EIGEN_INSTALL_DIR})
    set_property(TARGET test_kdtree PROPERTY CUDA_ARCHITECTURES 35-real 35-virtual)
    add_dependencies(test_kdtree eigen)
endif()

if (BUILD_GPU)
    set(INTERFACE_SOURCE_LIST ${INTERFACE_GPU_SOURCE_LIST})
else()
    set(INTERFACE_SOURCE_LIST ${INTERFACE_CPU_SOURCE_LIST})
endif()

#PYTHON_ADD_MODULE(oum_export_release tsitsiklis.cpp OrderedUpwindMethod.cpp utility.cpp OUMNPInterface.cpp)

#set_target_properties(test_kdtree PROPERTIES
#                            CUDA_SEPARABLE_COMPILATION ON)

if (BUILD_GPU)
    #find_package(CUDA REQUIRED)
    #find_package(PythonInterp 3.1 REQUIRED)
    #find_package(PythonLibs 3.1 REQUIRED)
    include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    )
    ##pybind11 sets -fvisibility=hidden in INTERFACE_COMPILE_OPTIONS on it's module target
    #get_target_property(modifacecopts module INTERFACE_COMPILE_OPTIONS)
    #list(REMOVE_ITEM modifacecopts "-fvisibility=hidden")
    #set_target_properties(module PROPERTIES INTERFACE_COMPILE_OPTIONS "${modifacecopts}")
    add_library(cp_knn SHARED
                ${INTERFACE_SOURCE_LIST})
    #target_link_libraries(cp_knn cudart ${COMMON_LINKS})
    set_property(TARGET cp_knn PROPERTY CUDA_ARCHITECTURES 35-real 35-virtual)
    #set_target_properties(cp_knn PROPERTIES COMPILE_DEFINITIONS GPU_AVAILABLE=1)
    add_definitions(-DGPU_AVAILABLE)
    target_link_libraries(cp_knn
          ${PYTHON_LIBRARIES}
          cudart
          pybind11::module)

    #Fix wrong extension being created (at least in Windows)
    set_target_properties(
    cp_knn 
    PROPERTIES 
        OUTPUT_NAME "cp_knn"
        SUFFIX ${PYTHON_MODULE_EXTENSION}
    )
    #TODO: Remove lib prefix in linux
          
else()
    pybind11_add_module(cp_knn ${INTERFACE_SOURCE_LIST})
endif()



target_include_directories(cp_knn PRIVATE ${EIGEN_INSTALL_DIR})

#set_target_properties(test_kdtree PROPERTIES CUDA_ARCHITECTURES "50")
#set_target_properties(cp_knn PROPERTIES CUDA_ARCHITECTURES "50")
#if(BUILD_GPU)
#    set_property(TARGET cp_knn PROPERTY CUDA_ARCHITECTURES 50-real 50-virtual)
#endif()

#-fPIC
#set_property(cp_knn PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(cp_knn PUBLIC "$<$<CONFIG:RELEASE>:NDEBUG>")
set_target_properties(cp_knn PROPERTIES COMPILE_DEFINITIONS VERSION_INFO="${CP_KNN_VERSION_MAJOR}.${CP_KNN_VERSION_MINOR}")
#target_compile_definitions(test_kdtree PUBLIC "$<$<CONFIG:RELEASE>:NDEBUG>")

##################################
# Post-build commands
add_custom_command(TARGET cp_knn 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cp_knn> "${PROJECT_SOURCE_DIR}")

##################################
# Tests
enable_testing()
add_test (NAME python-tests
  COMMAND "python" #-m pytest 
  ${PROJECT_SOURCE_DIR}/../scripts/test_knn_unit.py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../"
)

#Dependencies
add_dependencies(cp_knn eigen)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS cp_knn)

##################################
# Debug: Print all variables
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()