cmake_minimum_required(VERSION 3.15)
project(bio_ik)

include(cmake/StandardProjectSettings.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
  if(ENABLE_BUILD_WITH_TIME_TRACE)
    target_compile_options(project_options INTERFACE -ftime-trace)
  endif()
endif()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# enable cache system
# include(cmake/Cache.cmake)

# Add linker configuration
include(cmake/Linker.cmake)
configure_linker(project_options)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)

# TODO(tylerjw): re-enable these
# find_package(OpenMP)
# # the specific flag is not yet present in cmake 2.8.12
# if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
#   message(STATUS "OPENMP FOUND")
#   add_compile_options(${OpenMP_CXX_FLAGS})
#   if(NOT OpenMP_CXX_LIBRARIES)
#     # cmake 2.8.12 does not yet specify the library - assume we might need libgomp
#     set(OpenMP_LIBS gomp)
#   else()
#     set(OpenMP_LIBS ${OpenMP_CXX_LIBRARIES})
#   endif()
# else()
#     message(WARNING "OPENMP NOT FOUND. You will suffer performance loss.")
#     set(OpenMP_LIBS)
# endif()

# option(USE_FANN "build the neural-network-based IK solver (experimental)" OFF)
# if(USE_FANN)
#     find_library(FANN_LIBRARIES NAMES fann)
#     find_path(FANN_INCLUDE_DIRS NAMES fann.h)
#     if(NOT FANN_INCLUDE_DIRS OR NOT FANN_LIBRARIES)
#         message(FATAL_ERROR "Neural network solver requested, but libfann was not found.")
#     else()
#         message("Found libfann: ${FANN_LIBRARIES} / ${FANN_INCLUDE_DIRS}")
#     endif()
# else()
#     set(FANN_LIBRARIES)
#     set(FANN_INCLUDE_DIRS)
# endif()

# option(USE_CPPOPTLIB "Include gradient-based solvers from CppNumericalSolvers (bio_ik also provides its own solver)" OFF)
# if(USE_CPPOPTLIB)
#     find_path(CPPOPTLIB_INCLUDE_DIRS
#         NAMES cppoptlib/solver/bfgssolver.h
#         HINTS ../../CppNumericalSolvers/include)
#     if(NOT CPPOPTLIB_INCLUDE_DIRS)
#         message(FATAL_ERROR "cppoptlib support requested, but the headers could not be found.")
#     else()
#         message("Found cppoptlib: ${CPPOPTLIB_INCLUDE_DIRS}")
#     endif()
#     add_definitions(-DENABLE_CPP_OPTLIB)
# else()
#     set(CPPOPTLIB_INCLUDE_DIRS)
# endif()

# TODO(tylerjw) re-enable optimizations
# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#   add_compile_options(-frecord-gcc-switches)
# endif()

# add_compile_options(-std=c++11)

# add_compile_options($<$<CONFIG:Release>:-O3>)
# add_compile_options($<$<CONFIG:Release>:-ftree-vectorize>)
# add_compile_options($<$<CONFIG:Release>:-ffast-math>)

set(SOURCES
    src/goal_types.cpp
    src/problem.cpp
    src/ik_test.cpp
    src/ik_gradient.cpp
    src/ik_evolution_1.cpp
    src/ik_evolution_2.cpp
)

# TODO(tylerjw): re-enable this
# if(USE_FANN)
#     list(APPEND SOURCES src/ik_neural.cpp)
# endif()

# if(USE_CPPOPTLIB)
#     list(APPEND SOURCES src/ik_cppoptlib.cpp)
# endif()

# include_directories(
#     include
#     ${FANN_INCLUDE_DIRS}
#     ${CPPOPTLIB_INCLUDE_DIRS}
# )

add_library(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    project_options
    project_warnings
  PUBLIC
    rclcpp::rclcpp
    moveit_core::moveit_robot_model
    moveit_core::moveit_kinematics_base
    moveit_core::moveit_collision_detection_fcl
    moveit_ros_planning::moveit_rdf_loader
    # ${rclcpp_LIBRARIES}
    # tf2_eigen_kdl::tf2_eigen_kdl
    Eigen3::Eigen
    # TODO(tylerjw): what does this change?
    -static-libgcc
    -static-libstdc++
)

add_library(${PROJECT_NAME}_plugin
  ${SOURCES}
  src/kinematics_plugin.cpp
)
target_include_directories(${PROJECT_NAME}_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${PROJECT_NAME}_plugin
  PRIVATE
    project_options
    project_warnings
    bio_ik
    # TODO(tylerjw): what does this change?
    -static-libgcc
    -static-libstdc++
)

option(BUILD_TESTING "Enable Test Builds" ON)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following lines skip linters
  set(ament_cmake_cpplint_FOUND TRUE) # Conflicts with clang-foramt
  set(ament_cmake_flake8_FOUND TRUE) # Conflicts with black
  set(ament_cmake_uncrustify_FOUND TRUE) # Conflicts with clang-format
  ament_lint_auto_find_test_dependencies()

  add_subdirectory(test)
endif()

# TODO(tylerjw): re-enable this
# set_target_properties(${MOVEIT_LIB_NAME} PROPERTIES VERSION ${${PROJECT_NAME}_VERSION})

# TODO(tylerjw): visibility-headers
# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
# target_compile_definitions(${PROJECT_NAME} PRIVATE "${PROJECT_NAME}_BUILDING_LIBRARY")

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_plugin
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

pluginlib_export_plugin_description_file(
  ${PROJECT_NAME}
  bio_ik_kinematics_description.xml
)

ament_export_include_directories(
  include
)
ament_export_libraries(
  ${PROJECT_NAME}_plugin
)
ament_export_targets(
  export_${PROJECT_NAME}
)

ament_package()
