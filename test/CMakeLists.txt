option(BUILD_TESTING "Enable Test Builds" ON)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # TODO(tylerjw): re-enable this test once it works
  # ament_add_gtest(frame_tests frame_tests.cpp)
  # target_link_libraries(frame_tests
  #   project_options
  #   project_warnings
  #   bio_ik
  # )
endif()
