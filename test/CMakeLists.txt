option(BUILD_TESTING "Enable Test Builds" ON)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)

  # TODO(tylerjw): re-enable this test once it works
  # ament_add_gtest(frame_tests frame_tests.cpp)
  # target_link_libraries(frame_tests
  #   project_options
  #   project_warnings
  #   bio_ik
  # )

  ament_add_gtest(parameter_tests parameter_tests.cpp)
  target_link_libraries(parameter_tests
    project_options
    project_warnings
    bio_ik
  )

  ament_add_gtest(goal_tests goal_tests.cpp)
  target_link_libraries(goal_tests
    project_options
    project_warnings
    bio_ik
    gmock
  )
endif()
