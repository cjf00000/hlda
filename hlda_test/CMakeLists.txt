project(Test)

#include (CTest)


add_subdirectory(lib/googletest-release-1.8.0)

enable_testing()

include_directories(${googletest_SOURCE_DIR}/include ${googletest_SOURCE_DIR})

add_executable(runTests test_tree.cpp)
target_link_libraries(runTests model common gtest gtest_main gmock)
