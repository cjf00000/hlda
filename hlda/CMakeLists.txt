add_subdirectory(common)
add_subdirectory(model)

set(LINK_LIBS common gflags)

add_executable(hlda main.cpp)

target_link_libraries(hlda model ${LINK_LIBS}  -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl)
