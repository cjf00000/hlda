include_directories(${BigTopicModel_SOURCE_DIR}/src/util)
include_directories(${BigTopicModel_SOURCE_DIR}/src/model)
include_directories(${BigTopicModel_SOURCE_DIR}/src/engine)
include_directories(/opt/intel/mkl/include)
link_directories(/opt/intel/mkl/lib/intel64)

add_library(hlda_lib base_hlda.cpp document.cpp partially_collapsed_sampling.cpp block_gibbs_sampling.cpp external_sampling.cpp concurrent_tree.cpp distributed_tree.cpp hlda_corpus.cpp)
add_executable(hlda main.cpp)
target_link_libraries(hlda hlda_lib util ${MPI_LIBRARIES} ${GLOG} ${GFLAGS} mkl_intel_lp64 mkl_core mkl_sequential m dl ${TCMALLOC})

add_executable(hlda_fmt hlda_fmt.cpp)
target_link_libraries(hlda_fmt hlda_lib ${MPI_LIBRARIES} util ${GLOG}
        ${GFLAGS} z)

add_executable(test_mpi test_mpi.cpp)
target_link_libraries(test_mpi hlda_lib util ${MPI_LIBRARIES} ${GLOG} ${GFLAGS} ${TCMALLOC})
#
#add_executable(test2 test2.cpp)
#
add_executable(test_parallel_tree test_parallel_tree.cpp)
target_link_libraries(test_parallel_tree hlda_lib util ${MPI_LIBRARIES} ${GLOG} ${GFLAGS} mkl_intel_lp64 mkl_core mkl_sequential m dl)

