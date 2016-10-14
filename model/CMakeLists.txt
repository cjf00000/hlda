set(LINK_LIBS common gflags)

add_executable(hlda main.cpp BaseHLDA.cpp FiniteSymmetricDirichlet.cpp Document.cpp CollapsedSampling.cpp PartiallyCollapsedSampling.cpp ExternalHLDA.cpp)
add_executable(hdp main_hdp.cpp DirectSamplingHDP.cpp)

target_link_libraries(hlda ${LINK_LIBS})