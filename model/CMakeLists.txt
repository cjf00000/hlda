set(LINK_LIBS common gflags)

add_executable(hlda main.cpp BaseHLDA.cpp FiniteSymmetricDirichlet.cpp Document.cpp CollapsedSampling.cpp)

target_link_libraries(hlda ${LINK_LIBS})