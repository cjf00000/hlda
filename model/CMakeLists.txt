set(LINK_LIBS common gflags)

add_executable(hlda main.cpp BaseHLDA.cpp FiniteSymmetricDirichlet.cpp)

target_link_libraries(hlda ${LINK_LIBS})