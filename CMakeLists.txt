cmake_minimum_required(VERSION 3.16)
project(ParProg)

find_package(MPI REQUIRED)

set(DIRS exp hello ring sum lab1 threads matr)

include(cmake/def_add_ex.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

set(TARGETS)

foreach(DIR ${DIRS})
  add_subdirectory(${DIR})
endforeach()

set(MSG_STR)
string(REPLACE ";" ", " MSG_STR "${TARGETS}")
message(STATUS "Collected targets: ${MSG_STR}")

foreach(TARGET IN LISTS TARGETS)
  target_include_directories(${TARGET} PRIVATE SYSTEM ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(${TARGET} PRIVATE ${MPI_CXX_LIBRARIES})

  target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_features(${TARGET} PRIVATE cxx_std_17)

  set_target_properties(${TARGET}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
endforeach()

