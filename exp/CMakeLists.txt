cmake_minimum_required(VERSION 3.16)
project(exp)

find_package(MPI REQUIRED)

set(TARGETS ${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}_hard)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

foreach(TAR ${TARGETS})
  add_executable(${TAR} ${TAR}.c)
  target_link_libraries(
  ${TAR}
  PRIVATE
  ${MPI_C_LIBRARIES})
  target_compile_options(${TAR} PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_features(${TAR} PRIVATE c_std_11)
endforeach()