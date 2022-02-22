cmake_minimum_required(VERSION 3.16)
project(sum)

find_package(MPI REQUIRED)

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
add_executable(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}.c)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

target_link_libraries(${CMAKE_PROJECT_NAME}
  PRIVATE
  ${MPI_C_LIBRARIES})

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE c_std_11)

add_custom_target(run DEPENDS ${CMAKE_PROJECT_NAME}
  COMMAND mpirun -np 2 ${CMAKE_PROJECT_NAME}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  VERBATIM
)