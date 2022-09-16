cmake_minimum_required(VERSION 3.16)
project(ParProg)

set(DIRS mpi threads omp)

include(cmake/def_add_ex.cmake)
include(cmake/subdirlist.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(TARGETS)

foreach(DIR ${DIRS})
  add_subdirectory(${DIR})
endforeach()

set(MSG_STR)
string(REPLACE ";" ", " MSG_STR "${TARGETS}")
message(STATUS "Collected targets: ${MSG_STR}")

foreach(TARGET IN LISTS TARGETS)
  target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_features(${TARGET} PRIVATE cxx_std_17)

  set_target_properties(${TARGET}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
endforeach()
