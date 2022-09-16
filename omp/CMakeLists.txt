find_package(OpenMP REQUIRED)

set(SDIRS hello sum seq)
foreach(DIR ${SDIRS})
  add_subdirectory(${DIR})
endforeach()

list(APPEND TARGETS ${NEW_TAR})
set(TARGETS ${TARGETS} PARENT_SCOPE)
