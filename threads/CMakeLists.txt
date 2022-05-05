find_package(Threads REQUIRED)

set(SDIRS hello)
foreach(DIR ${SDIRS})
  add_subdirectory(${DIR})
endforeach()

list(APPEND TARGETS ${NEW_TAR})
set(TARGETS ${TARGETS} PARENT_SCOPE)