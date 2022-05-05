find_package(Threads REQUIRED)

set(SDIRS hello sum ring attrs int)
foreach(DIR ${SDIRS})
  add_subdirectory(${DIR})
endforeach()

list(APPEND TARGETS ${NEW_TAR})
set(TARGETS ${TARGETS} PARENT_SCOPE)