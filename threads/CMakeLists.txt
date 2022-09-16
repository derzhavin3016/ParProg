find_package(Threads REQUIRED)

SUBDIRLIST(SDIRS ${CMAKE_CURRENT_SOURCE_DIR})
foreach(DIR ${SDIRS})
  add_subdirectory(${DIR})
endforeach()

list(APPEND TARGETS ${NEW_TAR})
set(TARGETS ${TARGETS} PARENT_SCOPE)
