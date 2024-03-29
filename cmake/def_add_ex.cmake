macro(UpdTarList NEW_TAR)
  # Append target name to targets list
  # & propagate updated list to parent scope
  list(APPEND TARGETS ${NEW_TAR})
  set(TARGETS ${TARGETS} PARENT_SCOPE)
endmacro()

macro(AddDefTarget NAME)
  # Define target name as variable
  set(TAR ${NAME})

  UpdTarList(${NAME})

  # Create new target w/ given name
  if(${ARGC} GREATER 1)
    add_executable(${TAR} ${ARGN}.cc)
  else()
    add_executable(${TAR} ${TAR}.cc)
  endif()
endmacro()

macro(AddDefPthTar NAME)
  AddDefTarget(${NAME} ${ARGN})
  target_link_libraries(${NAME} PRIVATE Threads::Threads)
endmacro()

macro(AddDefOmpTar NAME)
  AddDefTarget(${NAME} ${ARGN})
  target_link_libraries(${NAME} PRIVATE OpenMP::OpenMP_CXX)
endmacro()

macro(AddDefMpiTar NAME)
  AddDefTarget(${NAME} ${ARGN})
  target_include_directories(${NAME} SYSTEM PRIVATE ${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})
  target_link_libraries(${NAME} PRIVATE ${MPI_CXX_LIBRARIES})
endmacro()
