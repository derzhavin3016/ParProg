macro(AddDefTarget name)
  # Define target name as variable
  set(TAR ${name})

  # Append target name to targets list
  # & propagate updated list to parent scope
  list(APPEND TARGETS ${TAR})
  set(TARGETS ${TARGETS} PARENT_SCOPE)

  # Create new target w/ given name
  add_executable(${TAR} ${TAR}.c)
endmacro()
