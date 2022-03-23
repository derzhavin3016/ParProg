AddDefTarget(exp)

find_package(GMP REQUIRED)

AddDefTarget(exp_hard)
target_include_directories(exp_hard PRIVATE ${GMP_INCLUDE_DIR})
target_link_libraries(exp_hard PRIVATE gmp)
