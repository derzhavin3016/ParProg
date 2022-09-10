AddDefMpiTar(exp)

find_package(GMP REQUIRED)

AddDefMpiTar(exp_hard)
target_include_directories(exp_hard PRIVATE ${GMP_INCLUDE_DIR})
target_link_libraries(exp_hard PRIVATE gmp)
