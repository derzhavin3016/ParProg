AddDefOmpTar(matmul_omp matmul)
target_compile_options(matmul_omp PRIVATE -O0)
target_compile_definitions(matmul_omp PRIVATE CMP_WAYS)

AddDefOmpTar(matmul_omp_var matmul)
target_compile_options(matmul_omp_var PRIVATE -O0)
