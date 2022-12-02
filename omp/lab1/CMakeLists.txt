AddDefOmpTar(lab1_seq_omp lab1)
target_link_libraries(lab1_seq_omp PRIVATE m)
target_compile_definitions(lab1_seq_omp PRIVATE SEQ_VER)

AddDefOmpTar(lab1_par_omp lab1)
target_link_libraries(lab1_par_omp PRIVATE m)
