cmake_minimum_required(VERSION 3.29)
project(dc_lab2)

set(CMAKE_C_STANDARD 23)

# Add MPI Package to Project
find_package(MPI REQUIRED)

add_executable(hello main.c)

add_executable(matvec
        src/matvec/matvec.h
        src/matvec/matvec.c
        src/matvec/main.c)

add_executable(cannon
        src/cannon/cannon.c
        src/cannon/cannon.h
        src/cannon/main.c)

add_executable(dirichlet
        src/dirichlet/dirichlet.c
        src/dirichlet/dirichlet.h)


# Add libraries for code completion and compiling
target_link_libraries(hello PUBLIC MPI::MPI_CXX)
target_link_libraries(matvec PUBLIC MPI::MPI_CXX)
target_link_libraries(cannon PUBLIC MPI::MPI_CXX)
target_link_libraries(dirichlet PUBLIC MPI::MPI_CXX)
