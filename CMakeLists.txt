cmake_minimum_required(VERSION 3.10)
project(MidTermProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cerca OpenMP
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

# Collega OpenMP
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
