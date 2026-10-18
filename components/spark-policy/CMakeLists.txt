# DSpark host policy sidecar (policy v2): host-only, linked by ds41d when DS41_SPARK_POLICY_V2=ON.
cmake_minimum_required(VERSION 3.24)
project(ds41_spark_policy LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_library(ds41_spark_policy_v2 SHARED src/policy_context.cpp)
target_include_directories(ds41_spark_policy_v2 PUBLIC include)
target_compile_options(ds41_spark_policy_v2 PRIVATE -Wall -Wextra)
install(TARGETS ds41_spark_policy_v2 LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
