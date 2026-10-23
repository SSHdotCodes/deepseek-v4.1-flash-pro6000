# EXL3 K3 (mul1 codebook) grouped expert GEMMs behind the native GemmJob/DeviceExpert ABI.
# Trellis decode and codebook adapted from exllamav3 v1.6.3 (MIT, vendor/EXLLAMAV3_LICENSE).
cmake_minimum_required(VERSION 3.24)
project(ds41_exl3 LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_library(ds41_exl3 SHARED src/exl3.cu)
target_include_directories(ds41_exl3 PUBLIC include "${CMAKE_CURRENT_SOURCE_DIR}/../moe/include")
target_link_libraries(ds41_exl3 PUBLIC CUDA::cudart)
set_target_properties(ds41_exl3 PROPERTIES CUDA_ARCHITECTURES "120a" CUDA_STANDARD 17 CXX_STANDARD 17)
target_compile_options(ds41_exl3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-lineinfo>)
install(TARGETS ds41_exl3 LIBRARY DESTINATION lib)
install(FILES include/ds41/exl3.cuh DESTINATION include/ds41)
