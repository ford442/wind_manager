cmake_minimum_required(VERSION 3.16)
project(wind_ref CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# One translation unit; the library is just the scalar formulas.
add_library(wind_ref STATIC wind_ref.cpp)
target_include_directories(wind_ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(wind_ref_test wind_ref_test.cpp)
target_link_libraries(wind_ref_test PRIVATE wind_ref)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # -ffp-contract=off: no FMA fusion, so results do not depend on the CPU target.
  # Deliberately no -ffast-math: the goldens assume IEEE semantics.
  foreach(tgt wind_ref wind_ref_test)
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endforeach()
endif()

enable_testing()
add_test(NAME wind_ref_test COMMAND wind_ref_test)
