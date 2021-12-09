#pragma once

#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN

#include "common/catch.hpp"
#include "common/fmt.hpp"
#include "common/utils.hpp"

#include "assert.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <algorithm>
#include <random>
#include <string>
#include <chrono>

#include <cuda.h>

static bool verify(const std::vector<int> &ref_sol, const std::vector<int> &sol) {
    for (size_t i = 0; i < ref_sol.size(); i++) {
        INFO("Results differ from reference solution at " << i);
        REQUIRE(ref_sol[i] == sol[i]);
    }
    return true;
}


static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    LOG(critical,
        std::string(fmt::format("{}@{}: CUDA Runtime Error: {}\n", file, line,
                                cudaGetErrorString(result))));
    exit(-1);
  }
}


