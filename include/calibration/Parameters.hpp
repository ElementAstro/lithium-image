#pragma once

#include <opencv2/opencv.hpp>

struct CalibrationParams {
  double wavelength;         // Wavelength, unit: nanometer
  double aperture;           // Aperture diameter, unit: millimeter
  double obstruction;        // Obstruction diameter, unit: millimeter
  double filter_width;       // Filter bandwidth, unit: nanometer
  double transmissivity;     // Transmissivity
  double gain;               // Gain
  double quantum_efficiency; // Quantum efficiency
  double extinction;         // Extinction coefficient
  double exposure_time;      // Exposure time, unit: second

  // Validate parameters
  [[nodiscard]] bool isValid() const noexcept {
    return wavelength > 0 && aperture > 0 && obstruction >= 0 &&
           filter_width > 0 && transmissivity > 0 && transmissivity <= 1 &&
           gain > 0 && quantum_efficiency > 0 && quantum_efficiency <= 1 &&
           extinction >= 0 && extinction < 1 && exposure_time > 0;
  }
};

struct OptimizationParams {
  bool use_gpu{false};      // Whether to use GPU acceleration
  bool use_parallel{false}; // Whether to use parallel processing
  int num_threads{4};       // Number of parallel processing threads
  bool use_cache{false};    // Whether to use cache
  size_t cache_size{1024};  // Cache size (MB)
  bool use_simd{false};     // Whether to use SIMD instructions

  // Validate parameters
  [[nodiscard]] bool isValid() const noexcept {
    return num_threads > 0 && cache_size > 0;
  }
};