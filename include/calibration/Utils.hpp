#pragma once

#include <opencv2/opencv.hpp>

namespace utils {

// SIMD optimization detection
bool hasSIMDSupport() {
#ifdef __AVX2__
  return true;
#else
  return false;
#endif
}

} // namespace utils