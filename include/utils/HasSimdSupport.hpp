#pragma once

#include <opencv2/opencv.hpp>

namespace utils {

// SIMD optimization detection
inline bool hasSIMDSupport() {
#if defined(__AVX512F__)
  return true;
#elif defined(__AVX2__)
  return true;
#elif defined(__AVX__)
  return true;
#elif defined(__SSE4_2__)
  return true;
#elif defined(__SSE4_1__)
  return true;
#elif defined(__SSSE3__)
  return true;
#elif defined(__SSE3__)
  return true;
#elif defined(__SSE2__)
  return true;
#elif defined(__SSE__)
  return true;
#else
  return false;
#endif
}

} // namespace utils