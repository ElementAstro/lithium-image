// ProcessError.hpp
#pragma once

#include <string>

/**
 * @struct ProcessError
 * @brief 处理操作的错误代码和描述
 */
struct ProcessError {
  enum class Code {
    INVALID_INPUT,
    INVALID_CONFIG,
    PROCESSING_FAILED,
    OUT_OF_MEMORY,
    UNSUPPORTED_OPERATION
  };

  Code code;
  std::string message;
};