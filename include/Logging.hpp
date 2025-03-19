#pragma once

#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

// Thread-safe singleton logger pattern
class Logger {
public:
  static std::shared_ptr<spdlog::logger> getInstance() {
    static std::once_flag flag;
    static std::shared_ptr<spdlog::logger> instance;

    std::call_once(flag, []() {
      try {
        std::filesystem::create_directories("logs");
        instance = spdlog::basic_logger_mt("CalibrationLogger",
                                           "logs/calibration.log");
        instance->set_level(spdlog::level::debug);
        instance->flush_on(spdlog::level::warn);
      } catch (const spdlog::spdlog_ex &ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        instance = spdlog::null_logger_mt("null_calibration_logger");
      }
    });

    return instance;
  }
};