#include "ImageCorrection.hpp"
#include "Logging.hpp"
#include "OptimizedCorrection.hpp"
#include <cmath>

cv::Mat instrument_response_correction(cv::InputArray &image,
                                       cv::InputArray &response_function) {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying instrument response correction.");

  try {
    cv::Mat img = image.getMat();
    cv::Mat resp = response_function.getMat();

    if (img.size() != resp.size()) {
      logger->error("Image and response function shapes do not match: {} vs {}",
                    img.size(), resp.size());
      throw InvalidParameterError(
          "Image and response function must have the same size.");
    }

    if (img.type() != resp.type()) {
      logger->warn(
          "Image and response function types do not match. Converting...");
      cv::Mat converted;
      resp.convertTo(converted, img.type());
      resp = converted;
    }

    cv::Mat corrected;
    cv::multiply(img, resp, corrected);
    logger->info("Instrument response correction applied successfully.");
    return corrected;
  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during response correction: {}", e.what());
    throw ProcessingError(
        std::string("OpenCV error during response correction: ") + e.what());
  } catch (const std::exception &e) {
    logger->error("Error during instrument response correction: {}", e.what());
    throw;
  }
}

cv::Mat background_noise_correction(cv::InputArray &image) noexcept {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying background noise correction.");

  try {
    double medianValue = cv::mean(image)[0];
    cv::Mat imgMat = image.getMat();
    cv::Mat corrected = imgMat - medianValue;
    logger->info("Background noise correction applied with median value: {}",
                 medianValue);
    return corrected;
  } catch (const std::exception &e) {
    logger->error("Error in background noise correction: {}", e.what());
    return image.getMat().clone(); // Return original image in case of failure
  }
}

cv::Mat apply_flat_field_correction(cv::InputArray &image,
                                    cv::InputArray &flat_field) {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying flat-field correction.");

  try {
    cv::Mat img = image.getMat();
    cv::Mat flat = flat_field.getMat();

    if (img.size() != flat.size()) {
      logger->error("Image and flat-field image shapes do not match: {} vs {}",
                    img.size(), flat.size());
      throw InvalidParameterError(
          "Image and flat-field image must have the same size.");
    }

    // Check for division by zero
    double minVal;
    cv::minMaxLoc(flat, &minVal);
    if (std::abs(minVal) < 1e-10) {
      logger->warn(
          "Very small values detected in flat field. Applying threshold.");
      cv::threshold(flat, flat, 1e-10, 1e-10, cv::THRESH_TOZERO_INV);
      flat += 1e-10;
    }

    cv::Mat corrected;
    cv::divide(img, flat, corrected);
    logger->info("Flat-field correction applied successfully.");
    return corrected;
  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during flat field correction: {}", e.what());
    throw ProcessingError(std::string("OpenCV error: ") + e.what());
  } catch (const std::exception &e) {
    logger->error("Error during flat field correction: {}", e.what());
    throw;
  }
}

cv::Mat apply_dark_frame_subtraction(cv::InputArray &image,
                                     cv::InputArray &dark_frame) {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying dark frame subtraction.");

  try {
    cv::Mat img = image.getMat();
    cv::Mat dark = dark_frame.getMat();

    if (img.size() != dark.size()) {
        logger->error("Image and dark frame image shapes do not match: {} vs {}",
                      img.size(), dark.size());
        throw InvalidParameterError(
            "Image and dark frame image must have the same size.");
      }
  
      if (img.type() != dark.type()) {
        logger->warn("Image and dark frame types do not match. Converting...");
        cv::Mat converted;
        dark.convertTo(converted, img.type());
        dark = converted;
      }
  
      cv::Mat corrected = img - dark;
      logger->info("Dark frame subtraction applied successfully.");
      return corrected;
    } catch (const cv::Exception &e) {
      logger->error("OpenCV error during dark frame subtraction: {}", e.what());
      throw ProcessingError(std::string("OpenCV error: ") + e.what());
    } catch (const std::exception &e) {
      logger->error("Error during dark frame subtraction: {}", e.what());
      throw;
    }
  }
  
  double compute_flx2dn(const CalibrationParams &params) {
    const auto &logger = Logger::getInstance();
    logger->debug("Starting FLX2DN computation.");
  
    if (!params.isValid()) {
      logger->error("Invalid calibration parameters");
      throw InvalidParameterError(
          "Invalid calibration parameters for FLX2DN computation");
    }
  
    try {
      const double c = 3.0e8;     // Speed of light, unit: m/s
      const double h = 6.626e-34; // Planck constant, unit: J·s
      double wavelength_m = params.wavelength * 1e-9; // Convert nm to m
  
      // Check for division by zero or negative values
      if (wavelength_m <= 0) {
        logger->error("Wavelength must be positive");
        throw InvalidParameterError("Wavelength must be positive");
      }
  
      if (c <= 0 || h <= 0) {
        logger->error("Physical constants must be positive");
        throw InvalidParameterError("Physical constants must be positive");
      }
  
      double aperture_area = M_PI * ((params.aperture * params.aperture -
                                      params.obstruction * params.obstruction) /
                                     4.0);
  
      if (aperture_area <= 0) {
        logger->warn("Calculated aperture area is not positive. Check aperture "
                     "and obstruction values.");
        throw InvalidParameterError("Calculated aperture area must be positive");
      }
  
      double FLX2DN = params.exposure_time * aperture_area * params.filter_width *
                      params.transmissivity * params.gain *
                      params.quantum_efficiency * (1 - params.extinction) *
                      (wavelength_m / (c * h));
  
      logger->info("Computed FLX2DN: {}", FLX2DN);
      return FLX2DN;
    } catch (const std::exception &e) {
      logger->error("Error computing FLX2DN: {}", e.what());
      throw;
    }
  }
  
  std::optional<FluxCalibrationResult>
  flux_calibration_ex(const cv::Mat &image, const CalibrationParams &params,
                      const cv::Mat *response_function,
                      const cv::Mat *flat_field,
                      const cv::Mat *dark_frame,
                      bool enable_optimization) {
    const auto &logger = Logger::getInstance();
    logger->debug("Starting flux calibration process.");
  
    if (image.empty()) {
      logger->error("Input image is empty");
      return std::nullopt;
    }
  
    if (!params.isValid()) {
      logger->error("Invalid calibration parameters");
      return std::nullopt;
    }
  
    OptimizationParams optParams;
    optParams.use_gpu = enable_optimization;
    optParams.use_parallel = enable_optimization;
    optParams.use_simd = enable_optimization && utils::hasSIMDSupport();
    optParams.use_cache = enable_optimization;
  
    try {
      // Start performance measurement
      auto start = std::chrono::high_resolution_clock::now();
  
      cv::Mat img;
      if (optParams.use_gpu && cv::ocl::haveOpenCL()) {
        cv::UMat uimg = image.getUMat(cv::ACCESS_READ);
        img = uimg.getMat(cv::ACCESS_READ);
      } else {
        img = image.clone();
      }
  
      // Validate input images
      if (response_function != nullptr && response_function->empty()) {
        logger->warn("Empty response function provided");
        response_function = nullptr;
      }
  
      if (flat_field != nullptr && flat_field->empty()) {
        logger->warn("Empty flat field provided");
        flat_field = nullptr;
      }
  
      if (dark_frame != nullptr && dark_frame->empty()) {
        logger->warn("Empty dark frame provided");
        dark_frame = nullptr;
      }
  
      // Apply corrections
      if (response_function != nullptr) {
        cv::InputArray imgArray = img;
        cv::InputArray respArray = *response_function;
        if (enable_optimization) {
          img = instrument_response_correction_optimized(imgArray, respArray, optParams);
        } else {
          img = instrument_response_correction(imgArray, respArray);
        }
      }
  
      if (flat_field != nullptr) {
        cv::InputArray imgArray = img;
        cv::InputArray flatArray = *flat_field;
        img = apply_flat_field_correction(imgArray, flatArray);
      }
  
      if (dark_frame != nullptr) {
        cv::InputArray imgArray = img;
        cv::InputArray darkArray = *dark_frame;
        img = apply_dark_frame_subtraction(imgArray, darkArray);
      }
  
      // Calculate flux-to-DN conversion factor
      double FLX2DN = compute_flx2dn(params);
      if (FLX2DN <= 0) {
        logger->error("Invalid FLX2DN value: {}", FLX2DN);
        return std::nullopt;
      }
  
      // Apply flux calibration
      cv::Mat calibrated;
      if (optParams.use_gpu && cv::ocl::haveOpenCL()) {
        cv::UMat uimg = img.getUMat(cv::ACCESS_READ);
        cv::UMat ucalibrated;
        cv::divide(uimg, FLX2DN, ucalibrated);
        calibrated = ucalibrated.getMat(cv::ACCESS_READ);
      } else {
        cv::divide(img, FLX2DN, calibrated);
      }
  
      // Apply background noise correction
      cv::InputArray calibratedArray = calibrated;
      if (enable_optimization) {
        calibrated = background_noise_correction_optimized(calibratedArray, optParams);
      } else {
        calibrated = background_noise_correction(calibratedArray);
      }
      logger->debug("Applied background noise correction.");
  
      // Normalize calibrated image to [0,1] range
      double minVal, maxVal;
      cv::minMaxLoc(calibrated, &minVal, &maxVal);
      double FLXMIN = minVal;
      double FLXRANGE = maxVal - minVal;
      cv::Mat rescaled;
  
      if (FLXRANGE > 0) {
        cv::normalize(calibrated, rescaled, 0, 1, cv::NORM_MINMAX, CV_32F);
        logger->info("Rescaled calibrated image to [0, 1] range.");
      } else {
        logger->warn("Zero range detected in calibrated image. Skipping rescaling.");
        rescaled = calibrated.clone();
      }
  
      // Measure performance
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      logger->info("Flux calibration completed in {} seconds", elapsed.count());
  
      // Return the result
      FluxCalibrationResult result;
      result.image = rescaled;
      result.min_value = FLXMIN;
      result.range_value = FLXRANGE;
      result.flx2dn_factor = FLX2DN;
  
      return result;
  
    } catch (const cv::Exception &e) {
      logger->error("OpenCV error during flux calibration: {}", e.what());
      return std::nullopt;
    } catch (const std::exception &e) {
      logger->error("Error during flux calibration: {}", e.what());
      return std::nullopt;
    } catch (...) {
      logger->error("Unknown error during flux calibration");
      return std::nullopt;
    }
  }