#include "Calibration.hpp"

int main() {
  // 基本图像校正示例
  cv::Mat image = cv::imread("test.jpg");
  cv::Mat dark = cv::imread("dark.jpg");

  cv::InputArray imgArray = image;
  cv::InputArray darkArray = dark;

  // 应用暗场校正
  cv::Mat corrected = apply_dark_frame_subtraction(imgArray, darkArray);

  // 通量校准示例
  CalibrationParams params;
  params.wavelength = 550;     // 550 nm
  params.aperture = 100;       // 100 mm
  params.obstruction = 25;     // 25 mm
  params.filter_width = 10;    // 10 nm
  params.transmissivity = 0.8; // 80%
  params.gain = 2.0;
  params.quantum_efficiency = 0.5;
  params.extinction = 0.1;
  params.exposure_time = 0.5; // 0.5 秒

  auto result = flux_calibration_ex(image, params);
  if (result) {
    cv::imwrite("calibrated.jpg", result->image);
  }

  // 摄像机标定示例
  CameraCalibrator calibrator;
  calibrator.processImages({"calib1.jpg", "calib2.jpg", "calib3.jpg"});
  auto calibResult = calibrator.calibrate();
  if (calibResult) {
    calibrator.generateReport();
    calibrator.saveCalibrationVisualization();
  }

  return 0;
}