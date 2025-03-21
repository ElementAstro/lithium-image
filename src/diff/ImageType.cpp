#include "diff/ImageType.hpp"

#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>

struct Image::Impl {
  int width = 0;
  int height = 0;
  Format format = Invalid;
  std::vector<unsigned char> data;
  int bytesPerLine = 0;
  int channels = 4; // 默认ARGB32

  void allocate(int w, int h, Format fmt) {
    width = w;
    height = h;
    format = fmt;

    switch (format) {
    case ARGB32:
    case RGB32:
      channels = 4;
      break;
    case Grayscale8:
      channels = 1;
      break;
    default:
      channels = 0;
      break;
    }

    bytesPerLine = width * channels;
    data.resize(height * bytesPerLine);
  }

  bool isValid() const {
    return width > 0 && height > 0 && format != Invalid && !data.empty();
  }
};

Image::Image() : pImpl(std::make_unique<Impl>()) {}

Image::Image(int width, int height, Format format)
    : pImpl(std::make_unique<Impl>()) {
  pImpl->allocate(width, height, format);
}

Image::Image(const Image &other) : pImpl(std::make_unique<Impl>()) {
  pImpl->width = other.pImpl->width;
  pImpl->height = other.pImpl->height;
  pImpl->format = other.pImpl->format;
  pImpl->data = other.pImpl->data;
  pImpl->bytesPerLine = other.pImpl->bytesPerLine;
  pImpl->channels = other.pImpl->channels;
}

Image::Image(Image &&other) noexcept : pImpl(std::move(other.pImpl)) {
  other.pImpl = std::make_unique<Impl>();
}

Image::~Image() = default;

Image &Image::operator=(const Image &other) {
  if (this != &other) {
    pImpl->width = other.pImpl->width;
    pImpl->height = other.pImpl->height;
    pImpl->format = other.pImpl->format;
    pImpl->data = other.pImpl->data;
    pImpl->bytesPerLine = other.pImpl->bytesPerLine;
    pImpl->channels = other.pImpl->channels;
  }
  return *this;
}

Image &Image::operator=(Image &&other) noexcept {
  if (this != &other) {
    pImpl = std::move(other.pImpl);
    other.pImpl = std::make_unique<Impl>();
  }
  return *this;
}

bool Image::load(const std::string &filename) {
  cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
  if (img.empty())
    return false;

  Format format;
  int channels = img.channels();
  
  if (channels == 4)
    format = ARGB32;
  else if (channels == 3)
    format = RGB32;
  else if (channels == 1)
    format = Grayscale8;
  else {
    return false;
  }

  pImpl->allocate(img.cols, img.rows, format);
  
  // OpenCV 图像数据是连续存储的
  if (img.isContinuous()) {
    std::memcpy(pImpl->data.data(), img.data, img.total() * channels);
  } else {
    // 如果不是连续存储，则逐行复制
    for (int i = 0; i < img.rows; ++i) {
      std::memcpy(pImpl->data.data() + i * pImpl->bytesPerLine,
                 img.ptr<unsigned char>(i), img.cols * channels);
    }
  }
  
  return true;
}

bool Image::save(const std::string &filename) const {
  if (isNull())
    return false;

  // 创建OpenCV矩阵
  cv::Mat img(height(), width(), 
              pImpl->channels == 1 ? CV_8UC1 : 
              pImpl->channels == 3 ? CV_8UC3 : CV_8UC4, 
              const_cast<unsigned char*>(bits()));

  // 确定文件扩展名
  auto pos = filename.find_last_of('.');
  if (pos == std::string::npos)
    return false;
  
  std::vector<int> compression_params;
  std::string ext = filename.substr(pos + 1);
  if (ext == "jpg" || ext == "jpeg") {
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(90);
  } else if (ext == "png") {
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
  }

  try {
    return cv::imwrite(filename, img, compression_params);
  } catch (const cv::Exception&) {
    return false;
  }
}

Image Image::scaled(int width, int height, bool /* smooth */) const {
  if (isNull() || width <= 0 || height <= 0)
    return Image();

  // 创建源OpenCV矩阵
  cv::Mat src(this->height(), this->width(), 
              pImpl->channels == 1 ? CV_8UC1 : 
              pImpl->channels == 3 ? CV_8UC3 : CV_8UC4, 
              const_cast<unsigned char*>(bits()));
  
  // 缩放
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  
  // 创建结果图像
  Image result(width, height, format());
  
  // 复制数据
  if (dst.isContinuous()) {
    std::memcpy(result.bits(), dst.data, dst.total() * pImpl->channels);
  } else {
    for (int i = 0; i < dst.rows; ++i) {
      std::memcpy(result.bits() + i * result.pImpl->bytesPerLine,
                 dst.ptr<unsigned char>(i), dst.cols * pImpl->channels);
    }
  }

  return result;
}

Image Image::convertToFormat(Format format) const {
  if (isNull() || this->format() == format)
    return *this;

  Image result(width(), height(), format);

  for (int y = 0; y < height(); y++) {
    for (int x = 0; x < width(); x++) {
      RGB pixel = pixelAt(x, y);

      if (format == Grayscale8) {
        unsigned char gray = ColorUtils::gray(pixel);
        result.setPixelAt(x, y, RGB(gray, gray, gray));
      } else {
        result.setPixelAt(x, y, pixel);
      }
    }
  }

  return result;
}

RGB Image::pixelAt(int x, int y) const {
  if (x < 0 || x >= width() || y < 0 || y >= height() || isNull())
    return RGB();

  const unsigned char *line = scanLine(y);
  switch (format()) {
  case ARGB32:
  case RGB32: {
    const unsigned char *pixel = line + x * 4;
    return RGB(pixel[0], pixel[1], pixel[2], pixel[3]);
  }
  case Grayscale8: {
    unsigned char gray = line[x];
    return RGB(gray, gray, gray);
  }
  default:
    return RGB();
  }
}

void Image::setPixelAt(int x, int y, const RGB &color) {
  if (x < 0 || x >= width() || y < 0 || y >= height() || isNull())
    return;

  unsigned char *line = scanLine(y);
  switch (format()) {
  case ARGB32:
  case RGB32: {
    unsigned char *pixel = line + x * 4;
    pixel[0] = color.r;
    pixel[1] = color.g;
    pixel[2] = color.b;
    pixel[3] = color.a;
    break;
  }
  case Grayscale8: {
    line[x] = ColorUtils::gray(color);
    break;
  }
  default:
    break;
  }
}

unsigned char *Image::scanLine(int y) {
  if (y < 0 || y >= height() || isNull())
    return nullptr;
  return bits() + y * pImpl->bytesPerLine;
}

const unsigned char *Image::scanLine(int y) const {
  if (y < 0 || y >= height() || isNull())
    return nullptr;
  return bits() + y * pImpl->bytesPerLine;
}

void Image::fill(const RGB &color) {
  if (isNull())
    return;

  for (int y = 0; y < height(); y++) {
    for (int x = 0; x < width(); x++) {
      setPixelAt(x, y, color);
    }
  }
}

void Image::drawLine(int x1, int y1, int x2, int y2, const RGB &color,
                     int /* width */) {
  // 简单的Bresenham算法实现
  int dx = std::abs(x2 - x1);
  int dy = std::abs(y2 - y1);
  int sx = (x1 < x2) ? 1 : -1;
  int sy = (y1 < y2) ? 1 : -1;
  int err = dx - dy;

  while (true) {
    setPixelAt(x1, y1, color);

    if (x1 == x2 && y1 == y2)
      break;
    int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x1 += sx;
    }
    if (e2 < dx) {
      err += dx;
      y1 += sy;
    }
  }
}

void Image::drawRect(const Rectangle &rect, const RGB &color) {
  drawLine(rect.x, rect.y, rect.x + rect.width, rect.y, color);
  drawLine(rect.x + rect.width, rect.y, rect.x + rect.width,
           rect.y + rect.height, color);
  drawLine(rect.x + rect.width, rect.y + rect.height, rect.x,
           rect.y + rect.height, color);
  drawLine(rect.x, rect.y + rect.height, rect.x, rect.y, color);
}

void Image::fillRect(const Rectangle &rect, const RGB &color) {
  for (int y = rect.y; y < rect.y + rect.height; y++) {
    for (int x = rect.x; x < rect.x + rect.width; x++) {
      if (x >= 0 && x < width() && y >= 0 && y < height()) {
        setPixelAt(x, y, color);
      }
    }
  }
}

void Image::drawText(int x, int y, const std::string &text, const RGB &color) {
  // 简化版本的文字绘制 - 在实际应用中可能需要使用freetype等库
  // 这里只是一个简单的占位实现
  for (size_t i = 0; i < text.length(); i++) {
    fillRect(Rectangle(x + i * 8, y, 6, 10), color);
  }
}

int Image::width() const noexcept { return pImpl->width; }

int Image::height() const noexcept { return pImpl->height; }

Image::Format Image::format() const noexcept { return pImpl->format; }

bool Image::isNull() const noexcept { return !pImpl->isValid(); }

size_t Image::sizeInBytes() const noexcept { return pImpl->data.size(); }

unsigned char *Image::bits() { return pImpl->data.data(); }

const unsigned char *Image::bits() const { return pImpl->data.data(); }

Rectangle Image::rect() const noexcept {
  return Rectangle(0, 0, width(), height());
}