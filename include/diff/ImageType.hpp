#pragma once

#include "Parameter.hpp"
#include <memory>
#include <string>

/**
 * @brief 表示图像的类，替代QImage
 */
class Image {
public:
  enum Format { Invalid, ARGB32, RGB32, Grayscale8 };

  Image();
  Image(int width, int height, Format format = ARGB32);
  Image(const Image &other);
  Image(Image &&other) noexcept;
  ~Image();

  Image &operator=(const Image &other);
  Image &operator=(Image &&other) noexcept;

  // 构造和转换方法
  bool load(const std::string &filename);
  bool save(const std::string &filename) const;
  Image scaled(int width, int height, bool smooth = true) const;
  Image convertToFormat(Format format) const;

  // 像素操作
  RGB pixelAt(int x, int y) const;
  void setPixelAt(int x, int y, const RGB &color);
  unsigned char *scanLine(int y);
  const unsigned char *scanLine(int y) const;
  void fill(const RGB &color);

  // 画图功能
  void drawLine(int x1, int y1, int x2, int y2, const RGB &color,
                int width = 1);
  void drawRect(const Rectangle &rect, const RGB &color);
  void fillRect(const Rectangle &rect, const RGB &color);
  void drawText(int x, int y, const std::string &text, const RGB &color);

  // 属性访问
  int width() const noexcept;
  int height() const noexcept;
  Format format() const noexcept;
  bool isNull() const noexcept;
  size_t sizeInBytes() const noexcept;
  unsigned char *bits();
  const unsigned char *bits() const;
  Rectangle rect() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> pImpl;
};