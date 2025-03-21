#pragma once

#include <cmath>

/**
 * @brief 表示矩形区域的结构体，替代QRect
 */
struct Rectangle {
  int x;
  int y;
  int width;
  int height;

  Rectangle() : x(0), y(0), width(0), height(0) {}
  Rectangle(int x, int y, int width, int height)
      : x(x), y(y), width(width), height(height) {}

  bool isNull() const noexcept { return width <= 0 || height <= 0; }

  Rectangle united(const Rectangle &other) const noexcept {
    int newX = std::min(x, other.x);
    int newY = std::min(y, other.y);
    int newRight = std::max(x + width, other.x + other.width);
    int newBottom = std::max(y + height, other.y + other.height);
    return Rectangle(newX, newY, newRight - newX, newBottom - newY);
  }

  bool intersects(const Rectangle &other) const noexcept {
    return !(x >= other.x + other.width || x + width <= other.x ||
             y >= other.y + other.height || y + height <= other.y);
  }

  Rectangle adjusted(int dx1, int dy1, int dx2, int dy2) const noexcept {
    return Rectangle(x + dx1, y + dy1, width + dx2 - dx1, height + dy2 - dy1);
  }
};

/**
 * @brief RGB颜色结构体，替代QRgb
 */
struct RGB {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char a;

  RGB(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255)
      : r(r), g(g), b(b), a(a) {}

  RGB() : r(0), g(0), b(0), a(0) {}
};

/**
 * @brief 颜色操作工具函数
 */
namespace ColorUtils {
inline RGB makeRGB(unsigned char r, unsigned char g, unsigned char b,
                   unsigned char a = 255) {
  return RGB(r, g, b, a);
}

inline unsigned char gray(const RGB &color) {
  return static_cast<unsigned char>(0.299 * color.r + 0.587 * color.g +
                                    0.114 * color.b);
}
} // namespace ColorUtils