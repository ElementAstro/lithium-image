#pragma once

#include "Core.hpp"
#include <span>

/**
 * @brief 图像比较策略的基类
 */
class ComparisonStrategyBase {
protected:
    static constexpr int BLOCK_SIZE = 16;
    static constexpr int SUBSAMPLE_FACTOR = 2;

    [[nodiscard]] Image preprocessImage(const Image& img) const;
    void compareBlockSIMD(std::span<const unsigned char> block1,
                         std::span<const unsigned char> block2,
                         std::span<unsigned char> dest) const noexcept;
    [[nodiscard]] std::vector<Rectangle> findDifferenceRegions(const Image& diffImg) const;

    class DisjointSet {
        std::vector<int> parent;
        std::vector<int> rank;

    public:
        explicit DisjointSet(int size);
        [[nodiscard]] int find(int x) noexcept;
        void unite(int x, int y) noexcept;
    };
};

/**
 * @brief 像素差异比较策略类
 */
class PixelDifferenceStrategy : public ComparisonStrategyBase {
public:
    /**
     * @brief 比较两个图像并找出像素差异
     * 
     * @param img1 第一个图像
     * @param img2 第二个图像
     * @param promise 用于报告比较结果的promise
     * @return 比较的结果
     * @throws std::runtime_error 如果比较失败
     */
    [[nodiscard]] ComparisonResult compare(const Image& img1, const Image& img2,
                                          Promise<ComparisonResult>& promise) const;

    /**
     * @brief 获取策略名称
     * 
     * @return 策略名称
     */
    [[nodiscard]] std::string name() const noexcept {
        return "Pixel Difference Comparison";
    }
};

/**
 * @brief 结构相似性指数(SSIM)比较策略
 */
class SSIMStrategy : public ComparisonStrategyBase {
public:
    [[nodiscard]] ComparisonResult compare(const Image& img1, const Image& img2,
                                         Promise<ComparisonResult>& promise) const;
    [[nodiscard]] std::string name() const noexcept {
        return "Structural Similarity Comparison";
    }

private:
    static constexpr double K1 = 0.01;
    static constexpr double K2 = 0.03;
    static constexpr int WINDOW_SIZE = 8;

    [[nodiscard]] double computeSSIM(const Image& img1, const Image& img2,
                                   int x, int y) const noexcept;
};

/**
 * @brief 感知哈希(pHash)比较策略
 */
class PerceptualHashStrategy : public ComparisonStrategyBase {
public:
    [[nodiscard]] ComparisonResult compare(const Image& img1, const Image& img2,
                                         Promise<ComparisonResult>& promise) const;
    [[nodiscard]] std::string name() const noexcept {
        return "Perceptual Hash Comparison";
    }

private:
    static constexpr int HASH_SIZE = 64;
    [[nodiscard]] uint64_t computeHash(const Image& img) const;
    [[nodiscard]] int hammingDistance(uint64_t hash1, uint64_t hash2) const noexcept;
};

/**
 * @brief 基于颜色直方图的比较策略
 */
class HistogramStrategy : public ComparisonStrategyBase {
public:
    [[nodiscard]] ComparisonResult compare(const Image& img1, const Image& img2,
                                         Promise<ComparisonResult>& promise) const;
    [[nodiscard]] std::string name() const noexcept {
        return "Color Histogram Comparison";
    }

private:
    static constexpr int HIST_BINS = 256;
    [[nodiscard]] std::vector<int> computeHistogram(const Image& img) const;
    [[nodiscard]] double compareHistograms(const std::vector<int>& hist1,
                                         const std::vector<int>& hist2) const noexcept;
};