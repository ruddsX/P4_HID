/**
 * abc_config.h -- ABCurves Planner constants and types for C++.
 *
 * These values match the exported planner_config.json from the Python
 * checkpoint. They are the minimum needed to drive the TCN + ProDMP decode.
 */
#pragma once

#define _USE_MATH_DEFINES
#include <cstdint>
#include <cmath>
#include <array>
#include <vector>
#include <string>

namespace abc {

// --- Model geometry (must match the exported checkpoint) ---
constexpr int HEADS          = 16;
constexpr int HORIZON        = 1000;   // max B->C duration in ms
constexpr int PREFIX_LEN     = 160;    // encoder window length
constexpr int HIDDEN         = 96;     // TCN hidden width
constexpr int BLOCKS         = 3;      // TCN residual blocks
constexpr int KERNEL_SIZE    = 5;      // causal conv kernel
constexpr int SUMMARY_DIM    = 62;     // summary feature count
constexpr int N_BASIS        = 20;     // ProDMP RBF forcing weights per DoF
constexpr int N_WEIGHTS      = N_BASIS + 1;  // +1 for goal coefficient
constexpr int OUT_DIM        = N_WEIGHTS * 2 + 1;  // 20*2 + 1*2 + 1 = 43
constexpr int DOF            = 2;      // dx, dy
constexpr int OFFSET_RADIUS  = 5;      // Renderer offset radius

// --- ProDMP parameters ---
constexpr double PRODMP_ALPHA       = 25.0;
constexpr double PRODMP_ALPHA_PHASE = 3.0;
constexpr double PRODMP_RIDGE       = 1e-3;

// --- Onset detection defaults ---
constexpr int    ONSET_NOISE_WINDOW_MS    = 24;
constexpr double ONSET_THRESHOLD_MAD_MULT = 6.0;
constexpr int    ONSET_CONSECUTIVE_TICKS  = 12;
constexpr int    ONSET_BACKTRACK_MS       = 4;
constexpr double ONSET_ALIGNMENT_MIN      = 0.3;
constexpr double ONSET_SPEED_FLOOR        = 0.5;

// --- B-trigger defaults ---
constexpr double B_THRESHOLD              = 0.8;
constexpr int    B_MAX_AB_MS              = 1500;
constexpr double B_MIN_REMAINING_COUNTS   = 8.0;
constexpr double B_PROGRESS_REGRESSION_TH = 0.18;
constexpr double B_MAX_CENTER_PROGRESS    = 0.92;

// --- Basis grid resolution ---
constexpr int BASIS_GRID_POINTS = 2001;

// --- Receptive field: the TCN only needs the last 25 ticks ---
// 3 blocks * 2 convs * (5-1) = 24, plus 1 = 25
constexpr int RECEPTIVE_FIELD = 1 + BLOCKS * 2 * (KERNEL_SIZE - 1);

// --- Types ---
struct Vec2 {
    double x, y;
    Vec2() : x(0), y(0) {}
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    double norm() const { return std::sqrt(x * x + y * y); }
    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
};

struct Intent {
    std::vector<float> smooth_dxdy;  // [duration, 2] float32
    std::vector<float> mask;         // [duration] float32
    int duration_ms;
    int head;
};

struct OnsetEvent {
    int index;
    double threshold;
    double speed_median;
    double speed_mad;
};

struct BFire {
    int t_ms;
    double progress_edge;
    double progress_center;
    Vec2 movement;
    Vec2 target_rel;
    double target_radius;
    double remaining;
};

// --- Config loaded from JSON ---
struct PlannerConfig {
    int heads = HEADS;
    int horizon = HORIZON;
    int prefix_len = PREFIX_LEN;
    int hidden = HIDDEN;
    int blocks = BLOCKS;
    int summary_dim = SUMMARY_DIM;

    // ProDMP
    int n_basis = N_BASIS;
    int n_weights = N_BASIS + 1;  // n_basis + 1 goal per axis
    double alpha = PRODMP_ALPHA;
    double alpha_phase = PRODMP_ALPHA_PHASE;
    double ridge = PRODMP_RIDGE;

    // Normalizer arrays (loaded from .bin files)
    std::vector<float> prefix_mean;   // [2]
    std::vector<float> prefix_std;    // [2]
    std::vector<float> summary_mean;  // [summary_dim]
    std::vector<float> summary_std;   // [summary_dim]
    std::vector<float> y_mean;        // [out_dim]
    std::vector<float> y_std;         // [out_dim]

    // Summary feature names (for validation)
    std::vector<std::string> summary_feature_names;

    // Seam contract
    double onset_alignment_min = ONSET_ALIGNMENT_MIN;
    double onset_speed_floor = ONSET_SPEED_FLOOR;
    int onset_noise_window_ms = ONSET_NOISE_WINDOW_MS;
    double onset_threshold_mad_mult = ONSET_THRESHOLD_MAD_MULT;
    int onset_consecutive_ticks = ONSET_CONSECUTIVE_TICKS;
    int onset_backtrack_ms = ONSET_BACKTRACK_MS;

    double b_threshold = B_THRESHOLD;
    int b_max_ab_ms = B_MAX_AB_MS;
    double b_min_remaining_counts = B_MIN_REMAINING_COUNTS;
    double b_progress_regression_th = B_PROGRESS_REGRESSION_TH;
    double b_max_center_progress = B_MAX_CENTER_PROGRESS;
};

}  // namespace abc
