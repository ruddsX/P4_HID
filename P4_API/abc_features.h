/**
 * abc_features.h -- 62 causal summary features for the ABCurves Planner.
 *
 * Direct port of abcurves/fast_summary.py _raw_summary62().
 * All features are computed from the A->B prefix and target geometry at B.
 * Nothing peeks at the future.
 */
#pragma once

#include "abc_config.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace abc {

namespace detail {

inline double norm2(double x, double y) { return std::sqrt(x * x + y * y); }
inline float norm2f(float x, float y) { return std::sqrt(x * x + y * y); }

inline double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

// Linear slope (least-squares) over values[0..length-1]
inline double linear_slope(const double* values, int length) {
    if (length <= 1) return 0.0;
    double mean = 0;
    for (int i = 0; i < length; ++i) mean += values[i];
    mean /= (double)length;
    double x_mean = 0.5 * (double)(length - 1);
    double denom = 0, numer = 0;
    for (int i = 0; i < length; ++i) {
        double x = (double)i - x_mean;
        denom += x * x;
        numer += x * (values[i] - mean);
    }
    return (denom <= 1e-9) ? 0.0 : numer / denom;
}

// Linear slope over float32 values (widened to float64)
inline double slope_f32(const float* values, int length) {
    if (length <= 1) return 0.0;
    std::vector<double> widened(length);
    for (int i = 0; i < length; ++i) widened[i] = (double)values[i];
    return linear_slope(widened.data(), length);
}

}  // namespace detail


/**
 * Compute all 62 canonical summary features.
 *
 * @param prefix     [n, 2] float32 raw dx/dy counts (A->B movement)
 * @param n          number of prefix ticks
 * @param target_x   target x relative to B (count space)
 * @param target_y   target y relative to B (count space)
 * @param radius     target radius
 * @param progress   center-referenced progress at B
 * @param out        [62] output double array
 */
inline void raw_summary62(
    const float* prefix, int n,
    double target_x, double target_y, double radius, double progress,
    double* out)
{
    using namespace detail;
    const double EPS = 1e-6;

    // Target geometry
    double tx = target_x, ty = target_y;
    radius = std::max(radius, EPS);
    double target_norm = norm2(tx, ty);
    double target_distance = std::max(target_norm, EPS);
    double toward_x, toward_y;
    if (target_norm <= EPS) { toward_x = 1; toward_y = 0; }
    else { toward_x = tx / target_norm; toward_y = ty / target_norm; }
    double tangent_x = -toward_y, tangent_y = toward_x;
    double distance_over_radius = target_distance / radius;

    // Prefix kinematics
    std::vector<double> speeds(n);
    double prefix_x = 0, prefix_y = 0, path_length = 0, peak_speed = 0;
    for (int i = 0; i < n; ++i) {
        double x = (double)prefix[i * 2], y = (double)prefix[i * 2 + 1];
        prefix_x += x; prefix_y += y;
        double s = norm2(x, y);
        speeds[i] = s;
        path_length += s;
        if (i == 0 || s > peak_speed) peak_speed = s;
    }
    double prefix_distance = norm2(prefix_x, prefix_y);
    double mean_speed = n ? path_length / (double)n : 0;
    double speed_var = 0;
    for (int i = 0; i < n; ++i) { double d = speeds[i] - mean_speed; speed_var += d * d; }
    double speed_std = n ? std::sqrt(speed_var / (double)n) : 0;

    // Last velocity and acceleration
    double lvx = 0, lvy = 0, speed_at_b = 0, accel_at_b = 0;
    double lax = 0, lay = 0;
    if (n) {
        lvx = (double)prefix[(n - 1) * 2];
        lvy = (double)prefix[(n - 1) * 2 + 1];
        speed_at_b = norm2f(prefix[(n - 1) * 2], prefix[(n - 1) * 2 + 1]);
    }
    if (n >= 2) {
        lax = (double)(prefix[(n - 1) * 2] - prefix[(n - 2) * 2]);
        lay = (double)(prefix[(n - 1) * 2 + 1] - prefix[(n - 2) * 2 + 1]);
        accel_at_b = norm2f((float)lax, (float)lay);
    }

    // Causal context (recent window)
    double inside = (target_norm <= radius) ? 1.0 : 0.0;
    double velocity_toward = 0, velocity_lateral = 0;
    double recent_zero = 0, recent_sign_flip = 0, recent_dir_change = 0;
    double recent_speed_slope = 0, recent_accel_slope = 0;

    if (n) {
        double ctx_tx = (target_norm > 1e-9) ? tx / target_norm : 1.0;
        double ctx_ty = (target_norm > 1e-9) ? ty / target_norm : 0.0;
        velocity_toward = lvx * ctx_tx + lvy * ctx_ty;
        velocity_lateral = lvx * ctx_ty - lvy * ctx_tx;

        int rlen = std::min(40, n);
        int start = n - rlen;
        std::vector<float> recent_mag(rlen), recent_accel_mag(rlen);
        int zeros = 0;
        for (int j = 0; j < rlen; ++j) {
            int i = start + j;
            float mx = prefix[i * 2], my = prefix[i * 2 + 1];
            float mag = norm2f(mx, my);
            recent_mag[j] = mag;
            if (mag <= 0) ++zeros;
            float ax = j ? prefix[i * 2] - prefix[(i - 1) * 2] : 0;
            float ay = j ? prefix[i * 2 + 1] - prefix[(i - 1) * 2 + 1] : 0;
            recent_accel_mag[j] = norm2f(ax, ay);
        }
        recent_zero = (double)zeros / (double)rlen;
        recent_speed_slope = slope_f32(recent_mag.data(), rlen);
        recent_accel_slope = slope_f32(recent_accel_mag.data(), rlen);

        // Sign flip rate
        int flips = 0, flip_denom = 0;
        for (int axis = 0; axis < 2; ++axis) {
            bool have_prev = false;
            int prev_sign = 0;
            for (int i = start; i < n; ++i) {
                float val = prefix[i * 2 + axis];
                if (std::abs(val) > 0) {
                    int sign = (val > 0) ? 1 : -1;
                    if (have_prev) { ++flip_denom; if (sign != prev_sign) ++flips; }
                    prev_sign = sign; have_prev = true;
                }
            }
        }
        recent_sign_flip = flip_denom ? (double)flips / (double)flip_denom : 0;

        // Direction change rate
        if (rlen > 2) {
            int dir_changes = 0, dir_denom = 0;
            bool prev_valid = false;
            double prev_ux = 0, prev_uy = 0;
            for (int i = start; i < n; ++i) {
                float mag = recent_mag[i - start];
                bool valid = mag > 1e-6f;
                if (valid) {
                    double ux = (double)(float)(prefix[i * 2] / mag);
                    double uy = (double)(float)(prefix[i * 2 + 1] / mag);
                    if (prev_valid) {
                        ++dir_denom;
                        if (prev_ux * ux + prev_uy * uy < 0.7071) ++dir_changes;
                    }
                    prev_ux = ux; prev_uy = uy;
                }
                prev_valid = valid;
            }
            recent_dir_change = dir_denom ? (double)dir_changes / (double)dir_denom : 0;
        }
    }

    // Path-relative features
    double crossed = 0, overshot = 0, min_prefix_dist = target_distance;
    int near_count = 0;
    int path_points = n ? n : 1;
    double cx = 0, cy = 0;
    double near_thresh = std::max(radius * 2.0, radius + 4.0);
    double max_axis = -1e18;
    for (int i = 0; i < path_points; ++i) {
        double px, py;
        if (n) {
            cx += (double)prefix[i * 2]; cy += (double)prefix[i * 2 + 1];
            px = cx - prefix_x; py = cy - prefix_y;
        } else { px = 0; py = 0; }
        double dx = tx - px, dy = ty - py;
        double dist = norm2(dx, dy);
        if (i == 0 || dist < min_prefix_dist) min_prefix_dist = dist;
        if (dist <= radius) crossed = 1;
        if (dist <= near_thresh) ++near_count;
        double axis = px * toward_x + py * toward_y;
        if (axis > max_axis) max_axis = axis;
    }
    if (max_axis > target_distance * 1.03) overshot = 1;
    double near_rate = (double)near_count / (double)path_points;

    // Speed drop
    int early_len = std::min(20, n), recent20_len = std::min(20, n);
    double early_mean = 0, recent_mean = 0;
    for (int i = 0; i < early_len; ++i) early_mean += speeds[i];
    for (int i = n - recent20_len; i < n; ++i) recent_mean += speeds[i];
    if (early_len) early_mean /= (double)early_len;
    if (recent20_len) recent_mean /= (double)recent20_len;
    double speed_drop = early_mean - recent_mean;

    // Direction alignment
    double movement_norm = std::max(norm2(lvx, lvy), EPS);
    double alignment_cos = (lvx * toward_x + lvy * toward_y) / movement_norm;
    double alignment_sin = (lvx * tangent_x + lvy * tangent_y) / movement_norm;
    double direction_error;
    if (movement_norm <= EPS && target_norm <= EPS) direction_error = 0;
    else if (movement_norm <= EPS || target_norm <= EPS) direction_error = 90;
    else {
        double cos = (lvx * tx + lvy * ty) / (movement_norm * target_norm);
        cos = clamp(cos, -1, 1);
        direction_error = std::acos(cos) * 180.0 / M_PI;
    }

    // Jerk
    double jerk = 0;
    if (n >= 4) {
        double a0x = (double)prefix[(n - 2) * 2] - (double)prefix[(n - 3) * 2];
        double a0y = (double)prefix[(n - 2) * 2 + 1] - (double)prefix[(n - 3) * 2 + 1];
        double a1x = (double)prefix[(n - 1) * 2] - (double)prefix[(n - 2) * 2];
        double a1y = (double)prefix[(n - 1) * 2 + 1] - (double)prefix[(n - 2) * 2 + 1];
        jerk = norm2(a1x - a0x, a1y - a0y);
    }

    // Projections
    double prefix_target_axis = prefix_x * toward_x + prefix_y * toward_y;
    double prefix_lateral = (prefix_x * tangent_x + prefix_y * tangent_y) / radius;
    double accel_toward = lax * toward_x + lay * toward_y;
    double accel_lateral = lax * tangent_x + lay * tangent_y;

    // Prefix shape (last 16 ticks)
    int shape_start = std::max(0, n - 16);
    double curv_signed_sum = 0, curv_abs_sum = 0;
    std::vector<double> turn_angles(15);
    int turn_count = 0;
    if (n - shape_start >= 2) {
        for (int i = shape_start; i < n - 1; ++i) {
            double x0 = (double)prefix[i * 2], y0 = (double)prefix[i * 2 + 1];
            double x1 = (double)prefix[(i + 1) * 2], y1 = (double)prefix[(i + 1) * 2 + 1];
            double n0 = norm2(x0, y0), n1 = norm2(x1, y1);
            if (n0 > 1e-6 && n1 > 1e-6) {
                double denom = std::max(n0 * n1, 1e-9);
                double sine = (x0 * y1 - y0 * x1) / denom;
                curv_signed_sum += sine;
                curv_abs_sum += std::abs(sine);
                double dot = clamp((x0 * x1 + y0 * y1) / denom, -1, 1);
                turn_angles[turn_count++] = std::acos(dot) * 180.0 / M_PI;
            }
        }
    }
    double curv_signed = turn_count ? curv_signed_sum / (double)turn_count : 0;
    double curv_abs = turn_count ? curv_abs_sum / (double)turn_count : 0;
    double dir_change_mean = 0;
    for (int i = 0; i < turn_count; ++i) dir_change_mean += turn_angles[i];
    if (turn_count) dir_change_mean /= (double)turn_count;
    double dir_change_slope = (turn_count >= 2) ? linear_slope(turn_angles.data(), turn_count) : 0;

    // Approach direction (last 10 ticks)
    double approach_x = 0, approach_y = 0;
    for (int i = std::max(0, n - 10); i < n; ++i) {
        approach_x += (double)prefix[i * 2];
        approach_y += (double)prefix[i * 2 + 1];
    }
    double approach_norm = norm2(approach_x, approach_y);
    double approach_cos = 0, approach_sin = 0;
    if (approach_norm > 1e-6) {
        approach_cos = (approach_x * toward_x + approach_y * toward_y) / approach_norm;
        approach_sin = (approach_x * tangent_x + approach_y * tangent_y) / approach_norm;
    }

    // Speed profile bins (4 bins)
    double speed_bins[4] = {0, 0, 0, 0};
    for (int b = 0; b < 4; ++b) {
        if (n == 0) { speed_bins[b] = 0; continue; }
        int lo = (int)std::floor((double)n * (double)b / 4.0);
        int hi = (int)std::floor((double)n * (double)(b + 1) / 4.0);
        if (hi <= lo) hi = std::min(lo + 1, n);
        if (lo < n) {
            double total = 0;
            for (int i = lo; i < hi; ++i) total += speeds[i];
            speed_bins[b] = total / (double)(hi - lo);
        } else {
            speed_bins[b] = speeds[n - 1];
        }
    }

    // Lateral drift (last 16 ticks)
    double along = 0, lateral = 0;
    for (int i = shape_start; i < n; ++i) {
        double x = (double)prefix[i * 2], y = (double)prefix[i * 2 + 1];
        along += std::abs(x * toward_x + y * toward_y);
        lateral += std::abs(x * tangent_x + y * tangent_y);
    }
    double lateral_drift = n ? lateral / std::max(along + lateral, 1e-6) : 0;

    // Write output in canonical order
    out[0] = (double)n;                           // prefix_duration_ms
    out[1] = prefix_x;                            // prefix_dx
    out[2] = prefix_y;                            // prefix_dy
    out[3] = prefix_distance;                     // prefix_distance
    out[4] = path_length;                         // prefix_path_length
    out[5] = prefix_distance / std::max(path_length, EPS);  // prefix_straightness
    out[6] = prefix_target_axis;                  // prefix_target_axis_progress
    out[7] = prefix_lateral;                      // prefix_lateral_error
    out[8] = tx;                                  // target_rel_x
    out[9] = ty;                                  // target_rel_y
    out[10] = target_distance;                    // target_distance
    out[11] = radius;                             // target_radius
    out[12] = distance_over_radius;               // distance_over_radius
    out[13] = toward_x;                           // target_unit_x
    out[14] = toward_y;                           // target_unit_y
    out[15] = progress;                           // progress_context
    out[16] = inside;                             // inside_target_at_B
    out[17] = (distance_over_radius <= 2.0 || inside > 0.5) ? 1.0 : 0.0;  // near_target_flag
    out[18] = crossed;                            // prefix_crossing_state
    out[19] = overshot;                           // prefix_overshoot_state
    out[20] = min_prefix_dist / radius;           // prefix_min_target_distance_over_radius
    out[21] = near_rate;                          // prefix_near_target_rate
    out[22] = lvx;                                // last_velocity_x
    out[23] = lvy;                                // last_velocity_y
    out[24] = lvx / movement_norm;                // movement_dir_x
    out[25] = lvy / movement_norm;                // movement_dir_y
    out[26] = speed_at_b;                         // speed_at_B
    out[27] = mean_speed;                         // mean_prefix_speed
    out[28] = peak_speed;                         // peak_prefix_speed
    out[29] = speed_std;                          // prefix_speed_std
    out[30] = speed_at_b / target_distance;       // relative_speed_distance
    out[31] = speed_at_b / radius;                // relative_speed_radius
    out[32] = velocity_toward;                    // velocity_toward_target
    out[33] = velocity_lateral;                   // velocity_tangential_to_target
    out[34] = alignment_cos;                      // direction_alignment_cos
    out[35] = alignment_sin;                      // direction_alignment_sin
    out[36] = direction_error;                    // direction_error_deg
    out[37] = accel_at_b;                         // accel_at_B
    out[38] = accel_toward;                       // accel_toward_target
    out[39] = accel_lateral;                      // accel_lateral
    out[40] = recent_speed_slope;                 // recent_speed_slope
    out[41] = recent_accel_slope;                 // recent_accel_slope
    out[42] = (recent_speed_slope < -0.03 || accel_at_b < -0.03) ? 1.0 : 0.0;  // deceleration_indicator
    out[43] = speed_drop;                         // speed_drop_recent
    out[44] = jerk;                               // jerk_at_B
    out[45] = std::abs(speed_at_b - mean_speed);  // discontinuity_speed_jump
    out[46] = recent_zero;                        // recent_zero_rate
    out[47] = recent_sign_flip;                   // recent_sign_flip_rate
    out[48] = recent_dir_change;                  // recent_direction_change_rate
    out[49] = (speed_at_b > 0.5 || recent_mean > 0.5) ? 1.0 : 0.0;  // active_motion_flag
    out[50] = (distance_over_radius <= 1.5 && speed_at_b < std::max(1.0, mean_speed * 0.5)) ? 1.0 : 0.0;  // stabilization_like_flag
    out[51] = curv_signed;                        // prefix_shape_curvature_signed
    out[52] = curv_abs;                           // prefix_shape_curvature_abs
    out[53] = dir_change_mean;                    // prefix_shape_dir_change_mean_deg
    out[54] = dir_change_slope;                   // prefix_shape_dir_change_slope
    out[55] = approach_cos;                       // prefix_shape_approach_cos
    out[56] = approach_sin;                       // prefix_shape_approach_sin
    out[57] = speed_bins[0] / std::max(mean_speed, 1e-6);  // speed_bin_0
    out[58] = speed_bins[1] / std::max(mean_speed, 1e-6);  // speed_bin_1
    out[59] = speed_bins[2] / std::max(mean_speed, 1e-6);  // speed_bin_2
    out[60] = speed_bins[3] / std::max(mean_speed, 1e-6);  // speed_bin_3
    out[61] = lateral_drift;                      // prefix_shape_recent_lateral_drift
}

/**
 * Compute normalized summary vector (ready for TCN input).
 *
 * @param prefix     [n, 2] float32 raw dx/dy counts
 * @param n          number of prefix ticks
 * @param target_x   target x relative to B
 * @param target_y   target y relative to B
 * @param radius     target radius
 * @param progress   center-referenced progress
 * @param cfg        planner config with normalizers and feature reorder
 * @param out        [summary_dim] output float32 normalized vector
 */
inline void normalized_summary(
    const float* prefix, int n,
    double target_x, double target_y, double radius, double progress,
    const PlannerConfig& cfg,
    float* out)
{
    // Compute raw features
    double raw[62];
    raw_summary62(prefix, n, target_x, target_y, radius, progress, raw);

    // Reorder and normalize according to checkpoint feature names
    // The checkpoint stores features in a specific order that may differ
    // from the canonical order. We need a reorder map.
    // For now, assume canonical order (reorder[i] = i).
    // If the checkpoint uses a different order, load the reorder map from config.
    for (int j = 0; j < cfg.summary_dim; ++j) {
        float value = (float)raw[j];  // canonical order = checkpoint order
        if (!std::isfinite(value)) value = 0;
        float std_val = cfg.summary_std[j];
        if (std_val < 1e-6f) std_val = 1.0f;
        float normalized = (value - cfg.summary_mean[j]) / std_val;
        out[j] = std::isfinite(normalized) ? normalized : 0.0f;
    }
}

}  // namespace abc
