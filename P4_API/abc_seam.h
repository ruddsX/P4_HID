/**
 * abc_seam.h -- Causal onset detection (A) and B-trigger state machines.
 *
 * Port of abcurves/seam.py. These operate on closed 1 ms count bins and
 * use the exact contract embedded in the release Planner.
 */
#pragma once

#include "abc_config.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace abc {

// ---------------------------------------------------------------------------
// OnsetDetector -- find point A (movement start)
// ---------------------------------------------------------------------------
class OnsetDetector {
public:
    explicit OnsetDetector(const PlannerConfig& cfg)
        : cfg_(cfg)
    {
        reset();
    }

    void reset() {
        baseline_speeds_.clear();
        pending_.clear();
        threshold_ = -1.0;
        median_ = 0;
        mad_ = 0;
        run_ = 0;
        tick_count_ = 0;
        done_ = false;
        event_ = {};
    }

    /**
     * Feed one 1 ms (dx, dy) report. Returns true once when onset is detected.
     * After firing, returns false on subsequent calls.
     */
    bool push(double dx, double dy, const Vec2* target_rel = nullptr) {
        if (done_) return false;

        double speed = std::sqrt(dx * dx + dy * dy);
        bool aligned = true;

        if (target_rel) {
            double target_norm = target_rel->norm();
            double denom = speed * target_norm + 1e-6;
            double cos_angle = (dx * target_rel->x + dy * target_rel->y) / denom;
            aligned = (cos_angle >= cfg_.onset_alignment_min);
        }

        int tick = tick_count_++;
        bool preferred = false;

        if (threshold_ < 0) {
            // Still collecting baseline
            baseline_speeds_.push_back(speed);
            pending_.push_back({tick, speed, aligned});

            if ((int)baseline_speeds_.size() < cfg_.onset_noise_window_ms) {
                return false;
            }

            // Compute threshold from baseline
            compute_threshold();

            // Scan pending ticks
            for (auto& p : pending_) {
                if (scan(p.tick, p.speed, p.aligned)) {
                    return true;
                }
            }
            pending_.clear();
            return false;
        }

        return scan(tick, speed, aligned);
    }

    bool fired() const { return done_; }
    const OnsetEvent& event() const { return event_; }
    double threshold() const { return threshold_; }
    double median() const { return median_; }
    double mad() const { return mad_; }

private:
    struct PendingTick {
        int tick;
        double speed;
        bool aligned;
    };

    void compute_threshold() {
        std::vector<double> sorted = baseline_speeds_;
        std::sort(sorted.begin(), sorted.end());
        int n = (int)sorted.size();

        // Median
        median_ = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];

        // MAD (median absolute deviation)
        std::vector<double> abs_dev(n);
        for (int i = 0; i < n; ++i) {
            abs_dev[i] = std::abs(sorted[i] - median_);
        }
        std::sort(abs_dev.begin(), abs_dev.end());
        mad_ = (n % 2 == 0) ? (abs_dev[n / 2 - 1] + abs_dev[n / 2]) / 2.0 : abs_dev[n / 2];

        // Threshold
        if (median_ > cfg_.onset_speed_floor) {
            threshold_ = cfg_.onset_speed_floor;
        } else {
            threshold_ = std::max(
                cfg_.onset_speed_floor,
                median_ + cfg_.onset_threshold_mad_mult * mad_
            );
        }
    }

    bool scan(int tick, double speed, bool aligned) {
        bool preferred = (speed > threshold_) && aligned;
        run_ = preferred ? run_ + 1 : 0;

        if (run_ < cfg_.onset_consecutive_ticks) return false;

        done_ = true;
        int onset_start = tick - cfg_.onset_consecutive_ticks + 1;
        event_ = {
            std::max(0, onset_start - cfg_.onset_backtrack_ms),
            threshold_,
            median_,
            mad_
        };
        return true;
    }

    PlannerConfig cfg_;
    std::vector<double> baseline_speeds_;
    std::vector<PendingTick> pending_;
    double threshold_;
    double median_;
    double mad_;
    int run_;
    int tick_count_;
    bool done_;
    OnsetEvent event_;
};

// ---------------------------------------------------------------------------
// BTrigger -- find point B (handoff to Planner)
// ---------------------------------------------------------------------------
class BTrigger {
public:
    explicit BTrigger(const PlannerConfig& cfg)
        : cfg_(cfg)
    {
        reset();
    }

    void reset() {
        armed_ = false;
        movement_ = {};
        target_at_A_ = {};
        radius_at_A_ = 0;
        t_ms_ = 0;
        max_progress_center_ = 0;
        fired_ = false;
        fire_ = {};
    }

    void arm(const Vec2& target_rel, double target_radius) {
        armed_ = true;
        movement_ = {};
        target_at_A_ = target_rel;
        radius_at_A_ = target_radius;
        t_ms_ = 0;
        max_progress_center_ = 0;
        fired_ = false;
    }

    void disarm() { armed_ = false; }
    bool is_armed() const { return armed_ && !fired_; }
    bool has_fired() const { return fired_; }
    const BFire& fire_event() const { return fire_; }
    const Vec2& armed_target() const { return target_at_A_; }
    double armed_radius() const { return radius_at_A_; }

    double progress(const Vec2& target, double radius, bool edge) const {
        double distance = target.norm();
        double denom = edge ? distance - std::max(radius, 0.0) : distance;
        denom = std::max(denom, 1e-9);
        Vec2 unit = target.norm() > 1e-9 ? Vec2(target.x / target.norm(), target.y / target.norm()) : Vec2(0, 0);
        return movement_.dot(unit) / denom;
    }

    /**
     * Feed one 1 ms (dx, dy) report with current target info.
     * Returns true when B fires.
     */
    bool push_tick(double dx, double dy, const Vec2& target_rel_now, double target_radius_now) {
        if (!armed_ || fired_) return false;

        movement_.x += dx;
        movement_.y += dy;
        t_ms_++;

        double prog_center = progress(target_rel_now, target_radius_now, false);
        double prog_edge = progress(target_rel_now, target_radius_now, true);

        // Max progress tracking (regression detection)
        if (prog_center > max_progress_center_) {
            max_progress_center_ = prog_center;
        } else if (cfg_.b_progress_regression_th > 0 &&
                   max_progress_center_ - prog_center > cfg_.b_progress_regression_th) {
            disarm();
            return false;
        }

        // Time limit
        if (t_ms_ > cfg_.b_max_ab_ms) {
            disarm();
            return false;
        }

        // Progress threshold
        if (prog_edge < cfg_.b_threshold) return false;

        // Center progress limit
        if (prog_center > cfg_.b_max_center_progress) {
            disarm();
            return false;
        }

        // Remaining distance check
        double remaining = target_rel_now.norm();
        if (remaining < cfg_.b_min_remaining_counts) {
            disarm();
            return false;
        }

        // Must be outside target
        if (remaining <= target_radius_now) {
            disarm();
            return false;
        }

        // Fire!
        fired_ = true;
        armed_ = false;
        fire_ = {
            t_ms_,
            prog_edge,
            prog_center,
            movement_,
            target_rel_now,
            target_radius_now,
            remaining
        };
        return true;
    }

private:
    PlannerConfig cfg_;
    bool armed_;
    bool fired_;
    Vec2 movement_;
    Vec2 target_at_A_;
    double radius_at_A_;
    int t_ms_;
    double max_progress_center_;
    BFire fire_;
};

}  // namespace abc
