/**
 * abc_prodmp.h -- ProDMP basis precomputation and trajectory decode.
 *
 * Port of abcurves/prodmp.py. The decode path converts a weight vector
 * (20 forcing + 1 goal, per axis) plus boundary velocity into a smooth
 * delta stream of any duration up to HORIZON.
 */
#pragma once

#include "abc_config.h"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

namespace abc {

class ProDMP {
public:
    ProDMP(const PlannerConfig& cfg)
        : cfg_(cfg)
        , n_basis_(cfg.n_basis)
        , alpha_(cfg.alpha)
        , alpha_phase_(cfg.alpha_phase)
        , half_alpha_(alpha_ / 2.0)
    {
        precompute_basis();
    }

    /**
     * Generate B-relative delta stream from ProDMP weights.
     *
     * @param w_g  [n_basis+1, 2] forcing weights + goal, row-major
     * @param ydot_b  boundary velocity [2]
     * @param duration  number of ms (1..HORIZON)
     * @param out_deltas  output [duration, 2] float32
     */
    void generate_deltas(const float* w_g, const double* ydot_b,
                         int duration, float* out_deltas) const
    {
        const int d = std::clamp(duration, 1, (int)HORIZON);
        const int nw = n_weights();

        // Interpolate basis at s = t/duration for t = 1..d
        // phi[t, j] for j = 0..n_basis (forcing) and j = n_basis (goal)
        std::vector<double> phi(d * nw);
        std::vector<double> dphi(d * nw);

        for (int t = 1; t <= d; ++t) {
            double s = (double)t / (double)d;
            basis_at(s, &phi[(t - 1) * nw], &dphi[(t - 1) * nw]);
        }

        // Boundary response: xi1 * y_b + xi2 * ydot_b
        // With t_b = 0: xi1 = 1, xi2 = t (for critically damped at t_b=0)
        // Actually from the Python: at t_b=0, the boundary correction is
        // just the free response of the DMP from (y_b=0, ydot_b).
        // y(t) = xi1(t)*y_b + xi2(t)*ydot_b where y_b=0 at B-relative coords
        // So boundary = xi2(t) * ydot_b
        // xi2(t) = (y1_b * y2(t) - y2_b * y1(t)) / det
        // At t_b=0: y1_b=1, y2_b=0, yd1_b=-a, yd2_b=1, det=1
        // So xi2(t) = y2(t) = t * exp(-a*t) (but in tau-normalized form)

        // For the canonical case (t_b=0, tau=d), the position is:
        // pos(t) = xi1(t)*0 + xi2(t)*ydot_b + H(t) . w_g
        // where H(t) = phi(t) (the normalized basis)
        // and xi2(t) = t * exp(-half_alpha * t/d) * d  (scaled by tau)

        // Actually, let's follow the Python more carefully.
        // generate() calls evaluate() which calls _components(t, tau, t_b)
        // _components returns (xi1, xi2, H) where H is the boundary-corrected basis
        // At t_b=0: H = phi (the raw basis), and the position is:
        //   xi1*y_b + xi2*ydot_b + H @ w_g
        // With y_b=0 (B-relative): position = xi2*ydot_b + H @ w_g

        // xi2 at t_b=0, tau=d:
        // xi2(t) = y2(t) = t * exp(-a*t) where a = alpha/(2*tau) = alpha/(2*d)
        // But we need it in the same units as ydot_b (counts/ms)
        // The Python uses: ydot_b is the last prefix velocity (counts/ms)
        // and tau = duration, so xi2 has units of ms

        const double a = alpha_ / (2.0 * (double)d);

        // Compute positions
        std::vector<double> pos_x(d), pos_y(d);
        for (int t = 1; t <= d; ++t) {
            double exp_at = std::exp(-a * (double)t);
            double xi2 = (double)t * exp_at;

            // H @ w_g for x and y
            double hx = 0, hy = 0;
            const double* phi_t = &phi[(t - 1) * nw];
            for (int j = 0; j < nw; ++j) {
                hx += phi_t[j] * (double)w_g[j * 2 + 0];
                hy += phi_t[j] * (double)w_g[j * 2 + 1];
            }

            pos_x[t - 1] = xi2 * ydot_b[0] + hx;
            pos_y[t - 1] = xi2 * ydot_b[1] + hy;
        }

        // Deltas = diff(positions), with prepend = y_b = 0
        double prev_x = 0, prev_y = 0;
        for (int t = 0; t < d; ++t) {
            out_deltas[t * 2 + 0] = (float)(pos_x[t] - prev_x);
            out_deltas[t * 2 + 1] = (float)(pos_y[t] - prev_y);
            prev_x = pos_x[t];
            prev_y = pos_y[t];
        }
    }

    int n_weights() const { return n_basis_ + 1; }

private:
    void precompute_basis() {
        // Precompute the normalized basis on a grid s = 0..1
        const int G = BASIS_GRID_POINTS;
        s_grid_.resize(G + 1);
        phi_grid_.resize((G + 1) * n_weights());
        dphi_grid_.resize((G + 1) * n_weights());

        // RBF centers in phase space
        std::vector<double> centers(n_basis_);
        for (int i = 0; i < n_basis_; ++i) {
            centers[i] = std::exp(-alpha_phase_ * (double)i / (double)(n_basis_ - 1));
        }

        // RBF widths (inverse squared spacing)
        std::vector<double> widths(n_basis_);
        for (int i = 0; i < n_basis_ - 1; ++i) {
            double spacing = centers[i] - centers[i + 1];
            widths[i] = 1.0 / std::max(spacing * spacing, 1e-9);
        }
        widths[n_basis_ - 1] = widths[n_basis_ - 2];

        // Evaluate basis on grid
        for (int g = 0; g <= G; ++g) {
            double s = (double)g / (double)G;
            s_grid_[g] = s;
            double phase = std::exp(-alpha_phase_ * s);

            // RBF values
            double rbf_sum = 0;
            std::vector<double> rbf(n_basis_);
            for (int i = 0; i < n_basis_; ++i) {
                double diff = phase - centers[i];
                rbf[i] = std::exp(-widths[i] * diff * diff);
                rbf_sum += rbf[i];
            }
            if (rbf_sum < 1e-9) rbf_sum = 1e-9;

            // Normalized RBF * phase = forcing basis
            std::vector<double> forcing(n_basis_);
            for (int i = 0; i < n_basis_; ++i) {
                forcing[i] = phase * rbf[i] / rbf_sum;
            }

            // Compute P1, P2 integrals (cumulative trapezoid)
            // We need these for the position basis phi_w
            // But for the grid evaluation, we compute them incrementally
            // For now, store the forcing values; we'll integrate below
            for (int i = 0; i < n_basis_; ++i) {
                phi_grid_[g * n_weights() + i] = forcing[i];  // temp: store forcing
            }
            phi_grid_[g * n_weights() + n_basis_] = 0;  // goal placeholder
        }

        // Now compute the actual position basis via cumulative integration
        // phi_w_i(s) = s*exp(-A*s)*P2_i(s) - exp(-A*s)*P1_i(s)
        // where P1_i(s) = integral_0^s s'*exp(A*s')*x(s')*phi_i(s') ds'
        //       P2_i(s) = integral_0^s exp(A*s')*x(s')*phi_i(s') ds'

        // We'll compute P1 and P2 using cumulative trapezoid on the grid
        std::vector<double> p1((G + 1) * n_basis_, 0.0);
        std::vector<double> p2((G + 1) * n_basis_, 0.0);

        for (int g = 1; g <= G; ++g) {
            double s = s_grid_[g];
            double s_prev = s_grid_[g - 1];
            double ds = s - s_prev;
            double ea = std::exp(half_alpha_ * s);
            double ea_prev = std::exp(half_alpha_ * s_prev);

            for (int i = 0; i < n_basis_; ++i) {
                double f = phi_grid_[g * n_weights() + i];  // forcing at s
                double f_prev = phi_grid_[(g - 1) * n_weights() + i];

                // integrand for P1: s * exp(A*s) * forcing
                double integ1 = s * ea * f;
                double integ1_prev = s_prev * ea_prev * f_prev;

                // integrand for P2: exp(A*s) * forcing
                double integ2 = ea * f;
                double integ2_prev = ea_prev * f_prev;

                p1[g * n_basis_ + i] = p1[(g - 1) * n_basis_ + i] + 0.5 * ds * (integ1 + integ1_prev);
                p2[g * n_basis_ + i] = p2[(g - 1) * n_basis_ + i] + 0.5 * ds * (integ2 + integ2_prev);
            }
        }

        // Now compute the actual position basis
        for (int g = 0; g <= G; ++g) {
            double s = s_grid_[g];
            double em = std::exp(-half_alpha_ * s);

            for (int i = 0; i < n_basis_; ++i) {
                double p1_val = p1[g * n_basis_ + i];
                double p2_val = p2[g * n_basis_ + i];
                phi_grid_[g * n_weights() + i] = s * em * p2_val - em * p1_val;
            }
            // Goal basis: 1 - exp(-A*s)*(1 + A*s)
            phi_grid_[g * n_weights() + n_basis_] = 1.0 - em * (1.0 + half_alpha_ * s);
        }

        // Velocity basis (for completeness, not used in generate_deltas)
        for (int g = 0; g <= G; ++g) {
            double s = s_grid_[g];
            double em = std::exp(-half_alpha_ * s);

            for (int i = 0; i < n_basis_; ++i) {
                double p1_val = p1[g * n_basis_ + i];
                double p2_val = p2[g * n_basis_ + i];
                dphi_grid_[g * n_weights() + i] = em * ((1.0 - half_alpha_ * s) * p2_val + half_alpha_ * p1_val);
            }
            dphi_grid_[g * n_weights() + n_basis_] = half_alpha_ * half_alpha_ * s * em;
        }
    }

    /**
     * Interpolate basis at normalized time s (0..1).
     */
    void basis_at(double s, double* phi_out, double* dphi_out) const {
        s = std::clamp(s, 0.0, 1.0);
        const int G = BASIS_GRID_POINTS;
        const int nw = n_weights();

        // Find grid interval
        double idx_f = s * (double)G;
        int idx = (int)idx_f;
        if (idx >= G) idx = G - 1;
        double t = idx_f - (double)idx;

        const double* phi_lo = &phi_grid_[idx * nw];
        const double* phi_hi = &phi_grid_[(idx + 1) * nw];
        const double* dphi_lo = &dphi_grid_[idx * nw];
        const double* dphi_hi = &dphi_grid_[(idx + 1) * nw];

        for (int j = 0; j < nw; ++j) {
            phi_out[j] = phi_lo[j] + t * (phi_hi[j] - phi_lo[j]);
            if (dphi_out) {
                dphi_out[j] = dphi_lo[j] + t * (dphi_hi[j] - dphi_lo[j]);
            }
        }
    }

    PlannerConfig cfg_;
    int n_basis_;
    double alpha_;
    double alpha_phase_;
    double half_alpha_;

    std::vector<double> s_grid_;       // [G+1]
    std::vector<double> phi_grid_;     // [G+1, n_weights] position basis
    std::vector<double> dphi_grid_;    // [G+1, n_weights] velocity basis
};

}  // namespace abc
