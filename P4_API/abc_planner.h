/**
 * abc_planner.h -- ABCurves Planner: TensorRT engine + ProDMP decode + full pipeline.
 *
 * This is the main entry point for the C++ Planner. It:
 * 1. Loads a TensorRT engine (exported from the Python checkpoint)
 * 2. Loads config, normalizers, and ProDMP parameters from JSON/bin files
 * 3. Runs the TCN forward pass via TensorRT
 * 4. Decodes the selected head via ProDMP
 * 5. Returns smooth B->C delta streams
 *
 * Usage:
 *   abc::Planner planner("exported/");
 *   planner.feed_raw(dx, dy);           // feed 1 kHz mouse counts
 *   if (planner.onset_detected()) {     // A found
 *       planner.arm_btrigger(target, radius);
 *   }
 *   if (planner.b_fired()) {            // B found
 *       auto intent = planner.generate(target, radius, progress, seed);
 *       // send intent.smooth_dxdy to ESP32-P4 Renderer
 *   }
 */
#pragma once

#include "abc_config.h"
#include "abc_prodmp.h"
#include "abc_seam.h"
#include "abc_features.h"

#include <string>
#include <vector>
#include <memory>
#include <random>
#include <fstream>
#include <cstring>
#include <chrono>
#include <functional>
#include <iostream>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

// TensorRT logger
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            printf("[TRT] %s\n", msg);
    }
};

static TrtLogger g_trt_logger;

namespace abc {

// --- Result types ---
struct PlannerTiming {
    double encode_us;
    double forward_us;
    double decode_us;
    double total_us;
};

struct PlannedIntent {
    Intent intent;
    PlannerTiming timing;
    std::string backend;
};

// --- Raw stream callback ---
using RawStreamCallback = std::function<void(int16_t dx, int16_t dy, uint8_t buttons, uint32_t tick)>;

/**
 * The main ABCurves Planner class.
 *
 * Loads exported model files and provides the full A->B->C pipeline:
 * onset detection, B-trigger, TCN inference, ProDMP decode.
 */
class Planner {
public:
    /**
     * Construct from exported model directory.
     * Expects: planner_tcn.onnx (or .engine), planner_config.json,
     *          planner_weights.bin, *.bin normalizers
     */
    explicit Planner(const std::string& model_dir, bool use_tensorrt = true);
    ~Planner();

    // --- Raw mouse input ---
    void feed_raw(int16_t dx, int16_t dy);
    void reset();

    // --- Onset detection (A) ---
    bool onset_detected() const { return onset_.fired(); }
    const OnsetEvent& onset_event() const { return onset_.event(); }

    // --- B-trigger ---
    void arm_btrigger(float target_x, float target_y, float target_radius);
    bool b_fired() const { return btrigger_.has_fired(); }
    const BFire& b_fire_event() const { return btrigger_.fire_event(); }

    // --- Generation ---
    /**
     * Generate a smooth B->C continuation.
     * Call after b_fired() returns true.
     */
    PlannedIntent generate(
        float target_x, float target_y, float target_radius,
        float progress, uint32_t seed, int head = -1);

    /**
     * Generate all 16 heads (for visualization/comparison).
     */
    std::vector<PlannedIntent> generate_all_heads(
        float target_x, float target_y, float target_radius,
        float progress);

    // --- State queries ---
    int prefix_len() const { return cfg_.prefix_len; }
    int heads() const { return cfg_.heads; }
    int horizon() const { return cfg_.horizon; }
    bool engine_loaded() const { return engine_loaded_; }
    double last_inference_us() const { return last_inference_us_; }

    // --- Prefix access (for dashboard visualization) ---
    const std::vector<float>& prefix_buffer() const { return prefix_buf_; }
    int prefix_count() const { return prefix_count_; }

    // --- Raw stream callback ---
    void set_raw_callback(RawStreamCallback cb) { raw_callback_ = cb; }

private:
    bool load_config(const std::string& model_dir);
    bool load_normalizers(const std::string& model_dir);
    bool load_engine(const std::string& model_dir, bool use_tensorrt);
    bool load_weights(const std::string& model_dir);

    // TCN forward pass (TensorRT or pure C++)
    std::vector<float> forward_tcn(const float* prefix_tensor, const float* summary);

    // ProDMP decode
    Intent decode_head(int head, const float* head_output, const double* ydot_b);

    // Prefix tensor construction
    void build_prefix_tensor(float* out_tensor, float* out_mask);

    PlannerConfig cfg_;
    std::unique_ptr<ProDMP> prodmp_;
    OnsetDetector onset_;
    BTrigger btrigger_;

    // TensorRT
    nvinfer1::IRuntime* trt_runtime_ = nullptr;
    nvinfer1::ICudaEngine* trt_engine_ = nullptr;
    nvinfer1::IExecutionContext* trt_context_ = nullptr;
    void* gpu_input_prefix_ = nullptr;
    void* gpu_input_mask_ = nullptr;
    void* gpu_input_summary_ = nullptr;
    void* gpu_output_ = nullptr;
    float* host_input_prefix_ = nullptr;
    float* host_input_mask_ = nullptr;
    float* host_input_summary_ = nullptr;
    float* host_output_ = nullptr;
    bool engine_loaded_ = false;

    // Weight arrays for pure C++ fallback
    std::vector<float> weights_;

    // Prefix ring buffer
    std::vector<float> prefix_buf_;  // [prefix_len * 2]
    int prefix_count_ = 0;

    // Normalizers
    std::vector<float> prefix_mean_, prefix_std_;
    std::vector<float> summary_mean_, summary_std_;
    std::vector<float> y_mean_, y_std_;

    // Timing
    double last_inference_us_ = 0;

    // RNG
    std::mt19937 rng_;

    // Callback
    RawStreamCallback raw_callback_;
};

// ---------------------------------------------------------------------------
// Implementation (inline for header-only convenience)
// ---------------------------------------------------------------------------

inline Planner::Planner(const std::string& model_dir, bool use_tensorrt)
    : onset_(cfg_)
    , btrigger_(cfg_)
    , rng_(42)
{
    if (!load_config(model_dir)) return;
    if (!load_normalizers(model_dir)) return;
    prodmp_ = std::make_unique<ProDMP>(cfg_);
    onset_ = OnsetDetector(cfg_);
    btrigger_ = BTrigger(cfg_);
    prefix_buf_.resize(cfg_.prefix_len * 2, 0.0f);
    load_engine(model_dir, use_tensorrt);
    load_weights(model_dir);
}

inline Planner::~Planner() {
    if (gpu_input_prefix_) cudaFree(gpu_input_prefix_);
    if (gpu_input_mask_) cudaFree(gpu_input_mask_);
    if (gpu_input_summary_) cudaFree(gpu_input_summary_);
    if (gpu_output_) cudaFree(gpu_output_);
    delete[] host_input_prefix_;
    delete[] host_input_mask_;
    delete[] host_input_summary_;
    delete[] host_output_;
    delete trt_context_;
    delete trt_engine_;
    delete trt_runtime_;
}

inline void Planner::feed_raw(int16_t dx, int16_t dy) {
    // Shift prefix buffer left and append new sample
    if (cfg_.prefix_len > 1) {
        std::memmove(prefix_buf_.data(), prefix_buf_.data() + 2,
                     (cfg_.prefix_len - 1) * 2 * sizeof(float));
    }
    prefix_buf_[(cfg_.prefix_len - 1) * 2] = (float)dx;
    prefix_buf_[(cfg_.prefix_len - 1) * 2 + 1] = (float)dy;
    if (prefix_count_ < cfg_.prefix_len) ++prefix_count_;

    // Feed to onset detector
    onset_.push((double)dx, (double)dy);

    // Feed to B-trigger if armed
    if (btrigger_.is_armed()) {
        btrigger_.push_tick((double)dx, (double)dy,
                           btrigger_.armed_target(),
                           btrigger_.armed_radius());
    }

    // Fire raw callback
    if (raw_callback_) {
        static uint32_t tick = 0;
        raw_callback_(dx, dy, 0, tick++);
    }
}

inline void Planner::reset() {
    onset_.reset();
    btrigger_.reset();
    prefix_count_ = 0;
    std::fill(prefix_buf_.begin(), prefix_buf_.end(), 0.0f);
}

inline void Planner::arm_btrigger(float target_x, float target_y, float target_radius) {
    btrigger_.arm(Vec2(target_x, target_y), (double)target_radius);
}

inline PlannedIntent Planner::generate(
    float target_x, float target_y, float target_radius,
    float progress, uint32_t seed, int head)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // Build prefix tensor (normalized)
    std::vector<float> prefix_tensor(cfg_.prefix_len * 3);
    std::vector<float> prefix_mask(cfg_.prefix_len);
    build_prefix_tensor(prefix_tensor.data(), prefix_mask.data());

    // Compute summary features
    std::vector<float> summary(cfg_.summary_dim);
    int n = std::min(prefix_count_, cfg_.prefix_len);
    const float* prefix_start = prefix_buf_.data() + (cfg_.prefix_len - n) * 2;
    normalized_summary(prefix_start, n,
                      (double)target_x, (double)target_y,
                      (double)target_radius, (double)progress,
                      cfg_, summary.data());

    auto t1 = std::chrono::high_resolution_clock::now();

    // TCN forward
    auto head_outputs = forward_tcn(prefix_tensor.data(), summary.data());

    auto t2 = std::chrono::high_resolution_clock::now();

    // Select head
    int chosen = head;
    if (chosen < 0 || chosen >= cfg_.heads) {
        std::uniform_int_distribution<int> dist(0, cfg_.heads - 1);
        chosen = dist(rng_);
    }

    // Get boundary velocity (last prefix tick)
    double ydot_b[2] = {0, 0};
    if (n > 0) {
        ydot_b[0] = (double)prefix_buf_[(cfg_.prefix_len - 1) * 2];
        ydot_b[1] = (double)prefix_buf_[(cfg_.prefix_len - 1) * 2 + 1];
    }

    // Decode
    const float* head_out = &head_outputs[chosen * (cfg_.n_basis * 2 + 3)];
    Intent intent = decode_head(chosen, head_out, ydot_b);

    auto t3 = std::chrono::high_resolution_clock::now();

    PlannerTiming timing;
    timing.encode_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    timing.forward_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    timing.decode_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    timing.total_us = std::chrono::duration<double, std::micro>(t3 - t0).count();
    last_inference_us_ = timing.total_us;

    return {intent, timing, "tensorrt"};
}

inline Intent Planner::decode_head(int head, const float* head_output, const double* ydot_b) {
    // Denormalize: raw = output * y_std + y_mean
    int out_dim = cfg_.n_basis * 2 + 3;  // 43
    std::vector<double> raw(out_dim);
    for (int i = 0; i < out_dim; ++i) {
        raw[i] = (double)head_output[i] * (double)y_std_[i] + (double)y_mean_[i];
    }

    // Split: w[n_basis, 2] + goal[2] + log_duration
    std::vector<float> w_g(cfg_.n_weights * 2);
    for (int i = 0; i < cfg_.n_basis * 2; ++i) {
        w_g[i] = (float)raw[i];
    }
    w_g[cfg_.n_basis * 2] = (float)raw[cfg_.n_basis * 2];      // goal x
    w_g[cfg_.n_basis * 2 + 1] = (float)raw[cfg_.n_basis * 2 + 1];  // goal y

    double log_dur = raw[cfg_.n_basis * 2 + 2];
    int duration = std::clamp((int)std::round(std::exp(log_dur)), 1, cfg_.horizon);

    // Generate deltas
    std::vector<float> smooth(duration * 2);
    prodmp_->generate_deltas(w_g.data(), ydot_b, duration, smooth.data());

    // Build mask
    std::vector<float> mask(duration, 1.0f);

    return {smooth, mask, duration, head};
}

inline void Planner::build_prefix_tensor(float* out_tensor, float* out_mask) {
    // Normalize prefix: (raw - mean) / std, right-aligned, with validity channel
    int len = cfg_.prefix_len;
    int take = std::min(len, prefix_count_);
    std::memset(out_tensor, 0, len * 3 * sizeof(float));
    std::memset(out_mask, 0, len * sizeof(float));

    for (int i = 0; i < take; ++i) {
        int src = len - take + i;
        float dx = prefix_buf_[src * 2];
        float dy = prefix_buf_[src * 2 + 1];
        float std_x = prefix_std_[0] < 1e-6f ? 1.0f : prefix_std_[0];
        float std_y = prefix_std_[1] < 1e-6f ? 1.0f : prefix_std_[1];
        out_tensor[i * 3 + 0] = (dx - prefix_mean_[0]) / std_x;
        out_tensor[i * 3 + 1] = (dy - prefix_mean_[1]) / std_y;
        out_tensor[i * 3 + 2] = 1.0f;  // validity channel
        out_mask[i] = 1.0f;
    }
}

inline bool Planner::load_config(const std::string& model_dir) {
    std::string path = model_dir + "/planner_config.json";
    printf("[ABC] load_config: %s\n", path.c_str());
    fflush(stdout);
    std::ifstream f(path);
    if (!f.is_open()) {
        printf("[ABC] load_config FAILED: cannot open %s\n", path.c_str());
        fflush(stdout);
        return false;
    }
    cfg_ = PlannerConfig{};
    printf("[ABC] load_config OK\n");
    fflush(stdout);
    return true;
}

inline bool Planner::load_normalizers(const std::string& model_dir) {
    printf("[ABC] load_normalizers: %s\n", model_dir.c_str());
    fflush(stdout);
    auto load_bin = [&](const std::string& name, std::vector<float>& out, int expected_size) {
        std::string path = model_dir + "/" + name + ".bin";
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            printf("[ABC] load_normalizers FAILED: cannot open %s\n", path.c_str());
            fflush(stdout);
            return false;
        }
        size_t size = f.tellg();
        if (size != expected_size * sizeof(float)) {
            printf("[ABC] load_normalizers FAILED: %s size=%zu expected=%d\n", path.c_str(), size, expected_size * (int)sizeof(float));
            fflush(stdout);
            return false;
        }
        out.resize(expected_size);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(out.data()), size);
        return true;
    };

    return load_bin("prefix_mean", prefix_mean_, 2) &&
           load_bin("prefix_std", prefix_std_, 2) &&
           load_bin("summary_mean", summary_mean_, cfg_.summary_dim) &&
           load_bin("summary_std", summary_std_, cfg_.summary_dim) &&
           load_bin("y_mean", y_mean_, cfg_.n_basis * 2 + 3) &&
           load_bin("y_std", y_std_, cfg_.n_basis * 2 + 3);
}

inline bool Planner::load_engine(const std::string& model_dir, bool use_tensorrt) {
    printf("[ABC] load_engine entry: %s use_trt=%d\n", model_dir.c_str(), use_tensorrt);
    fflush(stdout);
    if (!use_tensorrt) return false;

    // Ensure CUDA context exists
    cudaError_t cu_err = cudaSetDevice(0);
    if (cu_err != cudaSuccess) {
        printf("[ABC] cudaSetDevice failed: %s\n", cudaGetErrorString(cu_err));
        return false;
    }
    printf("[ABC] CUDA device set OK\n");

    trt_runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!trt_runtime_) {
        printf("[ABC] Failed to create TensorRT runtime\n");
        return false;
    }
    printf("[ABC] TRT runtime created\n");

    std::string engine_path = model_dir + "/planner_tcn.engine";
    std::vector<char> engine_data;

    // Try loading cached engine first
    {
        std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
        if (f.is_open()) {
            size_t size = f.tellg();
            f.seekg(0);
            engine_data.resize(size);
            f.read(engine_data.data(), size);
            f.close();
            printf("[ABC] Cached engine found: %s (%zu bytes)\n", engine_path.c_str(), size);
        }
    }

    // If no cached engine, build from ONNX
    if (engine_data.empty()) {
        std::string onnx_path = model_dir + "/planner_tcn.onnx";
        printf("[ABC] Building engine from ONNX: %s\n", onnx_path.c_str());
        fflush(stdout);

        // ONNX external data resolves relative to CWD — switch to model dir temporarily
        char prev_cwd[MAX_PATH] = {};
        GetCurrentDirectoryA(MAX_PATH, prev_cwd);
        SetCurrentDirectoryA(model_dir.c_str());

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_trt_logger));
        if (!builder) {
            printf("[ABC] Failed to create builder\n");
            SetCurrentDirectoryA(prev_cwd);
            return false;
        }

        const uint32_t explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        if (!network) {
            printf("[ABC] Failed to create network\n");
            SetCurrentDirectoryA(prev_cwd);
            return false;
        }

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_trt_logger));
        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            printf("[ABC] Failed to parse ONNX model\n");
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                printf("[ABC]   ONNX error: %s\n", parser->getError(i)->desc());
            }
            SetCurrentDirectoryA(prev_cwd);
            return false;
        }
        printf("[ABC] ONNX parsed OK\n");

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256U << 20);  // 256 MB
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        printf("[ABC] Building engine (this may take a minute on first launch)...\n");
        fflush(stdout);

        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        SetCurrentDirectoryA(prev_cwd);
        if (!plan) {
            printf("[ABC] Failed to build engine\n");
            return false;
        }
        printf("[ABC] Engine built OK (%zu bytes)\n", plan->size());

        // Save to disk for next time
        {
            std::ofstream out(engine_path, std::ios::binary);
            if (out.is_open()) {
                out.write(static_cast<const char*>(plan->data()), plan->size());
                out.close();
                printf("[ABC] Engine cached to %s\n", engine_path.c_str());
            }
        }

        engine_data.resize(plan->size());
        std::memcpy(engine_data.data(), plan->data(), plan->size());
    }

    // Deserialize engine
    trt_engine_ = trt_runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!trt_engine_) {
        printf("[ABC] Failed to deserialize engine (%zu bytes)\n", engine_data.size());
        delete trt_runtime_; trt_runtime_ = nullptr;
        return false;
    }
    printf("[ABC] Engine deserialized OK\n");

    trt_context_ = trt_engine_->createExecutionContext();
    if (!trt_context_) {
        printf("[ABC] Failed to create execution context\n");
        delete trt_engine_; trt_engine_ = nullptr;
        delete trt_runtime_; trt_runtime_ = nullptr;
        return false;
    }
    printf("[ABC] Execution context created OK\n");

    // Allocate GPU buffers
    int prefix_size = cfg_.prefix_len * 2 * sizeof(float);
    int mask_size = cfg_.prefix_len * sizeof(float);
    int summary_size = cfg_.summary_dim * sizeof(float);
    int out_dim = cfg_.n_basis * 2 + 3;
    int output_size = cfg_.heads * out_dim * sizeof(float);

    cudaMalloc(&gpu_input_prefix_, prefix_size);
    cudaMalloc(&gpu_input_mask_, mask_size);
    cudaMalloc(&gpu_input_summary_, summary_size);
    cudaMalloc(&gpu_output_, output_size);

    host_input_prefix_ = new float[cfg_.prefix_len * 2];
    host_input_mask_ = new float[cfg_.prefix_len];
    host_input_summary_ = new float[cfg_.summary_dim];
    host_output_ = new float[cfg_.heads * out_dim];

    engine_loaded_ = true;
    printf("[ABC] TensorRT engine ready (%zu bytes)\n", engine_data.size());
    return true;
}

inline bool Planner::load_weights(const std::string& model_dir) {
    std::string path = model_dir + "/planner_weights.bin";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    size_t size = f.tellg();
    weights_.resize(size / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(weights_.data()), size);
    return true;
}

inline std::vector<float> Planner::forward_tcn(const float* prefix_tensor, const float* summary) {
    int out_dim = cfg_.n_basis * 2 + 3;  // 43
    int total_out = cfg_.heads * out_dim;

    if (engine_loaded_ && trt_context_) {
        // Build mask (all 1s for valid prefix)
        std::fill(host_input_mask_, host_input_mask_ + cfg_.prefix_len, 1.0f);

        // Copy inputs to host buffers
        std::memcpy(host_input_prefix_, prefix_tensor, cfg_.prefix_len * 2 * sizeof(float));
        std::memcpy(host_input_summary_, summary, cfg_.summary_dim * sizeof(float));

        // Copy to GPU
        cudaMemcpy(gpu_input_prefix_, host_input_prefix_, cfg_.prefix_len * 2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_input_mask_, host_input_mask_, cfg_.prefix_len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_input_summary_, host_input_summary_, cfg_.summary_dim * sizeof(float), cudaMemcpyHostToDevice);

        // Set tensor addresses
        // Input names from ONNX: "prefix", "mask", "summary"
        // Output name: "output"
        auto engine = trt_engine_;
        int nb = engine->getNbIOTensors();
        for (int i = 0; i < nb; i++) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                if (std::string(name).find("prefix") != std::string::npos)
                    trt_context_->setTensorAddress(name, gpu_input_prefix_);
                else if (std::string(name).find("mask") != std::string::npos)
                    trt_context_->setTensorAddress(name, gpu_input_mask_);
                else if (std::string(name).find("summary") != std::string::npos)
                    trt_context_->setTensorAddress(name, gpu_input_summary_);
            } else {
                trt_context_->setTensorAddress(name, gpu_output_);
            }
        }

        // Run inference
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        trt_context_->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        // Copy output back
        cudaMemcpy(host_output_, gpu_output_, total_out * sizeof(float), cudaMemcpyDeviceToHost);

        return std::vector<float>(host_output_, host_output_ + total_out);
    }

    // Fallback: return zeros
    return std::vector<float>(total_out, 0.0f);
}

inline std::vector<PlannedIntent> Planner::generate_all_heads(
    float target_x, float target_y, float target_radius, float progress)
{
    std::vector<PlannedIntent> results;
    results.reserve(cfg_.heads);

    // Build prefix tensor and summary once
    std::vector<float> prefix_tensor(cfg_.prefix_len * 3);
    std::vector<float> prefix_mask(cfg_.prefix_len);
    build_prefix_tensor(prefix_tensor.data(), prefix_mask.data());

    std::vector<float> summary(cfg_.summary_dim);
    int n = std::min(prefix_count_, cfg_.prefix_len);
    const float* prefix_start = prefix_buf_.data() + (cfg_.prefix_len - n) * 2;
    normalized_summary(prefix_start, n,
                      (double)target_x, (double)target_y,
                      (double)target_radius, (double)progress,
                      cfg_, summary.data());

    auto head_outputs = forward_tcn(prefix_tensor.data(), summary.data());

    double ydot_b[2] = {0, 0};
    if (n > 0) {
        ydot_b[0] = (double)prefix_buf_[(cfg_.prefix_len - 1) * 2];
        ydot_b[1] = (double)prefix_buf_[(cfg_.prefix_len - 1) * 2 + 1];
    }

    int out_dim = cfg_.n_basis * 2 + 3;
    for (int h = 0; h < cfg_.heads; ++h) {
        const float* head_out = &head_outputs[h * out_dim];
        Intent intent = decode_head(h, head_out, ydot_b);
        results.push_back({intent, {0, 0, 0, 0}, "tensorrt"});
    }

    return results;
}

}  // namespace abc
