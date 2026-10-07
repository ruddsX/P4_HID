#include "pch.h"
#include "P4HID_API.h"
#include "abc_planner.h"
#include <cstring>
#include <cstdio>
#include <chrono>
#include <thread>
#include <atomic>

struct P4HID_Handle {
    SOCKET sock;
    struct sockaddr_in addr;
    /* AES-128-CBC */
    bool aes_enabled;
    uint8_t aes_key[P4HID_AES_KEY_SIZE];
    BCRYPT_ALG_HANDLE aes_alg;
    /* Button state tracking */
    uint8_t buttons;
    std::atomic<uint8_t> fw_real_buttons{0};
    std::atomic<uint8_t> fw_synth_buttons{0};
    std::atomic<bool> reader_running{false};
    std::thread reader_thread;
    SOCKET resp_sock;  // port+1 for button queries
};

static size_t aes_encrypt(P4HID_Handle* h, const uint8_t* in, size_t in_len, uint8_t* out, size_t out_max)
{
    size_t padded_len = ((in_len / 16) + 1) * 16;
    size_t needed = P4HID_AES_NONCE_SIZE + padded_len;
    if (out_max < needed) return 0;

    BCryptGenRandom(NULL, out, P4HID_AES_NONCE_SIZE, BCRYPT_USE_SYSTEM_PREFERRED_RNG);

    uint8_t iv[16];
    memset(iv, 0, sizeof(iv));
    memcpy(iv, out, P4HID_AES_NONCE_SIZE);

    BCRYPT_KEY_HANDLE key_handle = NULL;
    BCryptGenerateSymmetricKey(h->aes_alg, &key_handle, NULL, 0, h->aes_key, P4HID_AES_KEY_SIZE, 0);

    ULONG ct_len = 0;
    BCryptEncrypt(key_handle, (PUCHAR)in, (ULONG)in_len, NULL, iv, 16,
                  out + P4HID_AES_NONCE_SIZE, (ULONG)padded_len, &ct_len,
                  BCRYPT_BLOCK_PADDING);
    BCryptDestroyKey(key_handle);

    return P4HID_AES_NONCE_SIZE + ct_len;
}

static size_t aes_decrypt(P4HID_Handle* h, const uint8_t* in, size_t in_len, uint8_t* out, size_t out_max)
{
    if (in_len < P4HID_AES_NONCE_SIZE + 16) return 0;

    size_t ct_len = in_len - P4HID_AES_NONCE_SIZE;
    if (out_max < ct_len) return 0;

    uint8_t iv[16];
    memset(iv, 0, sizeof(iv));
    memcpy(iv, in, P4HID_AES_NONCE_SIZE);

    BCRYPT_KEY_HANDLE key_handle = NULL;
    BCryptGenerateSymmetricKey(h->aes_alg, &key_handle, NULL, 0, h->aes_key, P4HID_AES_KEY_SIZE, 0);

    ULONG dec_len = 0;
    NTSTATUS st = BCryptDecrypt(key_handle, (PUCHAR)(in + P4HID_AES_NONCE_SIZE), (ULONG)ct_len,
                                NULL, iv, 16, out, (ULONG)out_max, &dec_len,
                                BCRYPT_BLOCK_PADDING);
    BCryptDestroyKey(key_handle);

    if (!BCRYPT_SUCCESS(st)) return 0;
    return dec_len;
}

static bool write_packet(P4HID_Handle* h, const uint8_t* data, size_t len)
{
    uint8_t enc_buf[1024];
    const uint8_t* send_data;
    size_t send_len;

    if (h->aes_enabled) {
        send_len = aes_encrypt(h, data, len, enc_buf, sizeof(enc_buf));
        if (send_len == 0) return false;
        send_data = enc_buf;
    } else {
        send_data = data;
        send_len = len;
    }

    int sent = sendto(h->sock, (const char*)send_data, (int)send_len, 0,
                      (struct sockaddr*)&h->addr, sizeof(h->addr));
    return sent == (int)send_len;
}

static void reader_thread_fn(P4HID_Handle* h)
{
    uint8_t buf[256];
    uint8_t pkt[16];
    uint8_t pkt_len = 0;
    bool synced = false;
    auto last_query = std::chrono::steady_clock::now();

    while (h->reader_running.load()) {
        auto now = std::chrono::steady_clock::now();

        // Send button query every 50ms
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_query).count() >= 50) {
            last_query = now;
            uint8_t query[2] = {P4HID_PKT_MAGIC, P4HID_CMD_BUTTONS_STATE};
            write_packet(h, query, sizeof(query));
        }

        // Receive response
        struct sockaddr_in from;
        int fromlen = sizeof(from);
        int len = recvfrom(h->sock, (char*)buf, sizeof(buf), 0,
                           (struct sockaddr*)&from, &fromlen);
        if (len <= 0) { Sleep(1); continue; }

        // Decrypt if AES enabled
        if (h->aes_enabled) {
            uint8_t dec_buf[256];
            size_t dec_len = aes_decrypt(h, buf, len, dec_buf, sizeof(dec_buf));
            if (dec_len > 0 && dec_len <= sizeof(buf)) {
                memcpy(buf, dec_buf, dec_len);
                len = (int)dec_len;
            } else {
                continue;
            }
        }

        // Parse response
        for (int i = 0; i < len; i++) {
            if (!synced) {
                if (buf[i] == P4HID_PKT_MAGIC) {
                    synced = true;
                    pkt_len = 0;
                    pkt[pkt_len++] = buf[i];
                }
            } else {
                pkt[pkt_len++] = buf[i];
                if (pkt_len >= 2) {
                    uint8_t cmd = pkt[1];
                    uint8_t expected = 0;
                    switch (cmd) {
                        case P4HID_CMD_BUTTONS_STATE: expected = 5; break;
                        default: synced = false; pkt_len = 0; continue;
                    }
                    if (pkt_len >= expected) {
                        if (cmd == P4HID_CMD_BUTTONS_STATE) {
                            h->fw_real_buttons.store(pkt[2]);
                            h->fw_synth_buttons.store(pkt[3]);
                        }
                        synced = false;
                        pkt_len = 0;
                    }
                }
                if (pkt_len >= sizeof(pkt)) {
                    synced = false;
                    pkt_len = 0;
                }
            }
        }
    }
}

extern "C" {

P4HID_API void* P4HID_Connect(const char* ip, uint16_t udp_port, const uint8_t* aes_key)
{
    P4HID_Handle* h = new P4HID_Handle;
    h->sock = INVALID_SOCKET;
    h->aes_enabled = false;
    h->aes_alg = NULL;
    h->buttons = 0;
    h->resp_sock = INVALID_SOCKET;

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        delete h;
        return nullptr;
    }

    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        WSACleanup();
        delete h;
        return nullptr;
    }

    u_long mode = 1;
    ioctlsocket(sock, FIONBIO, &mode);

    h->sock = sock;
    h->addr.sin_family = AF_INET;
    h->addr.sin_port = htons(udp_port);
    inet_pton(AF_INET, ip, &h->addr.sin_addr);

    // Init AES if key provided
    if (aes_key) {
        memcpy(h->aes_key, aes_key, P4HID_AES_KEY_SIZE);
        NTSTATUS st = BCryptOpenAlgorithmProvider(&h->aes_alg, BCRYPT_AES_ALGORITHM, NULL, 0);
        if (BCRYPT_SUCCESS(st)) {
            st = BCryptSetProperty(h->aes_alg, BCRYPT_CHAINING_MODE,
                                   (PUCHAR)BCRYPT_CHAIN_MODE_CBC, sizeof(BCRYPT_CHAIN_MODE_CBC), 0);
            if (BCRYPT_SUCCESS(st)) {
                h->aes_enabled = true;
            } else {
                BCryptCloseAlgorithmProvider(h->aes_alg, 0);
                h->aes_alg = NULL;
            }
        }
    }

    // Start reader thread
    h->reader_running.store(true);
    h->reader_thread = std::thread(reader_thread_fn, h);

    return h;
}

P4HID_API void P4HID_Disconnect(void* handle)
{
    if (!handle) return;
    P4HID_Handle* h = (P4HID_Handle*)handle;

    h->reader_running.store(false);
    if (h->reader_thread.joinable()) h->reader_thread.join();

    if (h->aes_alg) BCryptCloseAlgorithmProvider(h->aes_alg, 0);
    if (h->resp_sock != INVALID_SOCKET) closesocket(h->resp_sock);
    if (h->sock != INVALID_SOCKET) closesocket(h->sock);
    WSACleanup();
    delete h;
}

P4HID_API bool P4HID_Move(void* handle, int16_t dx, int16_t dy)
{
    if (!handle) return false;
    P4HID_Handle* h = (P4HID_Handle*)handle;
    uint8_t pkt[8] = {
        P4HID_PKT_MAGIC, P4HID_CMD_MOVE,
        (uint8_t)(dx & 0xFF), (uint8_t)((dx >> 8) & 0xFF),
        (uint8_t)(dy & 0xFF), (uint8_t)((dy >> 8) & 0xFF),
        0x00, 0x00
    };
    return write_packet(h, pkt, sizeof(pkt));
}

P4HID_API bool P4HID_Buttons(void* handle, uint8_t buttons)
{
    if (!handle) return false;
    P4HID_Handle* h = (P4HID_Handle*)handle;
    h->buttons = buttons;
    uint8_t pkt[5] = {
        P4HID_PKT_MAGIC, P4HID_CMD_BUTTONS,
        buttons, 0x00, 0x00
    };
    return write_packet(h, pkt, sizeof(pkt));
}

P4HID_API bool P4HID_Scroll(void* handle, int8_t wheel)
{
    if (!handle) return false;
    P4HID_Handle* h = (P4HID_Handle*)handle;
    uint8_t pkt[5] = {
        P4HID_PKT_MAGIC, P4HID_CMD_SCROLL,
        (uint8_t)wheel, 0x00, 0x00
    };
    return write_packet(h, pkt, sizeof(pkt));
}

P4HID_API bool P4HID_MoveButtonsScroll(void* handle, int16_t dx, int16_t dy, uint8_t buttons, int8_t wheel)
{
    if (!handle) return false;
    P4HID_Handle* h = (P4HID_Handle*)handle;
    h->buttons = buttons;
    uint8_t pkt[9] = {
        P4HID_PKT_MAGIC, P4HID_CMD_MOVE_BUTTONS,
        (uint8_t)(dx & 0xFF), (uint8_t)((dx >> 8) & 0xFF),
        (uint8_t)(dy & 0xFF), (uint8_t)((dy >> 8) & 0xFF),
        buttons, (uint8_t)wheel, 0x00
    };
    return write_packet(h, pkt, sizeof(pkt));
}

P4HID_API uint8_t P4HID_GetButtons(void* handle)
{
    if (!handle) return 0;
    P4HID_Handle* h = (P4HID_Handle*)handle;
    return h->fw_real_buttons.load() | h->fw_synth_buttons.load();
}

// ========================================================================
// Raw count streaming
// ========================================================================

P4HID_API bool P4HID_StartRawStream(void* handle, uint8_t interval)
{
    if (!handle) return false;
    P4HID_Handle* h = (P4HID_Handle*)handle;
    uint8_t pkt[4] = {P4HID_PKT_MAGIC, P4HID_CMD_STREAM_RAW, 0x01, interval};
    return write_packet(h, pkt, sizeof(pkt));
}

P4HID_API bool P4HID_StopRawStream(void* handle)
{
    if (!handle) return false;
    P4HID_Handle* h = (P4HID_Handle*)handle;
    uint8_t pkt[4] = {P4HID_PKT_MAGIC, P4HID_CMD_STREAM_RAW, 0x00, 0x00};
    return write_packet(h, pkt, sizeof(pkt));
}

P4HID_API void P4HID_SetRawCallback(void* handle, P4HID_RawCallback cb, void* user_data)
{
    // Raw callback is stored in the ABCurves Planner handle, not here.
    // This function is a placeholder for direct raw streaming without ABCurves.
    (void)handle; (void)cb; (void)user_data;
}

// ========================================================================
// ABCurves Planner
// ========================================================================

P4HID_API void* ABC_Create(const char* model_dir)
{
    if (!model_dir) return nullptr;

    // Resolve absolute path relative to DLL location
    HMODULE hDll = GetModuleHandleA("P4HID_API");
    char dll_path[MAX_PATH] = {};
    GetModuleFileNameA(hDll, dll_path, MAX_PATH);
    std::string dll_dir(dll_path);
    auto pos = dll_dir.find_last_of("\\/");
    if (pos != std::string::npos) dll_dir = dll_dir.substr(0, pos + 1);

    std::string abs_dir = dll_dir + model_dir;
    printf("[ABC] Model dir: %s\n", abs_dir.c_str());

    // Check if engine file exists
    std::string engine_path = abs_dir + "\\planner_tcn.engine";
    FILE* f = fopen(engine_path.c_str(), "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fclose(f);
        printf("[ABC] Engine file found: %s (%ld bytes)\n", engine_path.c_str(), sz);
    } else {
        printf("[ABC] Engine file NOT found: %s\n", engine_path.c_str());
    }

    try {
        abc::Planner* planner = new abc::Planner(abs_dir, true);
        if (planner->engine_loaded()) {
            printf("[ABC] Planner created with TensorRT engine\n");
            return planner;
        }
        printf("[ABC] TensorRT failed, trying CPU fallback\n");
        delete planner;
        planner = new abc::Planner(abs_dir, false);
        return planner;
    } catch (const std::exception& e) {
        printf("[ABC] Planner exception: %s\n", e.what());
        return nullptr;
    } catch (...) {
        printf("[ABC] Planner unknown exception\n");
        return nullptr;
    }
}

P4HID_API void ABC_Destroy(void* planner)
{
    if (planner) delete static_cast<abc::Planner*>(planner);
}

P4HID_API void ABC_Reset(void* planner)
{
    if (planner) static_cast<abc::Planner*>(planner)->reset();
}

P4HID_API void ABC_FeedRaw(void* planner, int16_t dx, int16_t dy)
{
    if (planner) static_cast<abc::Planner*>(planner)->feed_raw(dx, dy);
}

P4HID_API bool ABC_HasOnset(void* planner)
{
    return planner ? static_cast<abc::Planner*>(planner)->onset_detected() : false;
}

P4HID_API void ABC_ArmBTrigger(void* planner, float target_x, float target_y, float target_radius)
{
    if (planner) static_cast<abc::Planner*>(planner)->arm_btrigger(target_x, target_y, target_radius);
}

P4HID_API bool ABC_BFired(void* planner)
{
    return planner ? static_cast<abc::Planner*>(planner)->b_fired() : false;
}

static int s_last_duration = 0;
static int s_last_head = -1;

P4HID_API int ABC_Generate(void* planner, float target_x, float target_y,
                           float target_radius, float progress,
                           uint32_t seed, int16_t* out_dxdy, int max_pairs, int head)
{
    if (!planner || !out_dxdy || max_pairs <= 0) return 0;

    auto* p = static_cast<abc::Planner*>(planner);
    auto intent = p->generate(target_x, target_y, target_radius, progress, seed, head);

    int pairs = std::min(intent.intent.duration_ms, max_pairs);
    for (int i = 0; i < pairs; ++i) {
        out_dxdy[i * 2 + 0] = (int16_t)std::round(intent.intent.smooth_dxdy[i * 2 + 0]);
        out_dxdy[i * 2 + 1] = (int16_t)std::round(intent.intent.smooth_dxdy[i * 2 + 1]);
    }
    s_last_duration = intent.intent.duration_ms;
    s_last_head = intent.intent.head;
    return pairs;
}

P4HID_API int ABC_GetDuration(void* planner)
{
    return s_last_duration;
}

P4HID_API int ABC_GetHead(void* planner)
{
    return s_last_head;
}

P4HID_API double ABC_GetInferenceTimeUs(void* planner)
{
    return planner ? static_cast<abc::Planner*>(planner)->last_inference_us() : 0;
}

P4HID_API bool ABC_EngineLoaded(void* planner)
{
    return planner ? static_cast<abc::Planner*>(planner)->engine_loaded() : false;
}

P4HID_API int ABC_GetPrefixCount(void* planner)
{
    return planner ? static_cast<abc::Planner*>(planner)->prefix_count() : 0;
}

P4HID_API bool P4HID_SendABCIntent(void* handle, const int16_t* dxdy, uint16_t count, uint32_t seed)
{
    if (!handle || !dxdy || count == 0) return false;
    P4HID_Handle* h = (P4HID_Handle*)handle;

    /* Packet: [0xAA][0x07][seed:4 LE][count:2 LE][dx0:4 Q16][dy0:4 Q16]... */
    size_t pkt_size = 8 + (size_t)count * 8;
    if (pkt_size > 1024) return false;

    uint8_t pkt[1024];
    pkt[0] = P4HID_PKT_MAGIC;
    pkt[1] = P4HID_CMD_ABC_INTENT;
    pkt[2] = (uint8_t)(seed & 0xFF);
    pkt[3] = (uint8_t)((seed >> 8) & 0xFF);
    pkt[4] = (uint8_t)((seed >> 16) & 0xFF);
    pkt[5] = (uint8_t)((seed >> 24) & 0xFF);
    pkt[6] = (uint8_t)(count & 0xFF);
    pkt[7] = (uint8_t)((count >> 8) & 0xFF);

    for (uint16_t i = 0; i < count; i++) {
        /* Convert int16 raw counts to Q16 for the Renderer */
        int32_t dx_q16 = (int32_t)dxdy[i * 2 + 0] << 16;
        int32_t dy_q16 = (int32_t)dxdy[i * 2 + 1] << 16;
        size_t off = 8 + (size_t)i * 8;
        pkt[off + 0] = (uint8_t)(dx_q16 & 0xFF);
        pkt[off + 1] = (uint8_t)((dx_q16 >> 8) & 0xFF);
        pkt[off + 2] = (uint8_t)((dx_q16 >> 16) & 0xFF);
        pkt[off + 3] = (uint8_t)((dx_q16 >> 24) & 0xFF);
        pkt[off + 4] = (uint8_t)(dy_q16 & 0xFF);
        pkt[off + 5] = (uint8_t)((dy_q16 >> 8) & 0xFF);
        pkt[off + 6] = (uint8_t)((dy_q16 >> 16) & 0xFF);
        pkt[off + 7] = (uint8_t)((dy_q16 >> 24) & 0xFF);
    }

    return write_packet(h, pkt, pkt_size);
}

} // extern "C"
