#pragma once

#ifdef P4HIDAPI_EXPORTS
#define P4HID_API __declspec(dllexport)
#else
#define P4HID_API __declspec(dllimport)
#endif

#include <cstdint>

// Protocol constants (must match firmware)
#define P4HID_PKT_MAGIC        0xAA
#define P4HID_CMD_MOVE         0x01
#define P4HID_CMD_BUTTONS      0x02
#define P4HID_CMD_SCROLL       0x03
#define P4HID_CMD_MOVE_BUTTONS 0x04
#define P4HID_CMD_BUTTONS_STATE 0x05
#define P4HID_DEFAULT_UDP_PORT 4444

// AES-128-CBC constants
#define P4HID_AES_KEY_SIZE   16
#define P4HID_AES_NONCE_SIZE 12

extern "C" {
    // Connect to P4HID device via UDP. Returns handle, or nullptr on failure.
    // ip: device IP address (e.g. "192.168.4.1")
    // udp_port: UDP port (default 4444)
    // aes_key: 16-byte AES key (NULL = plaintext)
    P4HID_API void* P4HID_Connect(const char* ip, uint16_t udp_port = P4HID_DEFAULT_UDP_PORT, const uint8_t* aes_key = nullptr);

    // Disconnect from device.
    P4HID_API void P4HID_Disconnect(void* handle);

    // Send relative mouse movement. Non-blocking.
    P4HID_API bool P4HID_Move(void* handle, int16_t dx, int16_t dy);

    // Set mouse buttons. Non-blocking.
    P4HID_API bool P4HID_Buttons(void* handle, uint8_t buttons);

    // Send scroll wheel delta. Non-blocking.
    P4HID_API bool P4HID_Scroll(void* handle, int8_t wheel);

    // Send combined move + buttons + scroll. Lowest latency.
    P4HID_API bool P4HID_MoveButtonsScroll(void* handle, int16_t dx, int16_t dy, uint8_t buttons, int8_t wheel);

    // Get firmware button state (real | synth). Polled every 50ms.
    P4HID_API uint8_t P4HID_GetButtons(void* handle);

    // ========================================================================
    // Raw count streaming
    // ========================================================================

    // New firmware command for raw count streaming
    #define P4HID_CMD_STREAM_RAW  0x06

    // Raw count callback type
    typedef void (*P4HID_RawCallback)(int16_t dx, int16_t dy, uint8_t buttons,
                                      uint32_t tick, void* user_data);

    // Enable raw count streaming from ESP32-P4 firmware
    P4HID_API bool P4HID_StartRawStream(void* handle, uint8_t interval = 1);

    // Disable raw count streaming
    P4HID_API bool P4HID_StopRawStream(void* handle);

    // Set callback for raw count stream
    P4HID_API void P4HID_SetRawCallback(void* handle, P4HID_RawCallback cb, void* user_data);
}
