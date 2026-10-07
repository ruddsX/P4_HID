#ifndef PCH_H
#define PCH_H

#include <winsock2.h>
#include <ws2tcpip.h>
#include <bcrypt.h>
#include "framework.h"  // includes <windows.h> — safe after winsock2
#include "P4HID_API.h"
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "bcrypt.lib")

#endif //PCH_H
