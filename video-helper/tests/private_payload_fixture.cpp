#include "PrivateInheritedPayload.h"
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <unistd.h>

int main(int argc, char** argv)
{
    std::array<uint8_t, programmableruntime::privatepayload::packetSize> packet{};
    if (!programmableruntime::privatepayload::readOneShot(3, packet)) return 2;
    const bool eof = (::read(3, packet.data(), 1) < 0 && errno == EBADF);
    bool leaked = false;
    for (int i=0; i<argc; ++i) leaked |= std::string(argv[i]).find("PRIVATE_TEST_SECRET") != std::string::npos;
    extern char** environ;
    for (char** p=environ; *p; ++p) leaked |= std::string(*p).find("PRIVATE_TEST_SECRET") != std::string::npos;
    std::printf("bytes=40 closed=%d leaked=%d first=%u last=%u\n", eof, leaked,
                unsigned(packet.front()), unsigned(packet.back()));
    return leaked || !eof;
}