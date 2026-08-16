#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

namespace
{
[[noreturn]] void exhaustAddressSpace()
{
    std::vector<void*> allocations;
    constexpr std::size_t chunkBytes = 1024u * 1024u;
    for (;;)
    {
        void* block = std::malloc(chunkBytes);
        if (block == nullptr) _exit(86);
        std::memset(block, 0xa5, chunkBytes);
        allocations.push_back(block);
    }
}

[[noreturn]] void exhaustCpu()
{
    volatile unsigned long long value = 1;
    for (;;)
        value = value * 6364136223846793005ULL + 1442695040888963407ULL;
}
}

int main(int argc, char** argv)
{
    if (argc != 2 || (std::strcmp(argv[1], "memory") != 0
                      && std::strcmp(argv[1], "cpu") != 0))
        return 2;

    std::string line;
    if (!std::getline(std::cin, line)) return 3;
    const auto idStart = line.find("\"id\"");
    const auto colon = line.find(':', idStart);
    if (idStart == std::string::npos || colon == std::string::npos) return 5;
    const auto id = std::strtoll(line.c_str() + colon + 1, nullptr, 10);
    const pid_t descendant = fork();
    if (descendant == 0)
        for (;;) pause();
    if (descendant < 0) return 84;

    std::cout << "{\"jsonrpc\":\"2.0\",\"id\":" << id
              << ",\"result\":{\"descendantPid\":"
              << descendant << "}}\n" << std::flush;

    if (!std::getline(std::cin, line)) return 4;
    if (std::strcmp(argv[1], "memory") == 0) exhaustAddressSpace();
    exhaustCpu();
}
