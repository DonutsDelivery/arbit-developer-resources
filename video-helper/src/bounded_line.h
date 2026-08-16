#pragma once

#include <cstddef>
#include <istream>
#include <vector>

namespace videohelper
{
enum class BoundedLineResult { line, end, oversized };

inline BoundedLineResult readBoundedLine (
    std::istream& input, std::vector<char>& destination, size_t maxBytes)
{
    destination.clear();
    // Allocate exactly the declared payload policy once. Keeping accepted
    // bytes in this fixed-size vector avoids implementation-defined string
    // capacity growth beyond maxBytes.
    destination.resize (maxBytes);
    size_t used = 0;
    char ch = 0;
    while (input.get (ch))
    {
        if (ch == '\n')
        {
            destination.resize (used);
            return BoundedLineResult::line;
        }
        if (used >= maxBytes)
        {
            destination.clear();
            return BoundedLineResult::oversized;
        }
        destination[used++] = ch;
    }
    if (used == 0) return BoundedLineResult::end;
    destination.resize (used);
    return BoundedLineResult::line;
}
} // namespace videohelper
