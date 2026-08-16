#include "PrivateInheritedPayload.h"
#include <cassert>
int main()
{
#if defined(_WIN32)
    static_assert(programmableruntime::privatepayload::platformCapability()
                  == programmableruntime::privatepayload::Capability::unavailable);
#else
    static_assert(programmableruntime::privatepayload::platformCapability()
                  == programmableruntime::privatepayload::Capability::available);
#endif
}