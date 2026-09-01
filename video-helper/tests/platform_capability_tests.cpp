#include "PrivateInheritedPayload.h"
#include <cassert>
int main()
{
    static_assert(programmableruntime::privatepayload::platformCapability()
                  == programmableruntime::privatepayload::Capability::available);
}