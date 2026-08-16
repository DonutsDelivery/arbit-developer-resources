#include "backend.h"

namespace arbitgpu
{

BackendInfo queryNativeBackend()
{
    BackendInfo result;
    result.error = "native GPU backend not compiled in";
    return result;
}

BackendSelfTest runNativeBackendSelfTest()
{
    BackendSelfTest result;
    result.error = "native GPU backend not compiled in";
    return result;
}

} // namespace arbitgpu
