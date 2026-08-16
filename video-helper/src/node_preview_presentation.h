#pragma once

#include <algorithm>
#include <string>

namespace videopreview
{
enum class Background { checker = 0, black = 1, white = 2 };
enum class Layout { hidden = 0, split = 1, overlay = 2 };

struct State
{
    Layout layout = Layout::hidden;
    Background background = Background::checker;
    float zoom = 1.0f;
    float panX = 0.0f;
    float panY = 0.0f;
    float split = 0.5f;
};

inline bool normalize(State& state, std::string& error)
{
    if (state.layout < Layout::hidden || state.layout > Layout::overlay)
    { error = "invalid node preview layout"; return false; }
    if (state.background < Background::checker || state.background > Background::white)
    { error = "invalid node preview background"; return false; }
    state.zoom = std::clamp(state.zoom, 0.05f, 32.0f);
    state.panX = std::clamp(state.panX, -4.0f, 4.0f);
    state.panY = std::clamp(state.panY, -4.0f, 4.0f);
    state.split = std::clamp(state.split, 0.15f, 0.85f);
    error.clear();
    return true;
}

inline const char* backendAvailability(bool metal, bool nativeResourceAdmitted = false)
{
    if (! metal) return "opengl";
    return nativeResourceAdmitted ? "metal-native" : "unavailable-native-metal-resource";
}

inline bool presentationAdmitted(bool metal, bool nativeResourceAdmitted,
                                 const State& state, std::string& error)
{
    if (! metal || state.layout == Layout::hidden || nativeResourceAdmitted)
    { error.clear(); return true; }
    error = "node preview presentation is unavailable on native Metal: retained inspection resource is not a Metal texture";
    return false;
}
} // namespace videopreview
