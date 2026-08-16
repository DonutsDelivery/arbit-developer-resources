#include "node_preview_presentation.h"
#include <cstdlib>
#include <iostream>

static void check(bool value, const char* message)
{
    if (! value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}

int main()
{
    videopreview::State state;
    state.layout = videopreview::Layout::split;
    state.background = videopreview::Background::white;
    state.zoom = 100.0f; state.panX = -8.0f; state.panY = 9.0f; state.split = 0.99f;
    std::string error;
    check(videopreview::normalize(state, error), "valid state accepted");
    check(state.zoom == 32.0f && state.panX == -4.0f && state.panY == 4.0f,
          "independent view controls clamp safely");
    check(state.split == 0.85f, "split clamps safely");
    state.layout = static_cast<videopreview::Layout>(99);
    check(! videopreview::normalize(state, error) && error == "invalid node preview layout",
          "unknown layout rejected");
    check(std::string(videopreview::backendAvailability(false)) == "opengl",
          "OpenGL presentation advertised");
    check(std::string(videopreview::backendAvailability(true)) ==
              "unavailable-native-metal-resource",
          "unadmitted Metal resource remains explicit");
    check(std::string(videopreview::backendAvailability(true, true)) == "metal-native",
          "genuine Metal resource is admitted");
    state.layout = videopreview::Layout::split;
    check(! videopreview::presentationAdmitted(true, false, state, error)
              && error.find("not a Metal texture") != std::string::npos,
          "strict Metal rejects non-native resources honestly");
    check(videopreview::presentationAdmitted(true, true, state, error),
          "strict Metal admits a genuine native texture");
    check(videopreview::presentationAdmitted(false, false, state, error),
          "OpenGL presents its retained native texture without Metal admission");
    state.layout = videopreview::Layout::hidden;
    check(videopreview::presentationAdmitted(true, false, state, error),
          "hidden preview preserves terminal viewer/export with no fake Metal handle");
    std::cout << "node preview presentation tests passed\n";
}
