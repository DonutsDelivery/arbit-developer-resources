#!/usr/bin/env python3

from pathlib import Path

root = Path(__file__).resolve().parents[1]
helper = (root / "src/helper_glfw.h").read_text()
mac_app = (root / "src/mac_helper_application.mm").read_text()
viewport = (root / "src/viewport.cpp").read_text()
exporter = (root / "src/exporter.cpp").read_text()
cmake = (root / "CMakeLists.txt").read_text()

assert "GLFW_COCOA_MENUBAR, GLFW_FALSE" in helper
assert "NSApplicationActivationPolicyAccessory" in mac_app
assert viewport.count("initializeHelperGlfw()") == 4
assert "glfwInit()" not in viewport
assert "initializeHelperGlfw()" in exporter
assert "glfwInit()" not in exporter
assert "src/mac_helper_application.mm" in cmake
