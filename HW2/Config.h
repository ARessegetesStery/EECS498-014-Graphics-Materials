// Configurations for the ray tracing assignment

#pragma once

#include <string_view>

constexpr bool DEBUG = false;

constexpr int SPP = 16;
constexpr int RESOLUTION = 512;
constexpr int MAX_DEPTH = 8;
constexpr float RR = 0.8f;
constexpr std::string_view MODEL_PATH = "../../models/";