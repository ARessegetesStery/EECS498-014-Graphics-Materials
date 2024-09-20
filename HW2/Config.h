// Configurations for the ray tracing assignment

#pragma once

#include <string_view>


constexpr int SPP = 16;

constexpr int SEED = 42;
constexpr int RESOLUTION = 512;
constexpr int MAX_DEPTH = 8;
constexpr float RR = 0.8f;

constexpr std::string_view OBJ_PATH = "./models/cornellBox/CornellBox-Original.obj";
constexpr std::string_view MTL_SEARCH_DIR = "./models/cornellBox/";
constexpr std::string_view OUTPUT_PATH = "./binary.ppm";

constexpr bool DEBUG = false; // reserved for future use. Do not enable it currently.