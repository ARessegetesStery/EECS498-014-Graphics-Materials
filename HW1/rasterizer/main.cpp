#include <iostream>

#include "renderer.hpp"

int main(int argc, char** argv)
{
    std::string configName = "config.yaml";
    if (argc > 1)
        configName = std::string(argv[1]) + ".yaml";

    Renderer renderer(configName);
    try
    {
        renderer.Render(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << "Rendering process failed..." << std::endl; 
        std::cerr << e.what() << std::endl;
    }
}
