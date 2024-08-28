#include <cstdint>
#include <iostream>
#include <string>

#include "image.hpp"
#include "loader.hpp"
#include "rasterizer.hpp"
#include "renderer.hpp"

void PrintTask(const Loader& loader)
{
    std::string sephead = "======================Config======================\n";
    std::string sep = "==================================================\n";
    std::string msg = "Running task with the following configuration:\n" + sephead + loader.Info() + sep;
    std::cout << msg;
}

void PrintTaskTriangle(const Triangle& trig)
{
    std::string sephead = "=====================Triangle=====================\n";
    std::string sep = "==================================================\n";
    std::string msg = sephead + 
        "Vertex 1 position: " + ToStr(trig.pos[0]) + "\n" +
        "Vertex 2 position: " + ToStr(trig.pos[1]) + "\n" +
        "Vertex 3 position: " + ToStr(trig.pos[2]) + "\n"
        + sep;
    std::cout << msg;
}

void PrintTaskTransformTest(const glm::vec3 input, const glm::vec4 output, const glm::vec3 expected)
{
    std::string sephead = "===============Task: Transform Test===============\n";
    std::string sep = "==================================================\n";
    glm::vec4 expected4(expected, 1);
    glm::vec4 normalizedOutput = output / output.w;
    std::string msg = sephead + 
        "Input: " + ToStr(input) + "\n" +
        "Output: " + ToStr(normalizedOutput) + "\n" +
        "Expected: " + ToStr(expected4) + "\n"
        + sep;
    std::cout << msg;
}

void Renderer::Render(int argc, char** argv)
{
    std::string modelName;
    std::string yamlConfigName = "config.yaml";

    if (argc != 1)
    {
        yamlConfigName = argv[1];
        std::cout << "using customized config name" << yamlConfigName << std::endl;
    }

    Loader loader(yamlConfigName);
    bool success = loader.Load();

    if (success)
    {
        PrintTask(loader);
        Image image(loader.GetWidth(), loader.GetHeight());

        Rasterizer rasterizer(loader);

        glm::mat4x4 viewxprojection{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };

        if (loader.GetType() == TestType::TRIANGLE)
        {
            // notice that glm::mat4x4 is column-major, so the actual matrix is the transpose of the matrix read off
            uint32_t halfWidth = loader.GetWidth() / 2;
            uint32_t halfHeight = loader.GetHeight() / 2;
            viewxprojection = glm::mat4x4{
                halfWidth, 0         , 0, 0,
                0        , halfHeight, 0, 0, 
                0        , 0         , 0, 0,             // discard z values
                halfWidth, halfHeight, 0, 1
            };
            rasterizer.model.push_back(glm::mat4x4(1.0f));      // Add an identity model matrix to avoid special judgement below
        }
        else
        {
            // First load the matrices to the rasterizer
            for (size_t index = 0; index != loader.GetTransforms().size(); ++index)
            {
                MeshTransform transform = loader.GetTransforms()[index];
                rasterizer.AddModel(transform);
            }

            rasterizer.SetView();
            rasterizer.SetProjection();
            rasterizer.SetScreenSpace();

            // Compose the matrices
            viewxprojection = rasterizer.screenspace * rasterizer.projection * rasterizer.view;
        }
        
        // If this is test on transforms, then do not need to iterate over the meshes
        if (loader.GetType() == TestType::TRANSFORM_TEST)
        {
            glm::vec3 input = loader.GetTestInput();
            glm::vec3 expected = loader.GetTestExpected();
            glm::vec4 input4(input, 1);

            if (rasterizer.model.size() == 0)
                throw std::runtime_error("No model matrix specified for transform test");

            glm::vec4 output = viewxprojection * rasterizer.model[0] * input4;
            PrintTaskTransformTest(input, output, expected);
        }
        else 
        {
            auto& shapes = loader.GetShapes();
            auto& attribs = loader.GetAttribs();

            if (loader.GetType() == TestType::SHADING_DEPTH || loader.GetType() == TestType::SHADING)
                rasterizer.InitZBuffer(rasterizer.ZBuffer);

            std::vector<Triangle> transformedTrigs;
            std::vector<Triangle> originalTrigs;
            
            const size_t fv = 3;
            for (size_t s = 0; s < shapes.size(); s++) 
            {
                if (loader.GetType() == TestType::SHADING)
                {
                    transformedTrigs.clear();
                    originalTrigs.clear();
                    transformedTrigs.reserve(shapes.size());
                    originalTrigs.reserve(shapes.size());
                }

                // Loop over faces(polygon)
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) 
                {
                    // Loop over vertices in the face.
                    Triangle transformed, original;
                    for (size_t v = 0; v < fv; v++) 
                    {
                        // access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        tinyobj::real_t vx = attribs.vertices[3 * size_t(idx.vertex_index) + 0];
                        tinyobj::real_t vy = attribs.vertices[3 * size_t(idx.vertex_index) + 1];
                        tinyobj::real_t vz = attribs.vertices[3 * size_t(idx.vertex_index) + 2];
                        glm::vec4 vec(vx, vy, vz, 1);
                        
                        // init to identity so that the program will no crash even without model matrices being added
                        glm::mat4 modelMat = glm::mat4(1.f);
                        if (rasterizer.model.size() > s)
                            modelMat = rasterizer.model[s];

                        if (loader.GetType() == TestType::TRIANGLE)
                            transformed.pos[v] = viewxprojection * vec;
                        else
                            transformed.pos[v] = viewxprojection * modelMat * vec;

                        original.pos[v] = modelMat * vec;

                        if (idx.normal_index >= 0) 
                        {
                            tinyobj::real_t nx = attribs.normals[3 * size_t(idx.normal_index) + 0];
                            tinyobj::real_t ny = attribs.normals[3 * size_t(idx.normal_index) + 1];
                            tinyobj::real_t nz = attribs.normals[3 * size_t(idx.normal_index) + 2];
                            original.normal[v] = modelMat * glm::vec4(nx, ny, nz, 1);
                        }
                    }

                    transformed.Homogenize();

#if defined PRINT_TRIG_DETAIL
                    PrintTaskTriangle(transformed);
#endif

                    if (loader.GetType() == TestType::TRIANGLE || loader.GetType() == TestType::TRANSFORM)
                        rasterizer.DrawPrimitiveRaw(image, transformed, loader.GetAntiAliasConfig(), loader.GetSpp());
                    else if (loader.GetType() == TestType::SHADING_DEPTH || loader.GetType() == TestType::SHADING)
                        rasterizer.DrawPrimitiveDepth(transformed, original, rasterizer.ZBuffer);
                    
                    if (loader.GetType() == TestType::SHADING)
                    {
                        transformedTrigs.push_back(transformed);
                        originalTrigs.push_back(original);
                    }

                    index_offset += fv;
                }

                if (loader.GetType() == TestType::SHADING)
                    for (size_t i = 0; i < transformedTrigs.size(); ++i)
                        rasterizer.DrawPrimitiveShaded(transformedTrigs[i], originalTrigs[i], image);
            }
        }

        if (loader.GetType() == TestType::SHADING_DEPTH)
            rasterizer.ZBuffer.Write();
        else if (loader.GetType() != TestType::TRANSFORM_TEST)
            image.Write();
    }
}
