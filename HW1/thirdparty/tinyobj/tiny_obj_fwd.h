#ifndef TINY_OBJ_FWD_H
#define TINY_OBJ_FWD_H

#include <vector>
#include <string>

namespace tinyobj {

#ifdef TINYOBJLOADER_USE_DOUBLE
//#pragma message "using double"
typedef double real_t;
#else
//#pragma message "using float"
typedef float real_t;
#endif

struct joint_and_weight_t {
  int joint_id;
  real_t weight;
};

struct skin_weight_t {
  int vertex_id;  // Corresponding vertex index in `attrib_t::vertices`.
                  // Compared to `index_t`, this index must be positive and
                  // start with 0(does not allow relative indexing)
  std::vector<joint_and_weight_t> weightValues;
};

struct attrib_t {
  std::vector<real_t> vertices;  // 'v'(xyz)

    // For backward compatibility, we store vertex weight in separate array.
  std::vector<real_t> vertex_weights;  // 'v'(w)
  std::vector<real_t> normals;         // 'vn'
  std::vector<real_t> texcoords;       // 'vt'(uv)

  // For backward compatibility, we store texture coordinate 'w' in separate
  // array.
  std::vector<real_t> texcoord_ws;  // 'vt'(w)
  std::vector<real_t> colors;       // extension: vertex colors

  //
  // TinyObj extension.
  //

  // NOTE(syoyo): array index is based on the appearance order.
  // To get a corresponding skin weight for a specific vertex id `vid`,
  // Need to reconstruct a look up table: `skin_weight_t::vertex_id` == `vid`
  // (e.g. using std::map, std::unordered_map)
  std::vector<skin_weight_t> skin_weights;

  attrib_t() {}

  //
  // For pybind11
  //
  const std::vector<real_t> &GetVertices() const { return vertices; }

  const std::vector<real_t> &GetVertexWeights() const { return vertex_weights; }
};

struct tag_t {
  std::string name;

  std::vector<int> intValues;
  std::vector<real_t> floatValues;
  std::vector<std::string> stringValues;
};

// Index struct to support different indices for vtx/normal/texcoord.
// -1 means not used.
struct index_t {
  int vertex_index;
  int normal_index;
  int texcoord_index;
};

struct mesh_t {
  std::vector<index_t> indices;
  std::vector<unsigned int>
      num_face_vertices;          // The number of vertices per
                                  // face. 3 = triangle, 4 = quad, ...
  std::vector<int> material_ids;  // per-face material ID
  std::vector<unsigned int> smoothing_group_ids;  // per-face smoothing group
                                                  // ID(0 = off. positive value
                                                  // = group id)
  std::vector<tag_t> tags;                        // SubD tag
};

// struct path_t {
//  std::vector<int> indices;  // pairs of indices for lines
//};

struct lines_t {
  // Linear flattened indices.
  std::vector<index_t> indices;        // indices for vertices(poly lines)
  std::vector<int> num_line_vertices;  // The number of vertices per line.
};

struct points_t {
  std::vector<index_t> indices;  // indices for points
};

struct shape_t {
  std::string name;
  mesh_t mesh;
  lines_t lines;
  points_t points;
};

}

#endif