#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

struct vec3 {
  int x, y, z;

  bool operator==(const vec3& o) const noexcept { return x == o.x && y == o.y && z == o.z; }
  vec3 operator+(const vec3& o) const noexcept { return {x + o.x, y + o.y, z + o.z}; }

  static vec3 min(const vec3& a, const vec3& b) noexcept {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
  }

  static vec3 max(const vec3& a, const vec3& b) noexcept {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
  }

  [[nodiscard]] float dist(const vec3& o) const noexcept {
    return std::hypot(float(x - o.x), float(y - o.y), float(z - o.z));
  }

  // limited to immediate neighbours: (int)dx,dy,dz: -1 => 1. uses lookup table
  [[nodiscard]] float fast_dist(const vec3& o) const noexcept {
    return hypot[x - o.x + 1][y - o.y + 1][z - o.z + 1];
  }

  inline static float hypot[3][3][3]{}; // NOLINT

  static bool hypot_init() noexcept {
    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++) {
          hypot[dx + 1][dy + 1][dz + 1] = std::hypot(float(dx), float(dy), float(dz));
        }
    return true;
  }

  inline static bool hypot_initialized = hypot_init();

  friend std::ostream& operator<<(std::ostream& os, const vec3& v3) {
    return os << "[" << v3.x << ", " << v3.y << ", " << v3.z << "]";
  }
};

class grid {
  struct cell; // fwd declaration

public:
  cell& operator[](vec3 const& index) noexcept {
    return cells[index.x * sy_ * sz_ + index.y * sz_ + index.z];
  }

  const cell& operator[](vec3 const& index) const noexcept {
    return cells[index.x * sy_ * sz_ + index.y * sz_ + index.z];
  }

  template <typename Filter>
  [[nodiscard]] std::vector<vec3> all_coords(const Filter& filter) const {
    vec3              d = dims();
    std::vector<vec3> all{};
    for (int x = 0; x < d.x; ++x)
      for (int y = 0; y < d.y; ++y)
        for (int z = 0; z < d.z; ++z) {
          vec3 coord{x, y, z};
          if (filter(coord, coord)) all.emplace_back(coord);
        }
    return all;
  }

  friend std::istream& operator>>(std::istream& istream, grid& g) {
    istream >> g.sx_ >> g.sy_ >> g.sz_;
    g.cells.resize(g.sx_ * g.sy_ * g.sz_);
    istream >> std::boolalpha;
    int i = 0;
    for (int x = 0; x < g.sx_; ++x)
      for (int y = 0; y < g.sy_; ++y)
        for (int z = 0; z < g.sz_; ++z) istream >> g.cells[i++];
    return istream;
  }

  [[nodiscard]] vec3 dims() const { return {sx_, sy_, sz_}; }

  template <typename Filter>
  [[nodiscard]] std::vector<vec3> find_path(const vec3& start, const vec3& end,
                                            const Filter& filter) const {

    if (!contains(start) || !contains(end))
      throw std::invalid_argument("start and/or end not contained in grid!");

    if (!filter(start, start) || !filter(end, end))
      throw std::invalid_argument("start and/or end fail cell filter for grid!");

    auto vertices = find_path_search(start, end, filter);
    return find_path_extract(end, vertices);
  }

  [[nodiscard]] bool path_segment_filter(const vec3& from, const vec3& to) const {
    if (from.y != to.y)
      // If the movement is vertical, then perform no diagonal check
      return is_free_floor(to);

    // Check if all cells we're moving through are floors
    // important when moving diagonally
    auto min = vec3::min(from, to);
    auto max = vec3::max(from, to);

    for (int x = min.x; x <= max.x; ++x)
      for (int z = min.z; z <= max.z; ++z)
        if (!is_free_floor({x, min.y, z})) return false;
    return true;
  }

private:
  int sx_{}, sy_{}, sz_{};

  struct cell {
    bool occupied;
    bool walkable_surface;

    friend std::istream& operator>>(std::istream& is, cell& c) {
      return is >> c.occupied >> c.walkable_surface;
    }
  };

  std::vector<cell> cells;

  struct vertex {
    vec3  coord;
    float dist;

    bool operator<(vertex const& o) const { return dist > o.dist; } // min-heap!

    friend std::ostream& operator<<(std::ostream& os, const vertex& v) {
      return os << v.coord << ": " << v.dist;
    }
  };

  template <typename T>
  struct grid_vector {
    explicit grid_vector(vec3 dims, const T& defv = T())
        : sx_(dims.x), sy_(dims.y), sz_(dims.z), data_(sx_ * sy_ * sz_, defv) {}

    T& operator[](const vec3& index) {
      return data_[index.x * sy_ * sz_ + index.y * sz_ + index.z];
    }

    const T& operator[](const vec3& index) const {
      return data_[index.x * sy_ * sz_ + index.y * sz_ + index.z];
    }

  private:
    int            sx_, sy_, sz_;
    std::vector<T> data_;
  };

  template <typename Filter>
  [[nodiscard]] grid_vector<vertex> find_path_search(const vec3& start, const vec3& end,
                                                     const Filter& filter) const {
    // previuous coord / finalised dist to start could be added to
    // grid.cells but that would make multi-threaded path finding very hard
    grid_vector<vertex> vertices(dims(), {{-1, -1, -1}, std::numeric_limits<float>::max()});

    // search queue: not prefilled, because not required and that
    // slows it down current coord / estimated dist to start
    std::priority_queue<vertex> queue;

    size_t max_queue_size = 0;

    vertices[start].dist = 0;
    queue.push({start, 0});

    while (!queue.empty()) {
      auto curr = queue.top();
      queue.pop();
      if (curr.dist != vertices[curr.coord].dist) continue; // lazy remove/skip of old queue value
      if (curr.coord == end) break;                         // we arrived. stop.

      foreach_neighbours(curr.coord, [&](const vec3& v) {
        if (filter(curr.coord, v)) {
          float new_dist = vertices[curr.coord].dist + curr.coord.fast_dist(v);
          if (new_dist < vertices[v].dist) {
            // update min distance to "start", record path "back"
            vertices[v] = {curr.coord, new_dist};
            queue.push({v, new_dist}); // leave old one in, to be lazily removed
            if (queue.size() > max_queue_size) max_queue_size = queue.size();
          }
        }
      });
    }
    return vertices; // uses NRVO
  }

  static std::vector<vec3> find_path_extract(const vec3& end, const grid_vector<vertex>& vertices) {
    std::vector<vec3> path;
    if (vertices[end].coord.x != -1) {
      vec3 current = end;
      while (current.x != -1) {
        path.push_back(current);
        current = vertices[current].coord;
      }
      std::reverse(path.begin(), path.end());
    }
    return path;
  }

  [[nodiscard]] bool is_free_floor(const vec3& pos) const {
    return pos.y > 0 && !(*this)[pos].occupied && (*this)[pos + vec3{0, -1, 0}].walkable_surface;
  }

  [[nodiscard]] bool contains(vec3 const& coord) const {
    // clang-format off
    return coord.x >= 0 && coord.x < sx_ &&
           coord.y >= 0 && coord.y < sy_ &&
           coord.z >= 0 && coord.z < sz_;
    // clang-format on
  }

  // faster to loop with callback than to materialise
  // a vector on heap and RVO return it
  template <typename Callback>
  void foreach_neighbours(const vec3& coord, const Callback& callback) const {
    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++) {
          if (dx == 0 && dy == 0 && dz == 0) continue; // ignore self
          auto new_coord = coord + vec3{dx, dy, dz};
          if (abs(dx) + abs(dy) + abs(dz) <= 2 && // connected
              contains(new_coord))                // within grid
            callback(new_coord);
        }
  }
}; // Grid

void bench_find_path(benchmark::State& state) {
  std::string_view grid_filename = "grid.txt";
  std::ifstream    gridFile(grid_filename.data());
  if (!gridFile.is_open()) {
    std::cerr << "Could not read from grid file: " << grid_filename << "\n";
    exit(EXIT_FAILURE);
  }
  auto g = grid{};
  gridFile >> g;
  vec3 start  = {9, 2, 6};  // NOLINT
  vec3 end    = {45, 2, 0}; // NOLINT
  auto filter = [&g](const vec3& from, const vec3& to) { return g.path_segment_filter(from, to); };
  for (auto _: state) {
    auto path = g.find_path(start, end, filter);
  }
}

// BENCHMARK(bench_find_path);

int main() {
  std::string_view grid_filename = "grid.txt";
  std::ifstream    gridFile(grid_filename.data());
  if (!gridFile.is_open()) {
    std::cerr << "Could not read from grid file: " << grid_filename << "\n";
    exit(EXIT_FAILURE);
  }
  grid g;
  gridFile >> g;
  auto filter = [&g](const vec3& from, const vec3& to) { return g.path_segment_filter(from, to); };

  vec3 start = {9, 2, 6};  // NOLINT
  vec3 end   = {45, 2, 0}; // NOLINT

  try {
    auto path = g.find_path(start, end, filter);
    std::cout << "best path is " << path.size() << " steps long. \n";
    for (auto& e: path) std::cout << e << "\n";
  } catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << '\n';
    return (EXIT_FAILURE);
  }
}
