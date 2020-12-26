#include <iostream>
#include <numeric>

inline bool is_pythag_multiple(int x, int y, int z) noexcept { return std::gcd(std::gcd(x, y), std::gcd(x, z)) > 1; }

inline bool is_pythag_triple(int x, int y, int z) noexcept { return x * x + y * y == z * z; }

template <typename F>
inline void find_pythag_triples(F&& process_triple) {
  for (int z = 1;; ++z)
    for (int x = 1; x != z + 1; ++x)
      for (int y = x; y != z + 1; ++y)
        if (is_pythag_triple(x, y, z) && !is_pythag_multiple(x, y, z) && !process_triple(x, y, z)) return;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " TopN\n"; // NOLINT
    return EXIT_FAILURE;
  }
  int n = std::atoi(argv[1]); // NOLINT

  if (n < 1) {
    std::cerr << "N should be 1 or greater\n";
  }

  std::cout << "Finding top " << n << " Pythagorean triples\n";

  find_pythag_triples([&n](int x, int y, int z) {
    std::cout << '(' << x << ',' << y << ',' << z << ')' << '\n';
    return --n;
  });
}
