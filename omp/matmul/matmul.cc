#include "matmul.hh"

int CompareWays()
{
  mul::Mat mat1, mat2, answ;

  std::cin >> mat1 >> mat2 >> answ;

  if (mat1.getCols() != mat2.getRows())
  {
    std::cout << "Incompatible matrix sizes" << std::endl;
    return -1;
  }

  std::cout << "Naive impl\n";
  auto res = mul::Measure(mat1, mat2, mul::mulNaive);
  std::cout << res.second << " ms" << std::endl;
  assert(res.first == answ);

  std::cout << "Prom 16 + transpose + temp vars\n";
  res = mul::Measure(mat1, mat2, mul::mulProm16xTransp);
  std::cout << res.second << " ms" << std::endl;
  assert(res.first == answ);

  std::cout << "Naive omp\n";
  res = mul::Measure(mat1, mat2, mul::mulOMPNaive);
  std::cout << res.second << " ms" << std::endl;
  assert(res.first == answ);

  std::cout << "OMP promp 16 + transpose + temp vars\n";
  res = mul::Measure(mat1, mat2, mul::mulOMP16xTransp);
  std::cout << res.second << " ms" << std::endl;
  assert(res.first == answ);

  std::cout << "Prom 8 + transpose + temp vars + SIMD\n";
  res = mul::Measure(mat1, mat2, mul::mulProm8xTranspIntr);
  std::cout << res.second << " ms" << std::endl;
  assert(res.first == answ);

  std::cout << "OMP Prom 8 + transpose + temp vars + SIMD\n";
  res = mul::Measure(mat1, mat2, mul::mulOmpProm8xTranspIntr);
  std::cout << res.second << " ms" << std::endl;
  assert(res.first == answ);

  return 0;
}

template <std::forward_iterator It>
int VarSize(It beg, It end,
            std::pair<std::int32_t, std::int32_t> range_rnd = {0, 10})
{
  std::random_device dev{};
  std::mt19937_64 gen(dev());

  std::uniform_int_distribution<std::int32_t> dist(range_rnd.first,
                                                    range_rnd.second);
  auto &&rand_fill = [&dist, &gen](int, int) { return dist(gen); };
  struct Point
  {
    size_t size = 0;
    linal::ldbl time_ms = 0.0;
  };

  std::for_each(beg, end, [rand_fill](auto size) {
    mul::Mat mat1(size, size, rand_fill), mat2(size, size, rand_fill);

    auto [answ, ms] = mul::Measure(mat1, mat2, mul::mulOMP16xTransp);
    assert(answ == mul::mulNaive(mat1, mat2));

    std::cout << size << ", " << ms << std::endl;
  });

  return 0;
}

int main()
{
#if defined(CMP_WAYS)
  return CompareWays();
#else
  auto sizes = std::to_array<std::size_t>(
    {5,   6,   7,   8,   9,   10,  11,  12,   13,   14,  15,
     16,  17,  18,  19,  20,  30,  40,  50,   100,  150, 200,
     300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000});

  return VarSize(sizes.begin(), sizes.end());
#endif
}
