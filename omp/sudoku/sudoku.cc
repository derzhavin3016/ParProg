#include <algorithm>
#include <array>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <optional>
#include <sstream>

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

constexpr size_t CELL_SIZE = 4;
constexpr size_t MAX_NUM_W = 2;
constexpr size_t FIELD_SIZE = CELL_SIZE * CELL_SIZE;
using Field = std::array<std::array<size_t, FIELD_SIZE>, FIELD_SIZE>;
using Coord = std::pair<size_t, size_t>;

bool fillField(Field &field)
{
  for (auto &row : field)
    for (auto &num : row)
    {
      std::cin >> num;
      if (std::cin.bad() || num > FIELD_SIZE)
        return false;
    }
  return true;
}

void printFiller()
{
  for (size_t i = 0, end = FIELD_SIZE * (MAX_NUM_W + 1) + 2 + CELL_SIZE;
       i != end; ++i)
    std::cout << '-';
  std::cout << std::endl;
}

void printField(const Field &field)
{
  printFiller();
  auto row_cnt = 0;
  for (const auto &row : field)
  {
    std::cout << "| ";
    auto col_cnt = 0;
    for (const auto &num : row)
      std::cout << std::setw(MAX_NUM_W) << num
                << ((++col_cnt % CELL_SIZE == 0) ? " |" : " ");
    std::cout << std::endl;

    if (++row_cnt % CELL_SIZE == 0)
      printFiller();
  }
}

bool findInCol(const Field &field, size_t col_idx, size_t num)
{
  assert(num > 0 && num <= FIELD_SIZE);
  assert(col_idx < FIELD_SIZE);

  for (size_t i = 0, rows = field.size(); i != rows; ++i)
    if (field[i][col_idx] == num)
      return true;

  return false;
}

bool findInRow(const Field &field, size_t row_idx, size_t num)
{
  assert(num > 0 && num <= FIELD_SIZE);
  assert(row_idx < FIELD_SIZE);

  auto &row = field[row_idx];

  auto found_it = std::find(row.begin(), row.end(), num);

  return found_it != row.end();
}

bool findInCell(const Field &field, const Coord &upper_left, size_t num)
{
  auto [row_id, col_id] = upper_left;
  assert(row_id % CELL_SIZE == 0 && col_id % CELL_SIZE == 0);

  for (size_t i = 0; i < CELL_SIZE; ++i)
  {
    auto &row = field[row_id + i];
    for (size_t j = 0; j < CELL_SIZE; ++j)
      if (row[col_id + j] == num)
        return true;
  }

  return false;
}

bool isSetOk(const Field &field, const Coord &coord, size_t num)
{
  assert(num > 0 && num <= FIELD_SIZE);

  auto [row_id, col_id] = coord;

  assert(col_id < FIELD_SIZE && row_id < FIELD_SIZE);

  Coord upper_left = {
    (row_id / CELL_SIZE) * CELL_SIZE,
    (col_id / CELL_SIZE) * CELL_SIZE,
  };
  assert(upper_left.first % CELL_SIZE == 0 &&
         upper_left.second % CELL_SIZE == 0);

  for (size_t i = 0; i < FIELD_SIZE; ++i)
  {
    // check current row
    if (field[row_id][i] == num)
      return false;

    // check current col
    if (field[i][col_id] == num)
      return false;

    // check current cell
    auto x = i % CELL_SIZE;
    auto y = i / CELL_SIZE;
    if (field[upper_left.first + x][upper_left.second + y] == num)
      return false;
  }
  return true;
}

std::optional<Coord> findFree(const Field &field)
{
  for (size_t i = 0, rows = field.size(); i != rows; ++i)
  {
    auto &row = field[i];
    for (size_t j = 0, cols = row.size(); j != cols; ++j)
      if (row[j] == 0)
        return Coord{i, j};
  }
  return std::nullopt;
}

bool solveSudoku(Field &field, int depth = 1)
{
  auto free = findFree(field);

  if (!free)
    return true;

  bool solved = false;

  auto coord = free.value();
#pragma omp taskloop final(depth > 2) shared(solved, field)
  for (size_t num = 1; num <= FIELD_SIZE; ++num)
  {
    if (isSetOk(field, coord, num))
    {
      auto copy = field;
      copy[coord.first][coord.second] = num;

      if (solveSudoku(copy, depth + 1))
      {
#pragma omp critical
        {
          field = copy;
          solved = true;
        }
#pragma omp cancel taskgroup
      }
    }
  }

#pragma omp taskwait
  return solved;
}

bool isValid(const Field &field)
{
  auto cpy = field;
  for (size_t i = 0; i < cpy.size(); ++i)
    for (size_t j = 0; j < cpy[i].size(); ++j)
    {
      auto &relem = cpy[i][j];
      auto elem = relem;

      relem = 0;
      if (!isSetOk(cpy, {i, j}, elem))
      {
        std::cerr << "Error in " << i << " " << j << '=' << elem << std::endl;
        return false;
      }
      relem = elem;
    }

  return true;
}

int main(int argc, char *argv[])
{
  if (argc == 2)
    omp_set_num_threads(std::atoi(argv[1]));

  Field field{};
  if (!fillField(field))
  {
    std::cerr << "Input error\n";
    return 1;
  }
  std::cout << "Input done\n";

  bool res = false;
  double dt = 0;
#pragma omp parallel
  {
#pragma omp single nowait
    {
      dt = omp_get_wtime();
      res = solveSudoku(field);
      dt = omp_get_wtime() - dt;
    }
  }
  if (res)
  {
    printField(field);
    assert(isValid(field));
  }
  else
    std::cout << "The solution does not exist.\n";

  std::cout << "Executed in " << dt << " sec\n";
}
