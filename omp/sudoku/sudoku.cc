#include <iostream>
#include <iomanip>
#include <sstream>
#include <array>
#include <cassert>
#include <optional>
#include <algorithm>
#include <omp.h>

#if !defined(_OPENMP)
#error "This program requires OpenMP library"
#endif

constexpr size_t CELL_SIZE = 3;
constexpr size_t MAX_NUM_W = 2;
constexpr size_t FIELD_SIZE = CELL_SIZE * CELL_SIZE;
using Field = std::array<std::array<size_t, FIELD_SIZE>, FIELD_SIZE>;
using Coord = std::pair<size_t, size_t>;

void fillField(Field &field)
{
  for (auto &row : field)
    for (auto &num : row)
      std::cin >> num;
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

    if (++row_cnt % CELL_SIZE == 0) printFiller();
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

  Coord upper_left = {
    (coord.first / CELL_SIZE) * CELL_SIZE,
    (coord.second / CELL_SIZE) * CELL_SIZE,
  };

  return !findInRow(field, coord.first, num) &&
         !findInCol(field, coord.second, num) &&
         !findInCell(field, upper_left, num);
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

  auto coord = free.value();
  for (size_t num = 1; num <= FIELD_SIZE; ++num)
  {
    if (!isSetOk(field, coord, num))
      continue;
    #pragma omp task firstprivate()
    {
      auto copy = field;
      copy[coord.first][coord.second] = num;

      if (solveSudoku(copy, depth + 1))
      {
        assert(field != copy);
        field = copy;
        return true;
      }
    }
  }

  #pragma omp taskwait
  return false;
}

int main()
{
  Field field{};
  fillField(field);


  bool res = false;
  #pragma omp parallel sections
  #pragma omp single nowait
  {
    res = solveSudoku(field);
  }
  if (res)
    printField(field);
  else
    std::cout << "The solution does not exist.\n";
}
