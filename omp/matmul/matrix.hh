#ifndef __OMP_MATMUL_MATRIX_HH__
#define __OMP_MATMUL_MATRIX_HH__

#include <cassert>
#include <cmath>
#include <concepts>
#include <iostream>

namespace linal
{
using ldbl = long double;

constexpr ldbl MAT_THRESHOLD = 1e-10;

template <typename T>
class Matrix final
{
private:
  T **matr_;
  size_t rows_, cols_;
  static ldbl threshold;
  // emplace function type
  // using empl_func = T (*)( int, int );

public:
  Matrix(size_t rows = 0, size_t cols = 0);

  template <typename It>
  Matrix(size_t rows, size_t cols, It begin, It end);

  Matrix(size_t rows, size_t cols, const std::initializer_list<T> &ilist);

  template <typename empl_func>
  Matrix(size_t rows, size_t cols, empl_func fnc);

  // copy constructors
  Matrix(const Matrix &matr);
  // move
  Matrix(Matrix &&matr);

  // assignments operators
  Matrix &operator=(const Matrix &matr);
  // move
  Matrix &operator=(Matrix &&matr);

  Matrix &operator+=(const Matrix &matr)
  {
    for (std::size_t i = 0; i < rows_; ++i)
      for (std::size_t j = 0; j < cols_; ++j)
        matr_[i][j] += matr.matr_[i][j];

    return *this;
  }
  Matrix &operator-=(const Matrix &matr)
  {
    for (std::size_t i = 0; i < rows_; ++i)
      for (std::size_t j = 0; j < cols_; ++j)
        matr_[i][j] -= matr.matr_[i][j];

    return *this;
  }

  Matrix &Transpose();

  Matrix Transposing() const;

  size_t getCols() const
  {
    return cols_;
  }
  size_t getRows() const
  {
    return rows_;
  }

  static Matrix Identity(size_t rows);

  const T &At(size_t i, size_t j) const;

  const T *operator[](size_t i) const
  {
    return matr_[i];
  }

  T *operator[](size_t i)
  {
    return matr_[i];
  }

  bool empty() const
  {
    return cols_ == 0 || rows_ == 0;
  }

  void splitByFour(Matrix &a11, Matrix &a12, Matrix &a21, Matrix &a22) const
  {
    if (cols_ != rows_ || cols_ % 2 != 0)
      return;

    auto half_size = cols_ / 2;
    a11 = Matrix(half_size, half_size,
                 [this](auto i, auto j) { return matr_[i][j]; });
    a12 = Matrix(half_size, half_size, [this, half_size](auto i, auto j) {
      return matr_[i][j + half_size];
    });
    a21 = Matrix(half_size, half_size, [this, half_size](auto i, auto j) {
      return matr_[i + half_size][j];
    });
    a22 = Matrix(half_size, half_size, [this, half_size](auto i, auto j) {
      return matr_[i + half_size][j + half_size];
    });
  }

  ~Matrix()
  {
    for (size_t i = 0; i < rows_; ++i)
      delete[] matr_[i];

    delete[] matr_;
    matr_ = nullptr;
    rows_ = cols_ = 0;
  }

  bool IsEq(const Matrix &matr) const;

  void Dump(std::ostream &ost) const;

  static void SetThreshold(ldbl new_thres)
  {
    threshold = new_thres;
  }
  static void SetDefThres()
  {
    threshold = MAT_THRESHOLD;
  }
  static ldbl GetThreshold()
  {
    return threshold;
  }

  template <std::floating_point fpT>
  static bool IsZero(fpT val)
  {
    return std::abs(val) < threshold;
  }

  template <std::integral iT>
  static bool IsZero(iT val)
  {
    return val == iT{0};
  }

  template <typename walk_func>
  void Walker(walk_func walk);

private:
  Matrix &Transpose_Quad();

  void Alloc();

  template <typename It>
  void FillByIt(It begin, It end);

  static void Swap(Matrix &lhs, Matrix &rhs);

  /* copy matrix with identical sizes function */
  static void Copy(Matrix &dst, const Matrix &src);
};

template <typename T>
bool operator==(const Matrix<T> &lhs, const Matrix<T> &rhs);

template <typename T>
std::ostream &operator<<(std::ostream &ost, const Matrix<T> &matr);

template <typename T>
std::istream &operator>>(std::istream &ist, Matrix<T> &matr);

template <typename T>
std::istream &InputQuadr(std::istream &ist, Matrix<T> &matr);

template <typename InputIt, typename T>
void MatToIt(InputIt beg, InputIt end, const Matrix<T> &mat);
} // namespace linal

namespace Mul
{
using type = float;
using Mat = linal::Matrix<type>;
} // namespace Mul

template <typename T>
linal::ldbl linal::Matrix<T>::threshold = linal::MAT_THRESHOLD;

template <typename T>
linal::Matrix<T>::Matrix(size_t rows, size_t cols)
  : matr_(nullptr), rows_(rows), cols_(cols)
{
  Alloc();
  Walker([](int, int) { return T{}; });
}

template <typename T>
template <typename It>
linal::Matrix<T>::Matrix(size_t rows, size_t cols, It begin, It end)
  : matr_(nullptr), rows_(rows), cols_(cols)
{
  Alloc();
  FillByIt(begin, end);
}

template <typename T>
linal::Matrix<T>::Matrix(size_t rows, size_t cols,
                         const std::initializer_list<T> &ilist)
  : matr_(nullptr), rows_(rows), cols_(cols)
{
  Alloc();
  FillByIt(ilist.begin(), ilist.end());
}

template <typename T>
template <typename empl_func>
linal::Matrix<T>::Matrix(size_t rows, size_t cols, empl_func fnc)
  : matr_(nullptr), rows_(rows), cols_(cols)
{
  Alloc();
  Walker(fnc);
}

template <typename T>
template <typename walk_func>
void linal::Matrix<T>::Walker(walk_func walk)
{
  for (size_t i = 0; i < rows_; ++i)
    for (size_t j = 0; j < cols_; ++j)
      matr_[i][j] = walk(i, j);
}

template <typename T>
linal::Matrix<T>::Matrix(const linal::Matrix<T> &matr)
  : matr_(nullptr), rows_(matr.rows_), cols_(matr.cols_)
{
  Alloc();
  Copy(*this, matr);
}

template <typename T>
linal::Matrix<T>::Matrix(linal::Matrix<T> &&matr)
  : matr_(matr.matr_), rows_(matr.rows_), cols_(matr.cols_)
{
  matr.matr_ = nullptr;
  matr.rows_ = matr.cols_ = 0;
}

template <typename T>
linal::Matrix<T> &linal::Matrix<T>::operator=(const linal::Matrix<T> &matr)
{
  if (this == &matr)
    return *this;

  if (rows_ == matr.rows_ && cols_ == matr.cols_)
    Copy(*this, matr);
  else
  {
    Matrix tmp(matr);
    Swap(*this, tmp);
  }

  return *this;
}

template <typename T>
linal::Matrix<T> &linal::Matrix<T>::operator=(linal::Matrix<T> &&matr)
{
  if (this == &matr)
    return *this;

  Matrix tmp(std::move(matr));
  Swap(*this, tmp);

  return *this;
}

template <typename T>
linal::Matrix<T> &linal::Matrix<T>::Transpose()
{
  if (rows_ == cols_)
    return Transpose_Quad();

  Matrix<T> temp{cols_, rows_, [&](int i, int j) { return matr_[j][i]; }};

  Swap(*this, temp);

  return *this;
}

template <typename T>
linal::Matrix<T> linal::Matrix<T>::Transposing() const
{
  return Matrix<T>(cols_, rows_, [&](int i, int j) { return matr_[j][i]; });
}

template <typename T>
linal::Matrix<T> linal::Matrix<T>::Identity(size_t rows)
{
  Matrix id(rows, rows, [](int i, int j) { return i == j; });

  return id;
}
template <typename T>
const T &linal::Matrix<T>::At(size_t i, size_t j) const
{
  if (i >= rows_)
    throw std::out_of_range{"Row index is too big"};
  if (j >= cols_)
    throw std::out_of_range{"Col index is too big"};

  return matr_[i][j];
}

template <typename T>
bool linal::Matrix<T>::IsEq(const Matrix &matr) const
{
  if (rows_ != matr.rows_ || cols_ != matr.cols_)
    return false;

  for (size_t i = 0; i < rows_; ++i)
    for (size_t j = 0; j < cols_; ++j)
      if (!IsZero(matr_[i][j] - matr.matr_[i][j]))
        return false;
  return true;
}

template <typename T>
void linal::Matrix<T>::Dump(std::ostream &ost) const
{
  for (size_t i = 0; i < rows_; ++i)
  {
    ost << "|| ";
    for (size_t j = 0; j < cols_; ++j)
      ost << matr_[i][j] << (j == cols_ - 1 ? "" : ", ");
    ost << " ||\n";
  }
}

template <typename T>
linal::Matrix<T> &linal::Matrix<T>::Transpose_Quad()
{
  for (size_t i = 0; i < cols_; ++i)
    for (size_t j = i + 1; j < cols_; ++j)
      std::swap(matr_[i][j], matr_[j][i]);

  return *this;
}

template <typename T>
void linal::Matrix<T>::Alloc()
{
  if (cols_ == 0 || rows_ == 0)
    return;

  matr_ = new T *[rows_];

  for (size_t i = 0; i < rows_; ++i)
    matr_[i] = new T[cols_];
}

template <typename T>
template <typename It>
void linal::Matrix<T>::FillByIt(It begin, It end)
{
  size_t i = 0, size = rows_ * cols_;

  for (It it = begin; it != end && i < size; ++it, ++i)
    matr_[i / cols_][i % cols_] = *it;
}

template <typename T>
void linal::Matrix<T>::Swap(linal::Matrix<T> &lhs, linal::Matrix<T> &rhs)
{
  std::swap(lhs.matr_, rhs.matr_);
  std::swap(lhs.cols_, rhs.cols_);
  std::swap(lhs.rows_, rhs.rows_);
}

/* copy matrix with identical sizes function */
template <typename T>
void linal::Matrix<T>::Copy(linal::Matrix<T> &dst, const linal::Matrix<T> &src)
{
  if (dst.rows_ != src.rows_ || dst.cols_ != src.cols_)
    throw std::invalid_argument{"Matrixies have differrent sizes"};

  for (size_t i = 0; i < dst.rows_; ++i)
    for (size_t j = 0; j < dst.cols_; ++j)
      dst.matr_[i][j] = src.matr_[i][j];
}

template <typename T>
bool linal::operator==(const Matrix<T> &lhs, const Matrix<T> &rhs)
{
  return lhs.IsEq(rhs);
}

template <typename T>
std::ostream &linal::operator<<(std::ostream &ost, const linal::Matrix<T> &matr)
{
  matr.Dump(ost);

  return ost;
}

template <typename T>
std::istream &linal::operator>>(std::istream &ist, Matrix<T> &matr)
{
  size_t rows = 0, cols = 0;
  ist >> rows >> cols;

  matr = Matrix<T>{rows, cols, [&](int, int) {
                     T val{};
                     ist >> val;
                     return val;
                   }};

  return ist;
}

template <typename T>
std::istream &linal::InputQuadr(std::istream &ist, Matrix<T> &matr)
{
  size_t size = 0;
  ist >> size;

  matr = Matrix<T>{size, size, [&](int, int) {
                     T val{};
                     ist >> val;
                     return val;
                   }};

  return ist;
}

template <typename T>
linal::Matrix<T> operator+(const linal::Matrix<T> &lhs,
                           const linal::Matrix<T> &rhs)
{
  auto res = lhs;
  res += rhs;
  return res;
}

template <typename T>
linal::Matrix<T> operator-(const linal::Matrix<T> &lhs,
                           const linal::Matrix<T> &rhs)
{
  auto res = lhs;
  res -= rhs;
  return res;
}

template <typename InputIt, typename T>
void linal::MatToIt(InputIt beg, InputIt end, const Matrix<T> &mat)
{
  size_t i = 0, size = mat.getCols() * mat.getRows(), cols = mat.getCols();

  for (InputIt iit = beg; iit != end && i < size; ++i, ++iit)
    *iit = mat[i / cols][i % cols];
}

#endif /* __OMP_MATMUL_MATRIX_HH__ */
