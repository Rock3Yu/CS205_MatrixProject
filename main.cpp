#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include "Matrix.hpp"

using namespace std;

int main()
{
    cout << "Test 1: different types of matrix.\n";
    vector<vector<int> > vector1 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 1, 1, 2}, {3, 4, 5, 6}};
    Matrix<int> matrix1(vector1);
    cout << "Integer matrix matrix1:\n"
         << matrix1 << endl;
    vector<vector<double> > vector2 = {{1.1, 2.2, 3, 4}, {5, 6, 7, 8}, {9, 1, 1, 2}, {3, 4, 5, 6}};
    Matrix<double> matrix2(vector2);
    cout << "double matrix matrix2:\n"
         << matrix2 << endl;
    vector<vector<double> > vector3 = {{1.1, 2.2, 3, 4}, {5, 6, 7, 8}, {9, 0, 1, 2}, {3, 4, 5, 6}};
    Matrix<double> matrix3(vector3);
    cout << "double matrix matrix3:\n"
         << matrix3 << endl;
    vector<vector<complex<double> > > vector4 = {{0, complex<double>(1, 2), 3, 4}, {5, 6, complex<double>(7, 8), 9}, {complex<double>(0, 2), complex<double>(3, 6), 0, 4}};
    Matrix<complex<double> > matrix4(vector4);
    cout << "complex matrix matrix4:\n"
         << matrix4 << endl;

    cout << "\nTest 2: basic calculation of matrix(+ - * /)\n";
    cout << "matrix2 + matrix4:\n"
         << matrix2 + matrix3 << endl;
    cout << "matrix2 + 2:\n"
         << matrix2 + 2 << endl;
    cout << "matrix2 - matrix4:\n"
         << matrix2 - matrix3 << endl;
    cout << "matrix2 - 2:\n"
         << matrix2 - 2 << endl;
    cout << "matrix2 * 2:\n"
         << matrix2 * 2 << endl;
    cout << "matrix2 / 2:\n"
         << matrix2 / 2 << endl;
    cout << "matrix2 == matrix2:" << (matrix2 == matrix2) << endl;
    cout << "matrix2 == matrix3:" << (matrix2 == matrix3) << endl;

    cout << "\nTest 3: transpotion, conjugation and multiplication\n";
    cout << "transpotion of matrix1:\n"
         << matrix1.trans() << endl;
    cout << "conjugation of matrix4:\n"
         << matrix4.conj() << endl;

    cout << "element-wise multiplication of matrix2 and matrix3:\n"
         << matrix2.eleWiseMul(matrix3) << endl;
    
    Matrix<double> A({{-1, 2, 3}});
    Matrix<double> B({{1, -10, 6}});
    cout << "Matrix A and B are 1-dim matrices (or called vectors):" << endl
         << "A:" << A << "\nB:" << B << endl;
    cout << "cross product of matrixA and matrixB:\n"
         << A.crossPro(B) << endl;
    cout << "dot product of matrixA and matrixB:\n"
         << A.dotPro(B) << endl;

    cout << "\nTest 3: max, min, sum, mean\n";
    cout << "Max matrix1 = " << matrix1.max() << ", max row 0 of matrix1 = " << matrix1.max(0) << ", max column 0 of matrix1 = " << matrix1.max(0, true) << endl;
    cout << "Min matrix1 = " << matrix1.min() << ", min row 0 of matrix1 = " << matrix1.min(0) << ", min column 0 of matrix1 = " << matrix1.min(0, true) << endl;
    cout << "The sum of matrix1 = " << matrix1.sum() << ", sum row 0 of matrix1 = " << matrix1.sum(0) << ", sum column 0 of matrix1 = " << matrix1.sum(0, true) << endl;
    cout << "The mean of matrix1 = " << matrix1.mean() << ", mean row 0 of matrix1 = " << matrix1.mean(0) << ", mean column 0 of matrix1 = " << matrix1.mean(0, true) << endl;

    cout << "\n Test 4: eigenvalue, eigenvector, trace, inverse, determinent\n";
    vector<vector<double> > vector5 = {{1, 2, 2}, {2, 1, 2}, {2, 2, 1}};
    Matrix<double> matrix5(vector5);
    // cout << "The eigrnvalue of " << matrix5 << "is: " << matrix5.eigenValue() << endl;
    // cout << "The eigrnvalue of is:\n"
    //      << matrix5.eigenVector() << endl;
    cout << "The trace is: " << matrix5.trace() << endl;
    cout << "The inverse is: " << matrix5.invert() << endl;
    cout << "The determinent is: " << matrix5.determinant() << endl;

    cout << "\nTest 5: reshape, slice, convolution";
    cout << "Reshape matrix1:\n"
         << matrix1 << "from " << matrix1.getRows() << "×" << matrix1.getCols() << " to 2×8:\n"
         << matrix1.reshape(2, 8) << endl;
    cout << "Reshape a 3-dim matrix full of 1 with 2×2×2 to 2-dim 4×2:\n";
    vector<vector<vector<int> > > three_dim = {{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}};
    vector<vector<int> > vector6 = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    Matrix<int> two_dim(vector6);
    two_dim.reshape(three_dim, 2, 2, 2);
    cout << two_dim << endl;
    cout << "Slice matrix1 drom (1,0) to (3,2): \n"
         << matrix1.slice(1, 0, 3, 2) << endl;
    cout << "The convolution output of matrix2 and matrix5:\nmatrix1:" << matrix2 << endl;
    cout << "matrix5:" << matrix5 << endl;
    cout << "matrix2 * matrix5:" << matrix2.convolution(matrix5) << endl;

    cout<<"\nTest 6: sparse matrix\n";
    SpareMatrix<double> sm1(5,5);
    cout << "The default (5, 5) sparse matrix: \n" << sm1;
    cout << "\nInsert 3 elements in it:" << endl;
    sm1.insert(1,1,20.0);
    sm1.insert(0,0,19.98);
    element<double> in(3,4,66.66);
    sm1.insert(in);
    cout << sm1 << endl;

}
