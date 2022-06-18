#include <iostream>
#include <vector>
#include <complex>
#include "Matrix.hpp"

using namespace std;

int main()
{
    cout << "Test 1: different types of matrix.\n";
    vector<vector<int>> vector1 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 1, 1, 2}, {3, 4, 5, 6}};
    Matrix matrix1(vector1);
    cout << "Integer matrix matrix1:\n"
         << matrix1 << endl;
    vector<vector<double>> vector2 = {{1.1, 2.2, 3, 4}, {5, 6, 7, 8}, {9, 1, 1, 2}, {3, 4, 5, 6}};
    Matrix matrix2(vector2);
    cout << "double matrix matrix2:\n"
         << matrix2 << endl;
    vector<vector<double>> vector3 = {{1.1, 2.2, 3, 4}, {5, 6, 7, 8}, {9, 0, 1, 2}, {3, 4, 5, 6}};
    Matrix matrix3(vector3);
    cout << "double matrix matrix3:\n"
         << matrix3 << endl;
    vector<vector<complex<double>>> vector4 = {{0, complex<double>(1, 2), 3, 4}, {5, 6, complex<double>(7, 8), 9}, {complex<double>(0, 2), complex<double>(3, 6), 0, 4}};
    Matrix matrix4(vector4);
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
    cout << "matrix2==matrix2:" << (matrix2 == matrix2) << endl;
    cout << "matrix2==matrix3:" << (matrix2 == matrix3) << endl;

    cout << "\nTest 3: transpotion, conjugation and multiplication\n";
    cout << "transpotion of matrix1:\n"
         << matrix1.trans() << endl;
    cout << "conjugation of matrix4:\n"
         << matrix4.conj() << endl;

    cout << "element-wise multiplication of matrix2 and matrix3:\n"
         << matrix2.eleWiseMul(matrix3) << endl;
    cout << "cross product of matrix2 and matrix3:\n"
         << matrix2.eleWiseMul(matrix3) << endl;
    cout << "dot product of matrix2 and matrix3:\n"
         << matrix2.dotPro(matrix3) << endl;
}
