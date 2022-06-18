#include <iostream>
#include <vector>
#include <complex>
#include "Matrix.hpp"

using namespace std;

int main()
{
    vector<vector<int> > vec1 = {{1, 1}, {1, 1}}; // 2×2
    Matrix<int> mat1(vec1);
    vector<vector<int> > vec2 = {{1, 1, 1}}; // 1×3
    Matrix<int> mat2(vec2);
    cout << "This the exception test of matrix.hpp." << endl;
    cout << "1.For matrix adding and subtracting, if the size are not the same:" << endl;
    cout << "mat1:\n"
         << mat1 << endl;
    cout << "mat2:\n"
         << mat2 << endl;
    cout << "mat1 + mat2 =" << mat1 + mat2 << endl;
    cout << "mat1 * mat2 =" << mat1 * mat2 << endl;
    cout << "mat1.eleWiseMul(mat2) : " << mat1.eleWiseMul(mat2) << endl;
    cout << "mat1.crossPro(mat2) : " << mat1.crossPro(mat2) << endl;
    cout << "mat1.dotPro(mat2) : " << mat1.dotPro(mat2) << endl;

    cout << "2.Invalid parameter:" << endl;
    cout << "mat1 / 0 =" << mat1 / 0 << endl;
    cout << "Reshape: " << mat1.reshape(1, 1);

    cout << "3.Out of bound exception:" << endl;
    cout << "mat1.max(10): " << mat1.max(10);
    cout << "mat1.min(10): " << mat1.min(10) << endl;
    cout << "mat1.sum(10): " << mat1.sum(10) << endl;
    cout << "mat1.mean(10): " << mat1.mean(10) << endl;
    cout << "Slice: " << mat1.slice(10, 10, 10, 10) << endl;

    cout << "4.The matrix have no inverse/determinent..." << endl;
    // cout << "Eigenvalue: " << mat2.eigenValue() << endl;
    // cout << "Eigenvector: " << mat2.eigenVector() << endl;
    cout << "Inverse: " << mat2.invert() << endl;
    cout << "Trace: " << mat2.trace() << endl;
    cout << "Determinent: " << mat2.determinant() << endl;
    return 0;
}
