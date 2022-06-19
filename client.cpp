#include <cstdio>
#include <windows.h>
#include <iostream>
#include "matrix.hpp"

using namespace std;

int main() {
    cout << "Welcome to the matrix computer world!\nLoading: ";
    for (int i = 0; i <= 10; ++i) {
        if (i == 10) cout << "100% Here we go!" << endl;
        else cout << i * 10 << "%, ";
        Sleep(300);
    }
    cout << "\nIn this calculator, you are free to do following things:\n"
         << "(0)Print a stored matrix;\n"
         << "(1)Type in and store a matrix;\n"
         << "(2)Add,           (3)Minus,        (4)Multiply.                in Matrix-Matrix;\n"
         << "(5)Add,           (6)Minus,        (7)Multiply,    (8)divide.  in scalar;\n"
         << "(9)Transposition, (10)Conjugation;\n"
         << "(11)Max value,    (12)Min value,   (13)Sum,        (14)Average;\n"
         << "(15)Eigenvalue,   (16)Eigenvector, (17)Trace,      (18)Inverse,    (19)determinant;\n"
         << "(20)Reshape,      (21)Slicing,     (22)Convolution.\n" << endl;


    vector<Matrix<double>> vec;
    int choose = 1;
    while (true) {
        switch (choose) {
            case 0: {
                if (vec.empty()) cout << "There has no stored matrix now." << endl;
                else {
                    int i = 0;
                    cout << "There are " << vec.size() - 1 << " matrix now (0-index), choose one:";
                    cin >> i;
                    if (i < vec.size()) cout << vec[i] << endl;
                    else cout << "Out of the bound!" << endl;
                }
                break;
            }
            case 1: {
                int r, c;
                cout << "Type in the integer row and col:" << endl;
                cin >> r >> c;
                cout << "Type in the element:" << endl;
                double t;
                vector<vector<double>> t1;
                vector<double> t2;
                for (int i = 0; i < r; ++i) {
                    for (int j = 0; j < c; ++j) {
                        cin >> t;
                        t2.push_back(t);
                    }
                    t1.push_back(t2);
                    t2.resize(0);
                }
                Matrix<double> m(t1);
                vec.push_back(m);
                cout << "Your new matrix is ID-" << vec.size() << " (0-index)." << endl;
                break;
            }
            case 2: {
                if (vec.size() < 2) cout << "Please first store at least 2 matrix!" << endl;
                else {
                    cout << "Choose two matrix to add:" << endl;
                    int a, b;
                    cin >> a >> b;
                    if (min(a, b) >= 0 && max(a, b) < vec.size())
                        cout << vec[a] + vec[b] << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 3: {
                if (vec.size() < 2) cout << "Please first store at least 2 matrix!" << endl;
                else {
                    cout << "Choose two matrix to minus:" << endl;
                    int a, b;
                    cin >> a >> b;
                    if (min(a, b) >= 0 && max(a, b) < vec.size())
                        cout << vec[a] - vec[b] << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 4: {
                if (vec.size() < 2) cout << "Please first store at least 2 matrix!" << endl;
                else {
                    cout << "Choose two matrix to multiply:" << endl;
                    int a, b;
                    cin >> a >> b;
                    if (min(a, b) >= 0 && max(a, b) < vec.size())
                        cout << vec[a] * vec[b] << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 5: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to add a scalar, and the scalar:" << endl;
                    int a, b;
                    cin >> a >> b;
                    if (a >= 0 && a < vec.size()) {
                        vec[a] += b;
                        cout << vec[a] << endl;
                    } else cout << "Out of the bound!";
                }
                break;
            }
            case 6: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to minus a scalar, and the scalar:" << endl;
                    int a, b;
                    cin >> a >> b;
                    if (a >= 0 && a < vec.size()) {
                        vec[a] -= b;
                        cout << vec[a] << endl;
                    } else cout << "Out of the bound!";
                }
                break;
            }
            case 7: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to multiply a scalar, and the scalar:" << endl;
                    int a, b;
                    cin >> a >> b;
                    if (a >= 0 && a < vec.size()) {
                        vec[a] *= b;
                        cout << vec[a] << endl;
                    } else cout << "Out of the bound!";
                }
                break;
            }
            case 8: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to divide a scalar, and the scalar:" << endl;
                    int a, b;
                    cin >> a >> b;
                    if (a >= 0 && a < vec.size()) {
                        vec[a] /= b;
                        cout << vec[a] << endl;
                    } else cout << "Out of the bound!";
                }
                break;
            }
            case 9: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get transposition:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].trans() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 10: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get Conjugation:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
//                        cout << vec[a].conj() << endl; // double itself is conj
                        cout << vec[a] << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 11: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get max value:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].max() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 12: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get min value:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].min() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 13: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get sum:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].sum() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 14: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get average value:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].mean() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 15: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get eigen-value:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].eigenValue() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 16: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get eigen-vector:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].eigenVector() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 17: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get trace:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].trace() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 18: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get inverse:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].invert() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 19: {
                if (vec.empty()) cout << "Please first store at least 1 matrix!" << endl;
                else {
                    cout << "Choose the matrix to get determinant:" << endl;
                    int a;
                    cin >> a;
                    if (a >= 0 && a < vec.size())
                        cout << vec[a].determinant() << endl;
                    else cout << "Out of the bound!";
                }
                break;
            }
            case 20:
            case 21:
            case 22:
            default:
                break;
        }


        cout << "Please type in num(0~22) to choose operation next, or others to quit:";
        cin >> choose;
        if (choose <= 0 or choose > 23) {
            cout << "Calculator detect you type in the wrong num. THE END! Bye" << endl;
            break;
        }
    }
}
