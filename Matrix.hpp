#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <map>
//#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

template<class T>
class Matrix {
private:
    vector<vector<T> > matrix;
    int rows, cols;

public:
    // Constructors
    Matrix() : rows(0), cols(0) { matrix.resize(0); }

    Matrix(int r, int c) {
        rows = r;
        cols = c;
        matrix.resize(r);
        for (int i = 0; i < r; ++i) matrix[i].resize(c);
    }
    
        // vector '=' assignment is deep-copy!
    /*
     * vector<vector<double>> vec1 = {{1, 2, 3}, {4, 5, 6}};
     * vector<vector<double>> vec2 = vec1;
     * vec1[0][0] = 100;
     * cout << vec2[0][0]; // output: 1
     */
    explicit Matrix(vector<vector<T> > matrix) {
        this->matrix = matrix;
        rows = matrix.size();
        cols = matrix[0].size();
    }

    // copy constructor
    Matrix(const Matrix<T> &m1) {
        matrix = m1.matrix;
        rows = m1.rows;
        cols = m1.cols;
    }
    
    void operator=(const Matrix<T> &m) {
        this->matrix = m.matrix;
        this->rows = m.rows;
        this->cols = m.cols;
    }
    
    virtual ~Matrix() = default;

    friend ostream &operator<<(ostream &os, const Matrix &m) {
        os << "[";
        for (int i = 0; i < m.rows; ++i) {
            if (i != 0) os << " ";
            os << "[" << m.matrix[i][0];
            for (int j = 1; j < m.cols; ++j) os << " " << m.matrix[i][j];
            if (i != m.rows - 1)os << "]\n";
            else os << "]]\n";
        }
        return os;
    }
	
    const vector<vector<T> > &getMatrix() const { return matrix; }

    int getRows() const { return rows; }

    int getCols() const { return cols; }
    
    
    //note: Put Lin Peijun's codes here:

    //addition
    Matrix<T> operator+(const Matrix<T> &m) {
        if (m.cols == cols && m.rows == rows) {
            Matrix<T> M(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    M.matrix[i][j] = matrix[i][j] + m.matrix[i][j];
            return M;
        } else {
            cerr << "The sizes do not match when using '+'" << endl;
            return Matrix<T>(0, 0);
        }
    }

    template<typename T2>
    Matrix<T> operator+(const T2 scalar) {
        Matrix<T> M(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                M.matrix[i][j] = matrix[i][j] + scalar;
        return M;
    }

    //subtraction
    Matrix<T> operator-(const Matrix<T> &m) {
        if (m.cols == cols && m.rows == rows) {
            Matrix<T> M(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    M.matrix[i][j] = matrix[i][j] - m.matrix[i][j];
            return M;
        } else {
            cerr << "The sizes do not match when using '-'" << endl;
            return Matrix<T>(0, 0);
        }
    }

    template<typename T2>
    Matrix<T> operator-(const T2 scalar) {
        Matrix<T> M(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                M.matrix[i][j] = matrix[i][j] - scalar;
        return M;
    }

    // matrix-matrix multiplication & matrix-vector multiplication
    Matrix<T> operator*(const Matrix<T> &m) {
        if (cols == m.rows) {
            int loop = cols;
            Matrix M(rows, m.cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < m.cols; j++)
                    for (int k = 0; k < loop; ++k)
                        M.matrix[i][j] += matrix[i][loop] * m.matrix[loop][j];
            return M;
        } else {
            cerr << "The sizes do not match when using '*'" << endl;
            return Matrix<T>(0, 0);
        }
    }

    //scalar multiplication
    template<typename T2>
    Matrix<T> operator*(const T2 scalar) {
        Matrix<T> out = *this;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] *= scalar;
        return out;
    }

    //scalar division
    template<typename T2>
    Matrix<T> operator/(const T2 scalar) {
        if (scalar == 0) {
            cerr << "0 is denominator" << endl;
            return Matrix<T>(0, 0);
        }
        Matrix<T> out = *this;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out.matrix[i][j] = matrix[i][j] / scalar;
        return out;
    }

    void operator+=(const Matrix<T> &m) { *this = *this + m; }

    template<typename T2>
    void operator+=(const T2 scalar) { *this = *this + scalar; }

    void operator-=(const Matrix<T> &m) { *this = *this - m; }

    template<typename T2>
    void operator-=(const T2 scalar) { *this = *this - scalar; }

    void operator*=(const Matrix<T> &m) { *this = *this * m; }

    template<typename T2>
    void operator/=(T2 scalar) { *this = *this / scalar; }

    template<typename T2>
    void operator*=(T2 scalar) { *this = *this * scalar; }

    bool operator==(const Matrix<T> &m) {
        if (m.cols == cols && m.rows == rows) {
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    if (matrix[i][j] != m.matrix[i][j]) return false;
            return true;
        } else return false;
    }

    // transposition
    Matrix<T> trans() {
        Matrix<T> M(cols, rows);
        for (int i = 0; i < cols; i++)
            for (int j = 0; j < rows; j++)
                M.matrix[i][j] = matrix[j][i];
        return M;
    }

    // conjugation
    Matrix<T> conj() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = std::conj(matrix[i][j]);
        return *this;
    }

    // element-wise multiplication
    Matrix<T> eleWiseMul(const Matrix<T> &m) {
        if (m.cols == cols && m.rows == rows) {
            Matrix<T> M(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    M.matrix[i][j] = matrix[i][j] * m.matrix[i][j];
            return M;
        } else {
            cerr << "The sizes do not match when using 'eleWiseMul'" << endl;
            return Matrix<T>(0, 0);
        }
    }

    //dot product (only 1xn vector accept)
    T dotPro(const Matrix<T> &m) {
        if (cols == 1 && m.cols == 1 && rows == m.rows) {
            T out = 0;
            for (int i = 0; i < rows; ++i)
                out += matrix[0][i] * m.matrix[0][i];
            return out;
        } else {
            cerr << "The sizes do not match when using 'dotPro'" << endl;
            return 0;
        }
    }

    // cross product (only 3-dim vector (1x3) accept)
    // n = u(x1, y1, z1) x v(x2, y2, z2)
    // = (y1z2 - y2z1, x2z1-z2x1, x1y2 -x2y1)
    Matrix<T> crossPro(const Matrix<T> &m) {
        if (rows == m.rows && rows == 1 && cols == m.cols && cols == 3) {
            Matrix<T> out(1,3);
            out.matrix[0][0] = matrix[0][1] * m.matrix[0][2] - m.matrix[0][1] * matrix[0][2];
            out.matrix[0][1] = m.matrix[0][0] * matrix[0][2] - m.matrix[0][2] * matrix[0][0];
            out.matrix[0][2] = matrix[0][0] * m.matrix[0][1] - m.matrix[0][0] * matrix[0][1];
            return out;
        } else {
            cerr << "The sizes do not match when using 'crossPro'" << endl;
            return Matrix<T>(0, 0);
        }
    }
	
    //reference:https://blog.csdn.net/u013355826/article/details/64905921
    //only for images of 1 channel and 3 channels, changing Matrix to Mat
    //pt = the point of i row pixel
    Mat matrix2Mat(int type){
        if (type == 3){
            Mat M = Mat::zeros(this->rows,this->cols/3,CV_8UC3);
            for (int i = 0;i < rows;i++)        
            {
                uchar* pt = M.ptr<uchar>(i); 
                for (int j = 0; j < cols; j++)   
                {
                    pt[j] = matrix[i][j];
                }
            }
            return M;
        }else if(type == 1){
            Mat M = Mat::zeros(this->rows,this->cols,CV_8UC1);
            for (int i = 0;i < rows;i++)        
            {
                uchar* pt = M.ptr<uchar>(i); 
                for (int j = 0; j < cols; j++)   
                {
                    pt[j] = matrix[i][j];
                }
            }
            return M;
        }else{
            cerr << "please input Mat of 1 or 3 channels" << endl;
            return Mat::zeros(0, 0, CV_8UC1);
        }
    }


    //changing Mat to Matrix
    //pt = the point of i row pixel, row = rows of matrix, col = cols of matrix
    Matrix<T> mat2Matrix(Mat M){
        int row = M.rows;
        int col = M.cols * M.channels();
        Matrix<T> out(row, col);
        for (int i=0; i<row; i++)        
            {
                uchar* pt = M.ptr<uchar>(i); 
                for (int j=0; j<col; j++)   
                {
                    out.matrix[i][j] = pt[j];
                }
            }
        return out;
    }
	
	
	
    // note: Put YU Kunyi's codes here:
    
    // Element-wise minimum and maximum: min(A, B), min(A, alpha), max(A, B), max(A, alpha)
    // cv::max()	逐元素求两个矩阵之间的最大值
    // 这4个应该可以忽略
    static Matrix<T> max(const Matrix<T> &m1, const Matrix<T> &m2) {
        try {
            if (m1.rows != m2.rows || m1.cols != m2.cols) throw 1;
            Matrix<T> out(m1.rows, m1.cols);
            for (int i = 0; i < m1.rows; ++i)
                for (int j = 0; j < m1.co; ++j)
                    out.matrix[i][j] = max(m1.matrix[i][j], m2.matrix[i][j]);
            return out;
        } catch (int num) {
            cerr << "error in max()" << endl;
        }
    }

    static Matrix<T> max(const Matrix<T> &m1, int alpha) {
        Matrix<T> out(m1.rows, m1.cols);
        for (int i = 0; i < m1.rows; ++i)
            for (int j = 0; j < m1.co; ++j)
                out.matrix[i][j] = max(m1.matrix[i][j], alpha);
        return out;
    }

    static Matrix<T> min(const Matrix<T> &m1, const Matrix<T> &m2) {
        try {
            if (m1.rows != m2.rows || m1.cols != m2.cols) throw 1;
            Matrix<T> out(m1.rows, m1.cols);
            for (int i = 0; i < m1.rows; ++i)
                for (int j = 0; j < m1.co; ++j)
                    out.matrix[i][j] = min(m1.matrix[i][j], m2.matrix[i][j]);
            return out;
        } catch (int num) {
            cerr << "error in max()" << endl;
        }
    }

    static Matrix<T> min(const Matrix<T> &m1, int alpha) {
        Matrix<T> out(m1.rows, m1.cols);
        for (int i = 0; i < m1.rows; ++i)
            for (int j = 0; j < m1.co; ++j)
                out.matrix[i][j] = min(m1.matrix[i][j], alpha);
        return out;
    }

    // 按照文档中函数要求理解的：
    /**
     * @brief the maximum in all matrix
     * @return max in type T
     * */
    T max() {
        T max = matrix[0][0];
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                if (max<matrix[i][j]) max = matrix[i][j];
        return max;
    }

    /**
    * @brief the minimum in all matrix
    * @return min in type T
    * */
    T min() {
        T min = matrix[0][0];
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                if (min > matrix[i][j]) min = matrix[i][j];
        return min;
    }

    /**
     * @param num the max you want to find in col(num) or row(num)
     * @param axis =0 means row, =1 means col
     * */
    T max(int num, bool axis = 0) {
        T max;
        if (!axis) {
            if (num >= rows) cerr << "error in function: T max(int num, bool axis);" << endl;
		return 0;
            // note: max in row(num)
            max = matrix[num][0];
            for (int i = 0; i < cols; ++i) if (max < matrix[num][i]) max = matrix[num][i];
        } else {
            if (num >= cols) cerr << "error in function: T max(int num, bool axis);" << endl;
		return 0;
            // note: max in col(num)
            max = matrix[0][num];
            for (int i = 0; i < rows; ++i) if (max < matrix[i][num]) max = matrix[i][num];
        }
        return max;
    }

    /**
    * @param num the min you want to find in col(num) or row(num)
    * @param axis =0 means row, =1 means col
    * */
    T min(int num, bool axis = 0) {
        T min;
        if (!axis) {
            if (num >= rows) cerr << "error in function: T min(int num, bool axis);" << endl;
		return 0;
            // note: max in row(num)
            min = matrix[num][0];
            for (int i = 0; i < cols; ++i) if (min > matrix[num][i]) min = matrix[num][i];
        } else {
            if (num >= cols) cerr << "error in function: T min(int num, bool axis);" << endl;
		return 0;
            // note: max in col(num)
            min = matrix[0][num];
            for (int i = 0; i < rows; ++i) if (min > matrix[i][num]) min = matrix[i][num];
        }
        return min;
    }
    
        /**
     * @brief sum of all matrix
     * @return sum
     * */
    T sum() {
        T sum = matrix[0][0] - matrix[0][0];
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                sum += matrix[i][j];
        return sum;
    }
    
    /**
     * @brief sum in axis
     * @param num number of col/row
     * @param axis =0 means row, =1 means col
     * @return sum of col(num) or row(num)
     * */
    T sum(int num, bool axis = 0) {
        T sum;
        if (!axis) {
            if (num >= rows) cerr << "error in function: T sum(int num, bool axis);" << endl;
		return 0;
            // note: sum in row(num)
            sum = matrix[num][0];
            for (int i = 1; i < cols; ++i) sum += matrix[num][i];
        } else {
            if (num >= cols) cerr << "error in function: T sum(int num, bool axis);" << endl;
		return 0;
            // note: sum in col(num)
            sum = matrix[0][num];
            for (int i = 1; i < rows; ++i) sum += matrix[i][num];
        }
        return sum;
    }
    
     /**
     * @brief average of all elements
     * */
    T mean() {
        return sum() / (rows * cols);
    }

    /**
     * @brief average in axis
     * @param num number of col/row
     * @param axis =0 means row, =1 means col
     * */
    T mean(int num, bool axis = 0) {
        return axis ? sum(num, axis) / rows : sum(num, axis) / cols;
    }
	
    /**
     * @brief 特征值和特征向量 幂法、雅克比法以及QR法
     * */
    Matrix<T> eigenValue() {
        vector<T> val;
        vector<vector<T>> vec;
        eigenUtility(val, vec);
        vector<vector<T>> temp;
        temp.push_back(val);
        Matrix<T> out(temp);
        return out;
    }

    Matrix<T> eigenVector() {
        vector<T> val;
        vector<vector<T>> vec;
        eigenUtility(val, vec);
        Matrix<T> out(vec);
        return out;
    }

    void eigenUtility(vector<T> &val, vector<vector<T>> &vec) {
        val.resize(rows);
        vec.resize(rows);
        for (int i = 0; i < rows; ++i) vec[i].resize(rows);
        vector<vector<T>> m = matrix;
        int n = rows;
        double eps = 1e-10;
        int loop = 10000;
        Jacobi(m, n, vec, val, eps, loop);
    }

    bool Jacobi(vector<vector<double>> matrix, int dim, vector<vector<double>> &eigenvectors,
                vector<double> &eigenvalues, double precision, int max) {
        for (int i = 0; i < dim; i++) {
            eigenvectors[i][i] = 1.0f;
            for (int j = 0; j < dim; j++) {
                if (i != j)
                    eigenvectors[i][j] = 0.0f;
            }
        }

        int nCount = 0;        //current iteration
        while (1) {
            //find the largest element on the off-diagonal line of the matrix
            double dbMax = matrix[0][1];
            int nRow = 0;
            int nCol = 1;
            for (int i = 0; i < dim; i++) {            //row
                for (int j = 0; j < dim; j++) {        //column
                    double d = fabs(matrix[i][j]);
                    if ((i != j) && (d > dbMax)) {
                        dbMax = d;
                        nRow = i;
                        nCol = j;
                    }
                }
            }

            if (dbMax < precision)     //precision check
                break;
            if (nCount > max)       //iterations check
                break;
            nCount++;

            double dbApp = matrix[nRow][nRow];
            double dbApq = matrix[nRow][nCol];
            double dbAqq = matrix[nCol][nCol];
            //compute rotate angle
            double dbAngle = 0.5 * atan2(-2 * dbApq, dbAqq - dbApp);
            double dbSinTheta = sin(dbAngle);
            double dbCosTheta = cos(dbAngle);
            double dbSin2Theta = sin(2 * dbAngle);
            double dbCos2Theta = cos(2 * dbAngle);
            matrix[nRow][nRow] = dbApp * dbCosTheta * dbCosTheta +
                                 dbAqq * dbSinTheta * dbSinTheta + 2 * dbApq * dbCosTheta * dbSinTheta;
            matrix[nCol][nCol] = dbApp * dbSinTheta * dbSinTheta +
                                 dbAqq * dbCosTheta * dbCosTheta - 2 * dbApq * dbCosTheta * dbSinTheta;
            matrix[nRow][nCol] = 0.5 * (dbAqq - dbApp) * dbSin2Theta + dbApq * dbCos2Theta;
            matrix[nCol][nRow] = matrix[nRow][nCol];

            for (int i = 0; i < dim; i++) {
                if ((i != nCol) && (i != nRow)) {
                    dbMax = matrix[i][nRow];
                    matrix[i][nRow] = matrix[i][nCol] * dbSinTheta + dbMax * dbCosTheta;
                    matrix[i][nCol] = matrix[i][nCol] * dbCosTheta - dbMax * dbSinTheta;
                }
            }

            for (int j = 0; j < dim; j++) {
                if ((j != nCol) && (j != nRow)) {
                    dbMax = matrix[nRow][j];
                    matrix[nRow][j] = matrix[nCol][j] * dbSinTheta + dbMax * dbCosTheta;
                    matrix[nCol][j] = matrix[nCol][j] * dbCosTheta - dbMax * dbSinTheta;
                }
            }

            //compute eigenvector
            for (int i = 0; i < dim; i++) {
                dbMax = eigenvectors[i][nRow];
                eigenvectors[i][nRow] = eigenvectors[i][nCol] * dbSinTheta + dbMax * dbCosTheta;
                eigenvectors[i][nCol] = eigenvectors[i][nCol] * dbCosTheta - dbMax * dbSinTheta;
            }
        }

        //sort eigenvalues
        std::map<double, int> mapEigen;
        for (int i = 0; i < dim; i++) {
            eigenvalues[i] = matrix[i][i];
            mapEigen.insert(make_pair(eigenvalues[i], i));
        }

        vector<vector<double>> pdbTmpVec;
        pdbTmpVec.resize(dim);
        for (int i = 0; i < dim; ++i) pdbTmpVec[i].resize(dim);
        std::map<double, int>::reverse_iterator iter = mapEigen.rbegin();
        for (int j = 0; iter != mapEigen.rend(), j < dim; ++iter, ++j) {
            for (int i = 0; i < dim; i++) {
                pdbTmpVec[i][j] = eigenvectors[i][iter->second];
            }
            eigenvalues[j] = iter->first;
        }

        for (int i = 0; i < dim; i++) {
            double dSumVec = 0;
            for (int j = 0; j < dim; j++)
                dSumVec += pdbTmpVec[j][i];
            if (dSumVec < 0) {
                for (int j = 0; j < dim; j++)
                    pdbTmpVec[j][i] *= -1;
            }
        }

        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                eigenvectors[i][j] = pdbTmpVec[i][j];

        return true;
    }

	
    /**
     * @brief trace of the Matrix
     * */
    T trace() {
        T trace = 0;
        if (cols != rows) cerr << "matrix with col != row have no trace!" << endl;
        else for (int i = 0; i < rows; ++i) trace += matrix[i][i];
        return trace;
    }

    /**
     * @brief det(A) or |A|
     * */
    T determinant() {
        if (rows != cols) {
            cerr << "matrix with col != row have no determinant!" << endl;
            return 0;
        }
        return detUtility(matrix, rows);
    }

    /**
     * @brief the actual func calculate the det(A)
     * */
    T detUtility(vector<vector<T>> v, int n) {
        if (n == 1) { return v[0][0]; }
        T sum = 0;
        for (int i = 0; i < n; ++i) {
            vector<vector<T>> s;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                vector<T> temp;
                for (int k = 1; k < n; ++k) temp.push_back(v[j][k]);
                s.push_back(temp);
            }
            sum += pow(-1, 0 + i) * v[i][0] * detUtility(s, n - 1);
        }
        return sum;
    }
	
    /**
     * @brief A^-1 or reverse of the Matrix
     * @attention use the Adjoint matrix method
     * */
    Matrix<T> invert() {
        if (rows != cols) {
            cerr << "Matrix with row != col has no invert!" << endl;
            return Matrix<T>(0, 0);
        }
        if (determinant() == 0) {
            cerr << "Matrix with det = 0 has no invert!" << endl;
            return Matrix<T>(0, 0);
        }
        vector<vector<T> > vec;
        for (int i = 0; i < rows; ++i) {
            vector<T> temp;
            for (int j = 0; j < cols; ++j)
                temp.push_back(adjointMatrix(i, j));
            vec.push_back(temp);
        }
        Matrix<T> out(vec);
        out *= 1 / determinant();
        return out;
    }

    T adjointMatrix(int r, int c) {
        vector<vector<T> > vec;
        for (int i = 0; i < rows; ++i) {
            if (i == r) continue;
            vector<T> temp;
            for (int j = 0; j < cols; ++j) {
                if (j == c) continue;
                temp.push_back(matrix[i][j]);
            }
            vec.push_back(temp);
        }
        return pow(-1, r + c) * detUtility(vec, rows - 1);
    }
    
    
    
    // note: Put Lei Qirong's codes here:
	
    /**
     * @brief reshape the matrix to a new form by certain row and column number
     *
     * @param row the number of rows of new matrix
     * @param column the number of cols of new matrix
     * @return the matrix after reshape. If invalid, return the original matrix.
     *
     */
    Matrix<T> reshape(int row, int column)
    {

        if (this->rows * this->cols != row * column)
        {
            cout << "The parameter is not valid!";
            return Matrix<T>(0, 0);
        }
        Matrix<T> output(row, column);
        int count = 0;
        while (count < row * column)
        {
            T i = count / this->cols;
            T j = count % this->cols;
            T ii = count / column;
            T jj = count % column;
            output.matrix[ii][jj] = this->matrix[i][j];
            count++;
        }

        return output;
    }

    /**
     * @brief reshape the matrix to certain row or column number automatically
     *
     * @param num the number of rows or column
     * @param isRow Whether @param num represents the number of row. If not, it represent the number of column.
     * @return the matrix after reshape. If invalid, return the original matrix.
     *
     */
    Matrix<T> reshape(int num, bool isRow)
    {
        if (this->rows * this->cols % num != 0)
        {
            cout << "The parameter is not valid!";
            return Matrix<T>(this->rows,this->cols);
        }
        else
        {
            int the_other_param = this->rows * this->cols % num;
            if (isRow)
            {
                return reshape(num, the_other_param);
            }
            else
                return reshape(the_other_param, num);
        }
    }

    /**
     * @brief reshape the matrix from 3-dimension to 2-dimension with certain row and column
     *
     * @param A the 3-dim array
     * @param x @param y @param z the length, the width and the height of @param A
     * @param row @param column the number of rows and column of the new matrix
     * @return the matrix after reshape. If invalid, return a matrix with required size but filled with 0.
     *
     * example:
     * Matrix m(a,b);
     * m.reshape(A,x,y,z);
     *
     */
    Matrix<T> reshape(vector<vector<vector<T> > > A, int x, int y, int z)
    {
        if (x * y * z != this->rows * this->cols)
        {
            cout << "The parameter is not valid!";
            return Matrix<T>(0,0);
        }
        int x_count = 0, y_count = 0, z_count = 0;
        Matrix<T> output(this->rows,this->cols);
        for (int i = 0; i < this->rows; i++)
        {
            for (int j = 0; j < this->cols; j++)
            {
                if (z_count == z)
                {
                    z_count = 0;
                    y_count++;
                }
                if (y_count == y)
                {
                    y_count = 0;
                    x_count++;
                }
                output.matrix[i][j] = A[x_count][y_count][z_count];
                z_count++;
            }
        }
        return output;
    }
	
    /**
     * @brief get the spice part of the giving matrix
     *
     * @param start_row , @param end_row represents the start and end row number.
     * @param start_col , @param end_col represents the start and end cloumn number.
     * @return the matrix after slicing
     *
     */
    Matrix<T> slice(int start_row, int end_row, int start_col, int end_col)
    {
        // Handle the exception  of invalid input.
        if (start_row < 0 || start_col < 0 || start_row >= this->rows || start_col >= this->cols || end_row < 0 || end_col < 0 || end_row >= this->rows || end_col >= this->cols)
        {
            cerr << "Invalid input!" << endl;
            return Matrix<T>(0, 0);
        }
        // Handle the exception that input the opposite direction of start and end number
        int x1, y1, x2, y2;
        if (start_row > end_row)
        {
            x1 = end_row;
            x2 = start_row;
        }
        else
        {
            x2 = end_row;
            x1 = start_row;
        }
        if (start_col > end_col)
        {
            y1 = end_col;
            y2 = start_col;
        }
        else
        {
            y2 = end_col;
            y1 = start_col;
        }
        Matrix<T> output(x2 - x1 + 1, y2 - y1 + 1);
        for (int i = 0; i < output.rows; i++)
        {
            for (int j = 0; j < output.cols; j++)
            {
                output.matrix[i][j] = this->matrix[x1 + i][y1 + j];
            }
        }
        return output;
    }

    /**
     * @brief the convolution calculation of two matrix
     *
     * @param m2 Another matrix to be convolutioned by this matrix
     * @return the matrix after convolution: A conv B;
     *
     * example:
     * Matrix m(a,b);
     * Matrix n(c,d);
     * m.reshape(n);
     *
     */
    Matrix<T> convolution(Matrix &m2)
    {
        // turn 180˚ of the matrix be convolutioned
        m2=m2.trans().trans();
        // extend the size of the matrix
        Matrix<T> output(this->rows, this->cols);
        for (int i = 0; i < output.rows; i++)
        {
            for (int j = 0; j < output.cols; j++)
            {
                int value = 0;
                for (int ii = 0; ii < m2.rows; ii++)
                {
                    for (int jj = 0; jj < m2.cols; jj++)
                    {
                        int x = i + ii - m2.rows / 2;
                        int y = j + jj - m2.cols / 2;
                        if (x >= 0 && y >= 0 && x < this->rows && y < this->cols)
                        {
                            value += this->matrix[x][y] * m2.matrix[ii][jj];
                        }
                    }
                }
                output.matrix[i][j] = value;
            }
        }
        return output;
    }
};

template<typename T>
struct element {
    int row, col;
    T value;
	
    element() : row(0), col(0), value(0) {}

    element(int row, int col, T value) : row(row), col(col), value(value) {}

    bool operator<(const element &rhs) const {
        if (row < rhs.row)
            return true;
        if (rhs.row < row)
            return false;
        return col < rhs.col;
    }
};

template<class T>
class SpareMatrix {
private:
    int rows, cols;
    int items, maxItems;
    vector<element<T> > spareMatrix;

public:
    // Constructors
    SpareMatrix() : rows(0), cols(0), items(0), maxItems(0) { spareMatrix.resize(0); }

    SpareMatrix(int row, int col) : rows(row), cols(col), items(0) {
        maxItems = row * col;
        spareMatrix.resize(0);
    }

    SpareMatrix(int row, int col, vector<element<T> > vec) {
        rows = row;
        cols = col;
        items = vec.size();
        maxItems = row * col;
        spareMatrix = vec;
    }

    SpareMatrix(const SpareMatrix<T> &sm) {
        rows = sm.getRows();
        cols = sm.getCols();
        items = sm.getItems();
        maxItems = sm.getMaxItems();
        spareMatrix = sm.getSpareMatrix();
    }

    explicit SpareMatrix(const Matrix<T> &m) {
        rows = m.getRows();
        cols = m.getCols();
        items = 0;
        maxItems = rows * cols;
        vector<vector<T> > matrix = m.getMatrix();
        for (int i = 0; i < matrix.size(); ++i) {
            for (int j = 0; j < matrix[0].size(); ++j) {
                if (matrix[i][j] != 0) {
                    element<T> temp(i, j, matrix[i][j]);
                    items++;
                    spareMatrix.push_back(temp);
                }
            }
        }
    }

    // Getters
    int getRows() const { return rows; }

    int getCols() const { return cols; }

    int getItems() const { return items; }

    int getMaxItems() const { return maxItems; }

    const vector<element<T> > &getSpareMatrix() const { return spareMatrix; }

    // Others func
    bool insert(int row, int col, T val) {
        element<T> e(row, col, val);
        return insert(e);
    }

    bool insert(element<T> &e){
        if (!(e.row >= 0 && e.row < rows) || !(e.col >= 0 && e.col < cols)) {
            cerr << "error in insert()! Out the Spare Matrix bounds!" << endl;
            return false;
        }
        // if it has duplication
        for (int i = 0; i < spareMatrix.size(); ++i) {
            if (spareMatrix[i].row == e.row && spareMatrix[i].col == e.col) {
                spareMatrix[i].value = e.value;
                return true;
            }
        }
        // else
        items++;
        spareMatrix.push_back(e);
        sort(spareMatrix.begin(), spareMatrix.end());
        return true;
    }

    friend ostream &operator<<(ostream &os, const SpareMatrix &sm) {
        os << "Spare Matrix with (row, col) = (" << sm.rows << ", "
            << sm.cols << ")" << endl;
        int index = 0;
        os << "[";
        for (int i = 0; i < sm.rows; ++i) {
            if (i != 0) os << " ";
            if (sm.spareMatrix[index].row == i && sm.spareMatrix[index].col == 0) {
                os << "[" << sm.spareMatrix[index].value;
                index++;
            } else os << "[0";
            for (int j = 1; j < sm.cols; ++j) {
                if (sm.spareMatrix[index].row == i && sm.spareMatrix[index].col == j) {
                    os << " " << sm.spareMatrix[index].value;
                    index++;
                } else os << " 0";
            }
            if (i != sm.rows - 1) os << "]\n";
            else os << "]]\n";
        }
        return os;
    }
};

//cv::abs()	矩阵内所有元素取绝对值并返回结果
//cv::absdiff()	计算两个矩阵差值的绝对值并返回结果
//cv::add()	两个矩阵逐元素相加
//cv::addWeighted()	两个矩阵逐元素加权求和，可以理解为Alpha混合
//cv::bitwise_and()	两个矩阵逐元素按位与运算
//cv::bitwise_not()	两个矩阵逐元素按位非运算
//cv::bitwise_or()	两个矩阵逐元素按位或运算
//cv::bitwise_xor()	两个矩阵逐元素按位异或运算
//cv::calcCovarMatrix()	计算一组n维向量的协方差
//cv::cartToPolar()	计算二维向量的角度和幅度
//cv::checkRange()	检查矩阵的无效值
//cv::compare()	对两个矩阵中的所有元素应用一个指定的比较运算符
//cv::completeSymm()	通过将一半元素复制到另一半使得矩阵对称
//cv::convertScaleAbs()	缩放矩阵，取绝对值，然后将其中数据格式转化为8位无符号型
//cv::countNonZero()	计算矩阵中的非零元素
//cv::arrToMat()	将2.1版本之前的数组转化为cv::Mat的实例
//cv::dct()	计算矩阵的离散余弦变换
//cv::determinant()	计算方阵的行列式
//cv::dft()	计算矩阵的离散傅立叶变换
//cv::divide()	对两个矩阵执行逐元素除法运算
//cv::eigen()	计算方针的特征值和特征向量
//cv::exp()	对矩阵执行逐元素求指数幂运算
//cv::extractImageCOI()	从2.1之前版本的数组中提取单个通道
//cv::flip()	绕指定轴翻转矩阵
//cv::gemm()	执行广义的矩阵乘法
//cv::getConvertElem()	获取单个像素的类型转换函数
//cv::getConvertScaleElem()	获取单个像素的类型转换和缩放函数
//cv::idct()	计算矩阵的离散余弦逆变换
//cv::idft()	计算矩阵的离散傅立叶逆变换
//cv::inRange()	测试矩阵的元素是否包含在其他两个矩阵的值之间
//cv::invert()	求方阵的逆
//cv::log()	逐元素计算自然对数
//cv::magnitude()	计算二维向量的幅度
//cv::LUT()	将矩阵转换为查找表的索引
//cv::Mahalanobis()	计算两个向量之间的马氏距离
//cv::max()	逐元素求两个矩阵之间的最大值
//cv::mean()	计算矩阵元素的平均值
//cv::meanStdDev()	计算数组元素的均值和标准差
//cv::merge()	将多个单通道矩阵合并为一个多通道矩阵
//cv::min()	逐元素求两个矩阵之间的最小值
//cv::minMaxLoc()	在矩阵中寻找最大和最小值
//cv::mixChannels()	打乱从输入矩阵到输出矩阵的通道
//cv::mulSpectrums()	对两个傅立叶谱矩阵执行逐元素乘法运算
//cv::multiply()	对两个矩阵执行逐元素乘法运算
//cv::mulTransposed()	计算矩阵和其转置对逐元素乘积
//cv::norm()	在两个矩阵之间计算归一化相关系数
//cv::normalize()	将矩阵中对元素标准化到某个值内
//cv::perspectiveTransform()	执行一系列向量的透视矩阵变换
//cv::phase()	计算二维向量的方向
//cv::polarToCart()	已知角度和幅度，求二维向量
//cv::pow()	对矩阵内对每个元素执行幂运算
//cv::randu()	使用均匀分布的随机数填充矩阵
//cv::randn()	使用正态分布的随机数填充矩阵
//cv::randShuffle()	随机打乱矩阵元素
//cv::reduce()	通过特定的操作将二维矩阵退化为向量
//cv::repeat()	将一个矩阵的内容复制到另外一个矩阵
//cv::saturate_cast<>()	饱和转换原始类型
//cv::scaleAdd()	逐元素的执行矩阵加法，第一个矩阵可以选择先执行缩放操作
//cv::setIdentity()	将对角线上的元素设置为1，其余元素设置为0
//cv::solve()	求出线性方程组的解
//cv::solveCubic()	计算三次方程的实根
//cv::solvePoly()	找到多项式方程的复根
//cv::sort()	排序矩阵中的任意行或者列的所有元素
//cv::sortIdx()	和函数cv::sort()类似，但是这里并不会修改矩阵本身，仅返回排序结果的索引值
//cv::split()	将多通道矩阵分解为多个单通道矩阵
//cv::sqrt()	逐元素计算矩阵的平方根
//cv::subtract()	逐元素对两个矩阵执行减法运算
//cv::sum()	计算数组所有元素的和
//cv::theRNG()	返回一个随机数生成器
//cv::trace()	计算一个矩阵的迹
//cv::transform()	对矩阵的每个元素应用矩阵变换
//cv::transpose()	计算矩阵的转置矩阵
//hhh
