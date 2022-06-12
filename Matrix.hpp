#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
//#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

template<class T>
class Matrix {
private:
    vector<vector<T>> matrix;
    int rows, cols;

public:
    // Constructors
    Matrix() : rows(0), cols(0) { matrix.resize(0, 0); }

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
    explicit Matrix(vector<vector<T>> matrix) {
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
    
    Matrix &operator=(const Matrix<T> &m) {
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
            else os << "]]";
        }
        return os;
    }
    
    
    //note: Put Lin Peijun's codes here:
    
    Matrix<T>	operator+(const Matrix<T> &m) const ;	
	Matrix<T>	operator-(const Matrix<T> &m) const ;		
	Matrix<T>	operator*(const Matrix<T> &m) const ;	
	void		operator+=(const Matrix<T> &m) ;			
	void		operator-=(const Matrix<T> &m) ;			
	void		operator*=(const Matrix<T> &m) ;			
	void		operator*=(double value) ;					
	Matrix<T>	    operator*(const double scalar) ;
    Matrix<T>	    operator/(const double scalar) ;
	bool		operator==(const Matrix<T> &m) const ;
    
    Matrix<T>   trans() const ;
    Matrix<T>   cong() const ;
    Matrix<T>      eleWiseMul(const Matrix<T> &m) ;
    Matrix<T>      crossPro(const Matrix<T> &m);
    Matrix<T>      dotPro(const Matrix<T> &m);
    
     //addition
Matrix<T>	Matrix::operator+(const Matrix<T> &m){
    try{
        if(m.colunm == cols && m.row == rows){
        Matrix<T> M(rows,cols);
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                M[i,j] = matrix[i,j] + m[i,j];
            }
        }
        return M;
    }else{
        throw "The sizes do not match when using \'+\'";
    }
    } catch(char *exceptionString) {
        cerr << exceptionString << endl;
    }
}

//subtraction
Matrix<T>	operator-(const Matrix<T> &m){
    try{
        if(m.colunm == cols && m.row == rows){
        return (*this + (-1)*m);
    }else{
        throw "The sizes do not match when using \'-\'";
    }
    } catch(char *exceptionString) {
        cerr << exceptionString << endl;
    }
}		

// matrix-matrix multiplication & matrix-vector multiplication
Matrix<T>	operator*(const Matrix<T> &m){
    try{
         if(m.colunm == rows && m.row == cols){
        Matrix M(rows,cols);
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                M[i,j] += (matrix[i,j] + m[j,i]);
            }
        }
        return M;
    }else{
        throw "The sizes do not match when using \'*\'";
            }

    } catch(char *exceptionString) {
        cerr << exceptionString << endl;
    }
   }

//scalar multiplication
Matrix<T>	operator*(const double scalar){
    Matrix<T> M(rows,cols);
     for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                M[i,j] = matrix[i,j] * scalar;
            }
        }
    return M;
}

//scalar division
Matrix<T>	operator/(const double scalar){
    try{
         Matrix<T> M(rows,cols);
        if scalar == 0{
        throw "0 is denominator";
        }
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                M[i,j] = matrix[i,j] / scalar;
            }
        }
        return M;
    } catch(char *exceptionString) {
        cerr << exceptionString << endl;
    }
   }

void	operator+=(const Matrix<T> &m) {
    *this = *this + m;
}

void	operator-=(const Matrix<T> &m) {
    *this = *this - m;
}			

void	operator*=(const Matrix<T> &m){
    *this = *this * m;
}			

void	operator/=(double value) {
    *this = *this / m;
}	

bool	operator==(const Matrix<T> &m){
    if(m.colunm == cols && m.row == rows){
        int num = 0;
        Matrix<T> M(rows,cols);
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                M[i,j] = matrix[i,j] - m[i,j];
                if(M[i,j] == 0)
                num++;
            }
        }
        if(num == cols*rows)
        return true;
        else
        return false;
    }else{
        return false;
    }
}


//transposition    
Matrix<T>   trans(){
    Matrix<T> M(cols,rows);
    for(int i=0; i<cols; i++){
        for(int j=0; j<rows; j++){
            M[i,j] = matrix[j,i];
        }
    }
    return M;
}

//conjugation
Matrix<T>   cong() {

}

//element-wise multiplication
Matrix<T>   eleWiseMul(const Matrix<T> &m){
    try{
         if(m.colunm == cols && m.row == rows){
        Matrix<T> M(rows,cols);
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            M[i,j] = matrix[i,j] * m[i,j];
        }
    }return M;
    }else{
        throw "The sizes do not match when using \'eleWiseMul\'";
    }
    } catch(char *exceptionString) {
        cerr << exceptionString << endl;
    }
   }

//cross product
Matrix<T>   crossPro(const Matrix<T> &m){
    try{
        if(m.colunm == rows && m.row == cols){
        Matrix<T> M(rows,cols);
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                M[i,j] += (matrix[i,j] + m[j,i]);
            }
        }
        return M;
        }else{
            throw "The sizes do not match when using \'crossPro\'";
        }
    } catch(char *exceptionString) {
        cerr << exceptionString << endl;
    }
}

//dot product
Matrix<T>   dotPro(const Matrix<T> &m){
    try{
        if(m.colunm == cols && m.row == rows){
            Matrix<T> M(rows,cols);
            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                M[i,j] = matrix[i,j] * m[i,j];
                }
            }return M;
        }else{
            throw "The sizes do not match when using \'dotPro\'";
        }
    } catch(char *exceptionString) {
        cerr << exceptionString << endl;
    }
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
            // note: max in row(num)
            max = matrix[num][0];
            for (int i = 0; i < cols; ++i) if (max < matrix[num][i]) max = matrix[num][i];
        } else {
            if (num >= cols) cerr << "error in function: T max(int num, bool axis);" << endl;
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
            // note: max in row(num)
            min = matrix[num][0];
            for (int i = 0; i < cols; ++i) if (min > matrix[num][i]) min = matrix[num][i];
        } else {
            if (num >= cols) cerr << "error in function: T min(int num, bool axis);" << endl;
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
            // note: sum in row(num)
            sum = matrix[num][0];
            for (int i = 1; i < cols; ++i) sum += matrix[num][i];
        } else {
            if (num >= cols) cerr << "error in function: T sum(int num, bool axis);" << endl;
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
    
    
    
    // note: Put Lei Qirong's codes here:
    
    
    
    

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
