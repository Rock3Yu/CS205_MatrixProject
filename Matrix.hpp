#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
//#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

template<typename T>
class Matrix {
private:
    vector<vector<T>> matrix;
    int rows, cols;

public:
    Matrix() : rows(0), cols(0) { matrix.resize(0, 0); }

    Matrix(const int r, const int c) {
        rows = r;
        cols = c;
        matrix = new T *[rows];
        for (int i = 0; i < rows; ++i) {
            matrix[i] = new T[cols];
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = 0;
            }
        }
    }

    void Display() {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << matrix[i][j] << " ";
            }
            cout << endl;
        }
    }

    Matrix add(const Matrix &m1, const Matrix &m2);

    Matrix subtract(const Matrix &m1, const Matrix &m2);

    Matrix operator+(const Matrix&m);

};

//cv::abs()	����������Ԫ��ȡ����ֵ�����ؽ��
//cv::absdiff()	�������������ֵ�ľ���ֵ�����ؽ��
//cv::add()	����������Ԫ�����
//cv::addWeighted()	����������Ԫ�ؼ�Ȩ��ͣ��������ΪAlpha���
//cv::bitwise_and()	����������Ԫ�ذ�λ������
//cv::bitwise_not()	����������Ԫ�ذ�λ������
//cv::bitwise_or()	����������Ԫ�ذ�λ������
//cv::bitwise_xor()	����������Ԫ�ذ�λ�������
//cv::calcCovarMatrix()	����һ��nά������Э����
//cv::cartToPolar()	�����ά�����ĽǶȺͷ���
//cv::checkRange()	���������Чֵ
//cv::compare()	�����������е�����Ԫ��Ӧ��һ��ָ���ıȽ������
//cv::completeSymm()	ͨ����һ��Ԫ�ظ��Ƶ���һ��ʹ�þ���Գ�
//cv::convertScaleAbs()	���ž���ȡ����ֵ��Ȼ���������ݸ�ʽת��Ϊ8λ�޷�����
//cv::countNonZero()	��������еķ���Ԫ��
//cv::arrToMat()	��2.1�汾֮ǰ������ת��Ϊcv::Mat��ʵ��
//cv::dct()	����������ɢ���ұ任
//cv::determinant()	���㷽�������ʽ
//cv::dft()	����������ɢ����Ҷ�任
//cv::divide()	����������ִ����Ԫ�س�������
//cv::eigen()	���㷽�������ֵ����������
//cv::exp()	�Ծ���ִ����Ԫ����ָ��������
//cv::extractImageCOI()	��2.1֮ǰ�汾����������ȡ����ͨ��
//cv::flip()	��ָ���ᷭת����
//cv::gemm()	ִ�й���ľ���˷�
//cv::getConvertElem()	��ȡ�������ص�����ת������
//cv::getConvertScaleElem()	��ȡ�������ص�����ת�������ź���
//cv::idct()	����������ɢ������任
//cv::idft()	����������ɢ����Ҷ��任
//cv::inRange()	���Ծ����Ԫ���Ƿ�������������������ֵ֮��
//cv::invert()	�������
//cv::log()	��Ԫ�ؼ�����Ȼ����
//cv::magnitude()	�����ά�����ķ���
//cv::LUT()	������ת��Ϊ���ұ������
//cv::Mahalanobis()	������������֮������Ͼ���
//cv::max()	��Ԫ������������֮������ֵ
//cv::mean()	�������Ԫ�ص�ƽ��ֵ
//cv::meanStdDev()	��������Ԫ�صľ�ֵ�ͱ�׼��
//cv::merge()	�������ͨ������ϲ�Ϊһ����ͨ������
//cv::min()	��Ԫ������������֮�����Сֵ
//cv::minMaxLoc()	�ھ�����Ѱ��������Сֵ
//cv::mixChannels()	���Ҵ����������������ͨ��
//cv::mulSpectrums()	����������Ҷ�׾���ִ����Ԫ�س˷�����
//cv::multiply()	����������ִ����Ԫ�س˷�����
//cv::mulTransposed()	����������ת�ö���Ԫ�س˻�
//cv::norm()	����������֮������һ�����ϵ��
//cv::normalize()	�������ж�Ԫ�ر�׼����ĳ��ֵ��
//cv::perspectiveTransform()	ִ��һϵ��������͸�Ӿ���任
//cv::phase()	�����ά�����ķ���
//cv::polarToCart()	��֪�ǶȺͷ��ȣ����ά����
//cv::pow()	�Ծ����ڶ�ÿ��Ԫ��ִ��������
//cv::randu()	ʹ�þ��ȷֲ��������������
//cv::randn()	ʹ����̬�ֲ��������������
//cv::randShuffle()	������Ҿ���Ԫ��
//cv::reduce()	ͨ���ض��Ĳ�������ά�����˻�Ϊ����
//cv::repeat()	��һ����������ݸ��Ƶ�����һ������
//cv::saturate_cast<>()	����ת��ԭʼ����
//cv::scaleAdd()	��Ԫ�ص�ִ�о���ӷ�����һ���������ѡ����ִ�����Ų���
//cv::setIdentity()	���Խ����ϵ�Ԫ������Ϊ1������Ԫ������Ϊ0
//cv::solve()	������Է�����Ľ�
//cv::solveCubic()	�������η��̵�ʵ��
//cv::solvePoly()	�ҵ�����ʽ���̵ĸ���
//cv::sort()	��������е������л����е�����Ԫ��
//cv::sortIdx()	�ͺ���cv::sort()���ƣ��������ﲢ�����޸ľ�����������������������ֵ
//cv::split()	����ͨ������ֽ�Ϊ�����ͨ������
//cv::sqrt()	��Ԫ�ؼ�������ƽ����
//cv::subtract()	��Ԫ�ض���������ִ�м�������
//cv::sum()	������������Ԫ�صĺ�
//cv::theRNG()	����һ�������������
//cv::trace()	����һ������ļ�
//cv::transform()	�Ծ����ÿ��Ԫ��Ӧ�þ���任
//cv::transpose()	��������ת�þ���
