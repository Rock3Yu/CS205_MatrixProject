#include <opencv2/opencv.hpp>
#include <iostream>

// opencv_demo.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.hpp"

using namespace cv;
using namespace std;

/**
 *@brief read image by imread
 *turning mat into matrix
 *using the matrix library to process images
 *transfer matrix into mat
 *@param img the Mat form of picture @param m the Matrix form of picture @param temp the convolution kernel
 *@param img2 Edge Detection @param img3 Laplace sharpening
*/
int main(int args, char **argv)
{
cout << "OpenCV Version: " << CV_VERSION << endl;
Mat img = imread("C:/Users/suoquan/Documents/Tencent Files/1516859433/FileRecv/DSCN0721.jpg",1);
Matrix<char> m;
m = m.mat2Matrix(img);
Mat img1 = m.slice(0,m.getRows()/4*3,0,m.getCols()/4*3).matrix2Mat(3);
vector<vector<char>> temp = {{0,0,-1,0,0},{0,0,-1,0,0},{0,0,4,0,0},{0,0,-1,0,0},{0,0,-1,0,0}}; 
Matrix<char> kernel(temp);
Mat img2 = m.convolution(kernel).matrix2Mat(3);
temp = {{0,-1,0},{-1,4,-1},{0,-1,0}}; 
Matrix<char> kernel2(temp);
Mat img3 = m.convolution(kernel2).matrix2Mat(3);
imwrite("F:\\opencv\\LANE_DETECTION\\img/read.jpg",img);
imwrite("F:\\opencv\\LANE_DETECTION\\img/slice.jpg",img1);
imwrite("F:\\opencv\\LANE_DETECTION\\img/edgeDetect.jpg",img2);
imwrite("F:\\opencv\\LANE_DETECTION\\img/LaSharpen.jpg",img3);
return 0;
}
