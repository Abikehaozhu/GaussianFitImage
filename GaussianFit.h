#pragma once
#pragma once
#ifndef GAUSSIANFIT_H
#define GAUSSIANFIT_H
#include<opencv2/opencv.hpp>
//高斯拟合的一些函数

//计算某一点的高斯函数值
double gaussian(double x, double y, double A, double sigma_x, double sigma_y);

//计算某个邻域的EMS误差
double CalSumEMS(cv::Mat neibor, double A, double sigma_x, double sigma_y);

//计算幅值的梯度
double gradA(cv::Mat neibor, double A, double sigma_x, double sigma_y);

//计算X方向标准差的梯度
double gradSigmaX(cv::Mat neibor, double A, double sigma_x, double sigma_y);

double gradSigmaY(cv::Mat neibor, double A, double sigma_x, double sigma_y);

//用于拟合高斯参数的代码
double LM(cv::Mat neighbor, std::vector<double>& para, double eps = 0.0001,
	double steplength = 0.0001, int epoch = 100000);

//用于创造高斯曲面的掩膜
cv::Mat CreateMask(std::vector<double> para, const int width);



#endif