#pragma once
#pragma once
#ifndef GAUSSIANFIT_H
#define GAUSSIANFIT_H
#include<opencv2/opencv.hpp>
//��˹��ϵ�һЩ����

//����ĳһ��ĸ�˹����ֵ
double gaussian(double x, double y, double A, double sigma_x, double sigma_y);

//����ĳ�������EMS���
double CalSumEMS(cv::Mat neibor, double A, double sigma_x, double sigma_y);

//�����ֵ���ݶ�
double gradA(cv::Mat neibor, double A, double sigma_x, double sigma_y);

//����X�����׼����ݶ�
double gradSigmaX(cv::Mat neibor, double A, double sigma_x, double sigma_y);

double gradSigmaY(cv::Mat neibor, double A, double sigma_x, double sigma_y);

//������ϸ�˹�����Ĵ���
double LM(cv::Mat neighbor, std::vector<double>& para, double eps = 0.0001,
	double steplength = 0.0001, int epoch = 100000);

//���ڴ����˹�������Ĥ
cv::Mat CreateMask(std::vector<double> para, const int width);



#endif