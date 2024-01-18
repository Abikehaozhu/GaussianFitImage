#include"GaussianFit.h"
#include<opencv.hpp>

const int roi_width = 7; //Ҫ�����������С

double gaussian(double x, double y, double A, double sigma_x, double sigma_y)//���ڼ���һ�㴦�ĸ�˹����ֵ
{
	return A * exp(-(pow(x - int(roi_width / 2), 2) / 2 / pow(sigma_x, 2)
		+ pow(y - int(roi_width / 2), 2) / 2 / pow(sigma_y, 2)));//����Ĭ������Ϊ(3,3)��������5*5��С����
}

double CalSumEMS(cv::Mat neibor, double A, double sigma_x, double sigma_y)//��������֮���EMS��࣬��������5*5��Mat
{
	double sum = 0;
	for (int i = 0; i < neibor.cols; i++)
	{
		for (int j = 0; j < neibor.cols; j++)
		{
			double img_value = neibor.at<double>(i, j);//ȡ���õ㴦������ֵ
			//std::cout << img_value << " " << gaussian(i, j, A, sigma_x, sigma_y) << std::endl;
			sum = sum + pow(img_value - gaussian(i, j, A, sigma_x, sigma_y), 2);
		}
	}
	return sum;
}

double gradA(cv::Mat neibor, double A, double sigma_x, double sigma_y)//����EMS�ͶԷ�ֵ��ƫ����
{
	double sum = 0;
	for (int i = 0; i < neibor.cols; ++i)
	{
		for (int j = 0; j < neibor.cols; j++)
		{
			double img_value = neibor.at<double>(i, j);//ȡ���õ㴦������ֵ
			sum = sum + (gaussian(i, j, A, sigma_x, sigma_y) - img_value) * gaussian(i, j, A, sigma_x, sigma_y) / A;
		}
	}
	return 2 * sum / neibor.cols / neibor.cols;
}

double gradSigmaX(cv::Mat neibor, double A, double sigma_x, double sigma_y)//����EMS�ͶԷ����ƫ����
{
	double sum = 0;
	for (int i = 0; i < neibor.cols; i++)
	{
		for (int j = 0; j < neibor.cols; j++)
		{
			double img_value = neibor.at<double>(i, j);//ȡ���õ㴦������ֵ
			sum = sum + (gaussian(i, j, A, sigma_x, sigma_y) - img_value) * gaussian(i, j, A, sigma_x, sigma_y)
				* (pow((i - neibor.cols / 2), 2) / pow(sigma_x, 3));
		}
	}
	return sum * 2 / neibor.cols / neibor.cols;
}

double gradSigmaY(cv::Mat neibor, double A, double sigma_x, double sigma_y)//����EMS�ͶԷ����ƫ����
{
	double sum = 0;
	for (int i = 0; i < neibor.cols; i++)
	{
		for (int j = 0; j < neibor.cols; j++)
		{
			double img_value = neibor.at<double>(i, j);//ȡ���õ㴦������ֵ
			sum = sum + (gaussian(i, j, A, sigma_x, sigma_y) - img_value) * gaussian(i, j, A, sigma_x, sigma_y)
				* (pow((j - neibor.cols / 2), 2) / pow(sigma_x, 3));
		}
	}
	return sum * 2 / neibor.cols / neibor.cols;
}

double LM(cv::Mat neighbor, std::vector<double>& para, double eps, double steplength, int epoch)
//������ϸ�˹�����LM�����㷨��ֻ������5*5������para��Ӧ��ΪA��sigma_X,sigma_Y
{
	int iter = 0;//���ڼ�¼��������
	//std::cout << CalSumEMS(neighbor, para[0], para[1], para[2]);
	while (CalSumEMS(neighbor, para[0], para[1], para[2]) > eps && iter < epoch)
	{
		iter++;
		//std::cout << iter;
		//std::cout << para[1] << "  ";
		double A = para[0] - steplength * gradA(neighbor, para[0], para[1], para[2]);
		double sigma_x = para[1] - steplength * gradSigmaX(neighbor, para[0], para[1], para[2]);
		double sigma_y = para[2] - steplength * gradSigmaY(neighbor, para[0], para[1], para[2]);
		//std::cout << gradA(neighbor, para[0], para[1], para[2]) << "  "
		//	<< gradSigmaX(neighbor, para[0], para[1], para[2]) << " "
		//	<< gradSigmaY(neighbor, para[0], para[1], para[2]) << " ";
		//std::cout << CalSumEMS(neighbor, para[0], para[1], para[2]) << std::endl;
		para[0] = A;
		para[1] = sigma_x;
		para[2] = sigma_y;
	}
	return CalSumEMS(neighbor, para[0], para[1], para[2]);
}

cv::Mat CreateMask(std::vector<double> para, const int width)//����ָ����С���ɰ�
{
	cv::Mat mask(width, width, CV_64F);
	//std::cout << para[0] <<" "<< para[1] <<" " << para[2] << std::endl;
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < width; ++j) {
			double x = j;
			double y = i;
			double exponent_x = (((x - int(width / 2)) * (x - int(width / 2)))) / (2 * para[1] * para[1]);
			double exponent_y = ((y - int(width / 2)) * (y - int(width / 2))) / (2 * para[2] * para[2]);
			double value = para[0] * exp(-(exponent_x + exponent_y));
			//std::cout << exponent_x << " " << exponent_y << " " << std::endl;
			mask.at<double>(i, j) = value;
		}
	}
	return mask;
}