#include"GaussianFit.h"
#include<opencv.hpp>

const int roi_width = 7; //要分析的邻域大小

double gaussian(double x, double y, double A, double sigma_x, double sigma_y)//用于计算一点处的高斯函数值
{
	return A * exp(-(pow(x - int(roi_width / 2), 2) / 2 / pow(sigma_x, 2)
		+ pow(y - int(roi_width / 2), 2) / 2 / pow(sigma_y, 2)));//这里默认中心为(3,3)，适用于5*5的小邻域
}

double CalSumEMS(cv::Mat neibor, double A, double sigma_x, double sigma_y)//计算两者之间的EMS差距，仅适用于5*5的Mat
{
	double sum = 0;
	for (int i = 0; i < neibor.cols; i++)
	{
		for (int j = 0; j < neibor.cols; j++)
		{
			double img_value = neibor.at<double>(i, j);//取出该点处的像素值
			//std::cout << img_value << " " << gaussian(i, j, A, sigma_x, sigma_y) << std::endl;
			sum = sum + pow(img_value - gaussian(i, j, A, sigma_x, sigma_y), 2);
		}
	}
	return sum;
}

double gradA(cv::Mat neibor, double A, double sigma_x, double sigma_y)//计算EMS和对幅值的偏导数
{
	double sum = 0;
	for (int i = 0; i < neibor.cols; ++i)
	{
		for (int j = 0; j < neibor.cols; j++)
		{
			double img_value = neibor.at<double>(i, j);//取出该点处的像素值
			sum = sum + (gaussian(i, j, A, sigma_x, sigma_y) - img_value) * gaussian(i, j, A, sigma_x, sigma_y) / A;
		}
	}
	return 2 * sum / neibor.cols / neibor.cols;
}

double gradSigmaX(cv::Mat neibor, double A, double sigma_x, double sigma_y)//计算EMS和对方差的偏导数
{
	double sum = 0;
	for (int i = 0; i < neibor.cols; i++)
	{
		for (int j = 0; j < neibor.cols; j++)
		{
			double img_value = neibor.at<double>(i, j);//取出该点处的像素值
			sum = sum + (gaussian(i, j, A, sigma_x, sigma_y) - img_value) * gaussian(i, j, A, sigma_x, sigma_y)
				* (pow((i - neibor.cols / 2), 2) / pow(sigma_x, 3));
		}
	}
	return sum * 2 / neibor.cols / neibor.cols;
}

double gradSigmaY(cv::Mat neibor, double A, double sigma_x, double sigma_y)//计算EMS和对方差的偏导数
{
	double sum = 0;
	for (int i = 0; i < neibor.cols; i++)
	{
		for (int j = 0; j < neibor.cols; j++)
		{
			double img_value = neibor.at<double>(i, j);//取出该点处的像素值
			sum = sum + (gaussian(i, j, A, sigma_x, sigma_y) - img_value) * gaussian(i, j, A, sigma_x, sigma_y)
				* (pow((j - neibor.cols / 2), 2) / pow(sigma_x, 3));
		}
	}
	return sum * 2 / neibor.cols / neibor.cols;
}

double LM(cv::Mat neighbor, std::vector<double>& para, double eps, double steplength, int epoch)
//用于拟合高斯曲面的LM迭代算法，只适用于5*5的邻域，para的应当为A，sigma_X,sigma_Y
{
	int iter = 0;//用于记录迭代次数
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

cv::Mat CreateMask(std::vector<double> para, const int width)//创造指定大小的蒙版
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