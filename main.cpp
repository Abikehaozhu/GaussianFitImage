#include <opencv2/opencv.hpp>
#include <vector>
#include"GaussianFit.h"

// 全局变量用于存储点击点的坐标
std::vector<cv::Rect> clickedPoints;

// 全局变量用于存储滑动条的值
int radius = 2;

//用于显示小ROI的Rect
cv::Rect smallRoi(0, 0, 2 * radius, 2 * radius);
cv::Rect selectRoi(0, 0, 100, 100);//进一步选择的区域
cv::Point centerRoi(0, 0);//当前展示区域的中心
int select = -1;//当前是否已经选择过区域

//用于计算3*3邻域的灰度值与每一圈的灰度平均值
int calNeibor(cv::Mat neibor, std::vector<double>& para)
{
    //如果图像过小则不进行处理
    if (neibor.cols < 3)
    {
        std::cout << "邻域过小" << std::endl;
        return 0;
    }

    //转化三通道
    if (neibor.channels() == 3)
    {
        cv::cvtColor(neibor, neibor, cv::COLOR_BGR2GRAY);
    }

    //首先计算3*3的灰度值
    // 获取图像的中心坐标
    int centerX = neibor.cols / 2;
    int centerY = neibor.rows / 2;
    // 提取中心3x3邻域
    int size = 3; // 邻域尺寸
    int startX = centerX - size / 2;
    int startY = centerY - size / 2;
    cv::Rect regionOfInterest(startX, startY, size, size);
    cv::Mat centerRegion = neibor.clone()(regionOfInterest);
    //计算平均灰度
    cv::Scalar meanValue = cv::mean(centerRegion);
    para.push_back(meanValue[0]);//放入vector中
    std::cout << "此时3*3邻域的平均灰度为：" << meanValue[0] << std::endl;

    //计算每一圈的灰度
    for (int i = 5; i <= neibor.cols; i += 2)
    {
        //提取出大的一圈
        startX = centerX - i / 2;
        startY = centerY - i / 2;
        cv::Rect regionOfInterest(startX, startY, i, i);
        cv::Mat centerRegionBig = neibor.clone()(regionOfInterest);

        // 中心小区域设置为0
        int centerStart = 1;
        cv::Rect centerRect(centerStart, centerStart, i - 2, i - 2);
        centerRegionBig(centerRect) = 0;

        //转化为浮点数矩阵
        centerRegionBig.convertTo(centerRegionBig, CV_64F);
        double sum = cv::sum(centerRegionBig)[0];

        double average = sum / (i * i - (i - 2) * (i - 2));
        std::cout << "此时" << i << "*" << i << "邻域周围一圈的灰度为：" << average << std::endl;
        para.push_back(average);
    }
    std::cout << std::endl;
}

// 回调函数，用于处理鼠标点击事件
void onMouseClick1(int event, int x, int y, int flags, void* userdata) {
    //当此时还没有选择ROI区域时
    if (select < 0)
    {
        //通过鼠标左键选择
        if (event == cv::EVENT_LBUTTONDOWN) { // 当检测到鼠标左键按下事件时

            //判断是否重复选择
            /*if (select>0)
            {
                std::cerr << "已选择区域，请按z退出后重选" << std::endl;
                std::exit(EXIT_FAILURE);
            }*/
            //select = select * (-1);
            cv::Mat* image = static_cast<cv::Mat*>(userdata); // 将userdata转换为cv::Mat指针
            std::cout << (*image).cols << std::endl;
            //若还没有选择放大区域
            if ((*image).cols > 100)
            {
                //保存要显示的ROI的参数
                centerRoi.x = x;
                centerRoi.y = y;

                selectRoi.x = x - 50;
                selectRoi.y = y - 50;


                // 将点击的点的坐标存储到容器中
                select = 1;

                cv::Mat show = (*image)(selectRoi);

                // 更新显示窗口中的图像
                cv::imshow("Display Window", show);
            }
    }
    }

    //当此时已经选择了ROI区域
    else if (select > 0)
    {
        if (event == cv::EVENT_LBUTTONDOWN)
        {
            std::cout << "************************************************************************" << std::endl;
            clickedPoints.push_back(cv::Rect(int(centerRoi.x + x - 50 - radius / 2), int(centerRoi.y + y - 50 - radius / 2), radius, radius));
            cv::Mat* image = static_cast<cv::Mat*>(userdata); // 将userdata转换为cv::Mat指针
            cv::Mat calImg = (*image).clone()
                (cv::Rect(int(centerRoi.x + x - 50 - radius / 2), int(centerRoi.y + y - 50 - radius / 2), radius, radius));
            cv::rectangle(*image, cv::Rect(int(centerRoi.x + x - 50 - radius / 2), int(centerRoi.y + y - 50 - radius / 2), radius, radius), cv::Scalar(0, 0, 255), 2);
            cv::Mat show = (*image)(selectRoi);
            //select = false;

            //计算想要的各项指标
            std::cout << "当前处理的点为:(" << centerRoi.x + x - radius / 2 << "," << centerRoi.y + y - radius / 2 << ")" << std::endl;
            std::cout << "邻域大小为:" << radius << std::endl;

            //把待计算的值转化为灰度图像
            if (calImg.channels() == 3)
            {
                cv::cvtColor(calImg, calImg, cv::COLOR_BGR2GRAY);
            }

            cv::Mat calImgGaussian = calImg.clone();

            // 计算均值和标准差
            cv::Scalar mean, stddev;
            cv::meanStdDev(calImg, mean, stddev);
            std::cout << "图像的均值为：" << mean[0] << std::endl;
            std::cout << "图像的标准差为：" << stddev[0] << std::endl;

            // 计算每一圈的均值
            std::vector<double> roundPara;//用于存放每一圈的灰度
            int flag1 = calNeibor(calImg, roundPara);

            //对其进行高斯拟合
            std::cout << "进行高斯拟合，请等待" << std::endl;
            //为LM算法提供初始值
            std::vector<double> para;
            para.push_back(100);
            para.push_back(1.5);
            para.push_back(1.5);

            //将矩阵变为浮点数矩阵，便于后续求导
            cv::Mat dstMat;
            calImgGaussian.convertTo(dstMat, CV_64F);
            //std::cout << dstMat.type()<<std::endl;

            double ems = LM(dstMat, para, 0.001, 0.0001, 100000);//开始LM拟合
            std::cout << "高斯拟合参数：" << std::endl;
            std::cout << "A: " << para[0] << std::endl;
            std::cout << "sigma_x: " << para[1] << std::endl;
            std::cout << "sigma_y: " << para[2] << std::endl;
            std::cout << "EMS:" << ems << std::endl;


            // 更新显示窗口中的图像
            cv::imshow("Display Window", show);
            std::cout << "\n" << std::endl;
        }
        if (event == cv::EVENT_RBUTTONDOWN)
        {
            select = -1;//切换到没有显示小图
        }
    }
}

// 回调函数，用于处理滑动条事件
void onTrackbar(int pos, void* userdata) {
    radius = 2*pos+1; // 更新滑动条的值
    smallRoi.width = radius * 4;
    smallRoi.height = radius * 4;
}

int main(int argc, char* argv[]) {
    // 读取图像
    //cv::Mat image =
    //    cv::imread("E:\\temporary_work\\20231228\\C09\\4k\\B_C09-202205051939_3CTL_0011_1580.000mm_R01.bmp",0); // 替换成你的图像路径
    cv::Mat image;
    if (argc <= 1)
    {
        std::cout << "请读入图片" << std::endl;
        image =
            cv::imread("E:\\temporary_work\\20231228\\C09\\4k\\B_C09-202205051939_3CTL_0011_1580.000mm_R01.bmp", 0);
        if (image.empty()) {
            //std::cout << "无法读取图像文件！" << std::endl;
            return -1;
        }
        //return -1;
    }
    else {
        image = cv::imread(argv[1], 0);
    }

    if (image.empty()) {
        std::cout << "无法读取图像文件！" << std::endl;
        return -1;
    }

    //对图像进行高斯与中值滤波
    if (image.channels() == 3)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    cv::GaussianBlur(image, image, cv::Size(3, 3), 1);
    cv::medianBlur(image, image, 3);
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    cv::Mat showRoi;//这个是展示的小图

    //给图像上写字
    double scale = image.cols / 2048;
    int lineWidth = image.cols / 4096 * 5;
    int pos = image.cols / 4096 * 75;
    cv::putText(image, "Please first select an area with the left mouse button.", cv::Point(0, pos), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Then use the left mouse button to select the pixel points in the neighborhood for analysis.", cv::Point(0, pos * 2), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Use the right mouse button to exit the small image mode and re-display the complete image.", cv::Point(0, pos * 3), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Press the 'd' key to undo the most recently drawn red rectangle.", cv::Point(0, pos * 4), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Press the 'Esc' key to exit the program.", cv::Point(0, pos * 5), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Press the 's' key to save the image after selecting the neiborhood.", cv::Point(0, pos * 6), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    
    
    cv::Mat tempImage = image.clone();//这个是用于修改的图
    // 创建显示窗口
    cv::namedWindow("Display Window", 0);
    cv::imshow("Display Window", tempImage);

    // 创建滑动条
    cv::createTrackbar("Radius", "Display Window", &radius, 10, onTrackbar);

    //开始显示
    while (true)
    {
        int key =cv::waitKey(1);
        cv::setMouseCallback("Display Window", onMouseClick1, &tempImage);
        // 如果按下 "d" 键，删除最后一个点
        if (key == 'd') {
            if (!clickedPoints.empty()) {
                tempImage = image.clone(); // 克隆图像以便修改

                // 删除最后一个点
                clickedPoints.pop_back();

                // 清空图像
                //tempImage.setTo(cv::Scalar(0, 0, 0)); // 将图像置为黑色

                // 重新绘制所有点
                for (const auto& rect : clickedPoints) {
                    cv::rectangle(tempImage, rect, cv::Scalar(0, 0, 255), 2);
                }

                // 更新显示窗口中的图像
                if(select>0)
                    cv::imshow("Display Window", tempImage(selectRoi));
                else
                    cv::imshow("Display Window", tempImage);
            }
            //select = -1;
        }

        //如果按下"z"键，则返回大图显示
        else if (select<0)
        {
            cv::imshow("Display Window", tempImage);
            select = -1;
        }

        //如果按下ESC则退出
        if (key == 27) { // 如果按下 ESC 键，退出循环
            break;
        }

        //如果按下s键则保存当前图像
        if (key == 's')
        {
            if (select < 0)
            {
                std::cout << "请选择要保存的局部图像" << std::endl;
            }
            else if (select > 0)
            {
                cv::Rect save_rect = selectRoi;
                save_rect.x = selectRoi.x - 100;
                save_rect.y = selectRoi.y - 100;
                save_rect.width = 300;
                save_rect.height = 300;
                std::string save_name = 
                    "(" + std::to_string(save_rect.x + 150) + "," + std::to_string(save_rect.y + 150) + ")" + ".tif";
                cv::Mat save_img = image.clone()(save_rect);
                cv::imwrite(save_name, save_img);
                std::cout << "图片已保存为:" << save_name << std::endl;
            }
        }
    }
}