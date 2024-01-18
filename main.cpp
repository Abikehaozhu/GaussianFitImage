#include <opencv2/opencv.hpp>
#include <vector>
#include"GaussianFit.h"

// ȫ�ֱ������ڴ洢����������
std::vector<cv::Rect> clickedPoints;

// ȫ�ֱ������ڴ洢��������ֵ
int radius = 2;

//������ʾСROI��Rect
cv::Rect smallRoi(0, 0, 2 * radius, 2 * radius);
cv::Rect selectRoi(0, 0, 100, 100);//��һ��ѡ�������
cv::Point centerRoi(0, 0);//��ǰչʾ���������
int select = -1;//��ǰ�Ƿ��Ѿ�ѡ�������

//���ڼ���3*3����ĻҶ�ֵ��ÿһȦ�ĻҶ�ƽ��ֵ
int calNeibor(cv::Mat neibor, std::vector<double>& para)
{
    //���ͼ���С�򲻽��д���
    if (neibor.cols < 3)
    {
        std::cout << "�����С" << std::endl;
        return 0;
    }

    //ת����ͨ��
    if (neibor.channels() == 3)
    {
        cv::cvtColor(neibor, neibor, cv::COLOR_BGR2GRAY);
    }

    //���ȼ���3*3�ĻҶ�ֵ
    // ��ȡͼ�����������
    int centerX = neibor.cols / 2;
    int centerY = neibor.rows / 2;
    // ��ȡ����3x3����
    int size = 3; // ����ߴ�
    int startX = centerX - size / 2;
    int startY = centerY - size / 2;
    cv::Rect regionOfInterest(startX, startY, size, size);
    cv::Mat centerRegion = neibor.clone()(regionOfInterest);
    //����ƽ���Ҷ�
    cv::Scalar meanValue = cv::mean(centerRegion);
    para.push_back(meanValue[0]);//����vector��
    std::cout << "��ʱ3*3�����ƽ���Ҷ�Ϊ��" << meanValue[0] << std::endl;

    //����ÿһȦ�ĻҶ�
    for (int i = 5; i <= neibor.cols; i += 2)
    {
        //��ȡ�����һȦ
        startX = centerX - i / 2;
        startY = centerY - i / 2;
        cv::Rect regionOfInterest(startX, startY, i, i);
        cv::Mat centerRegionBig = neibor.clone()(regionOfInterest);

        // ����С��������Ϊ0
        int centerStart = 1;
        cv::Rect centerRect(centerStart, centerStart, i - 2, i - 2);
        centerRegionBig(centerRect) = 0;

        //ת��Ϊ����������
        centerRegionBig.convertTo(centerRegionBig, CV_64F);
        double sum = cv::sum(centerRegionBig)[0];

        double average = sum / (i * i - (i - 2) * (i - 2));
        std::cout << "��ʱ" << i << "*" << i << "������ΧһȦ�ĻҶ�Ϊ��" << average << std::endl;
        para.push_back(average);
    }
    std::cout << std::endl;
}

// �ص����������ڴ���������¼�
void onMouseClick1(int event, int x, int y, int flags, void* userdata) {
    //����ʱ��û��ѡ��ROI����ʱ
    if (select < 0)
    {
        //ͨ��������ѡ��
        if (event == cv::EVENT_LBUTTONDOWN) { // ����⵽�����������¼�ʱ

            //�ж��Ƿ��ظ�ѡ��
            /*if (select>0)
            {
                std::cerr << "��ѡ�������밴z�˳�����ѡ" << std::endl;
                std::exit(EXIT_FAILURE);
            }*/
            //select = select * (-1);
            cv::Mat* image = static_cast<cv::Mat*>(userdata); // ��userdataת��Ϊcv::Matָ��
            std::cout << (*image).cols << std::endl;
            //����û��ѡ��Ŵ�����
            if ((*image).cols > 100)
            {
                //����Ҫ��ʾ��ROI�Ĳ���
                centerRoi.x = x;
                centerRoi.y = y;

                selectRoi.x = x - 50;
                selectRoi.y = y - 50;


                // ������ĵ������洢��������
                select = 1;

                cv::Mat show = (*image)(selectRoi);

                // ������ʾ�����е�ͼ��
                cv::imshow("Display Window", show);
            }
    }
    }

    //����ʱ�Ѿ�ѡ����ROI����
    else if (select > 0)
    {
        if (event == cv::EVENT_LBUTTONDOWN)
        {
            std::cout << "************************************************************************" << std::endl;
            clickedPoints.push_back(cv::Rect(int(centerRoi.x + x - 50 - radius / 2), int(centerRoi.y + y - 50 - radius / 2), radius, radius));
            cv::Mat* image = static_cast<cv::Mat*>(userdata); // ��userdataת��Ϊcv::Matָ��
            cv::Mat calImg = (*image).clone()
                (cv::Rect(int(centerRoi.x + x - 50 - radius / 2), int(centerRoi.y + y - 50 - radius / 2), radius, radius));
            cv::rectangle(*image, cv::Rect(int(centerRoi.x + x - 50 - radius / 2), int(centerRoi.y + y - 50 - radius / 2), radius, radius), cv::Scalar(0, 0, 255), 2);
            cv::Mat show = (*image)(selectRoi);
            //select = false;

            //������Ҫ�ĸ���ָ��
            std::cout << "��ǰ����ĵ�Ϊ:(" << centerRoi.x + x - radius / 2 << "," << centerRoi.y + y - radius / 2 << ")" << std::endl;
            std::cout << "�����СΪ:" << radius << std::endl;

            //�Ѵ������ֵת��Ϊ�Ҷ�ͼ��
            if (calImg.channels() == 3)
            {
                cv::cvtColor(calImg, calImg, cv::COLOR_BGR2GRAY);
            }

            cv::Mat calImgGaussian = calImg.clone();

            // �����ֵ�ͱ�׼��
            cv::Scalar mean, stddev;
            cv::meanStdDev(calImg, mean, stddev);
            std::cout << "ͼ��ľ�ֵΪ��" << mean[0] << std::endl;
            std::cout << "ͼ��ı�׼��Ϊ��" << stddev[0] << std::endl;

            // ����ÿһȦ�ľ�ֵ
            std::vector<double> roundPara;//���ڴ��ÿһȦ�ĻҶ�
            int flag1 = calNeibor(calImg, roundPara);

            //������и�˹���
            std::cout << "���и�˹��ϣ���ȴ�" << std::endl;
            //ΪLM�㷨�ṩ��ʼֵ
            std::vector<double> para;
            para.push_back(100);
            para.push_back(1.5);
            para.push_back(1.5);

            //�������Ϊ���������󣬱��ں�����
            cv::Mat dstMat;
            calImgGaussian.convertTo(dstMat, CV_64F);
            //std::cout << dstMat.type()<<std::endl;

            double ems = LM(dstMat, para, 0.001, 0.0001, 100000);//��ʼLM���
            std::cout << "��˹��ϲ�����" << std::endl;
            std::cout << "A: " << para[0] << std::endl;
            std::cout << "sigma_x: " << para[1] << std::endl;
            std::cout << "sigma_y: " << para[2] << std::endl;
            std::cout << "EMS:" << ems << std::endl;


            // ������ʾ�����е�ͼ��
            cv::imshow("Display Window", show);
            std::cout << "\n" << std::endl;
        }
        if (event == cv::EVENT_RBUTTONDOWN)
        {
            select = -1;//�л���û����ʾСͼ
        }
    }
}

// �ص����������ڴ��������¼�
void onTrackbar(int pos, void* userdata) {
    radius = 2*pos+1; // ���»�������ֵ
    smallRoi.width = radius * 4;
    smallRoi.height = radius * 4;
}

int main(int argc, char* argv[]) {
    // ��ȡͼ��
    //cv::Mat image =
    //    cv::imread("E:\\temporary_work\\20231228\\C09\\4k\\B_C09-202205051939_3CTL_0011_1580.000mm_R01.bmp",0); // �滻�����ͼ��·��
    cv::Mat image;
    if (argc <= 1)
    {
        std::cout << "�����ͼƬ" << std::endl;
        image =
            cv::imread("E:\\temporary_work\\20231228\\C09\\4k\\B_C09-202205051939_3CTL_0011_1580.000mm_R01.bmp", 0);
        if (image.empty()) {
            //std::cout << "�޷���ȡͼ���ļ���" << std::endl;
            return -1;
        }
        //return -1;
    }
    else {
        image = cv::imread(argv[1], 0);
    }

    if (image.empty()) {
        std::cout << "�޷���ȡͼ���ļ���" << std::endl;
        return -1;
    }

    //��ͼ����и�˹����ֵ�˲�
    if (image.channels() == 3)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    cv::GaussianBlur(image, image, cv::Size(3, 3), 1);
    cv::medianBlur(image, image, 3);
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    cv::Mat showRoi;//�����չʾ��Сͼ

    //��ͼ����д��
    double scale = image.cols / 2048;
    int lineWidth = image.cols / 4096 * 5;
    int pos = image.cols / 4096 * 75;
    cv::putText(image, "Please first select an area with the left mouse button.", cv::Point(0, pos), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Then use the left mouse button to select the pixel points in the neighborhood for analysis.", cv::Point(0, pos * 2), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Use the right mouse button to exit the small image mode and re-display the complete image.", cv::Point(0, pos * 3), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Press the 'd' key to undo the most recently drawn red rectangle.", cv::Point(0, pos * 4), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Press the 'Esc' key to exit the program.", cv::Point(0, pos * 5), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    cv::putText(image, "Press the 's' key to save the image after selecting the neiborhood.", cv::Point(0, pos * 6), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 0), lineWidth);
    
    
    cv::Mat tempImage = image.clone();//����������޸ĵ�ͼ
    // ������ʾ����
    cv::namedWindow("Display Window", 0);
    cv::imshow("Display Window", tempImage);

    // ����������
    cv::createTrackbar("Radius", "Display Window", &radius, 10, onTrackbar);

    //��ʼ��ʾ
    while (true)
    {
        int key =cv::waitKey(1);
        cv::setMouseCallback("Display Window", onMouseClick1, &tempImage);
        // ������� "d" ����ɾ�����һ����
        if (key == 'd') {
            if (!clickedPoints.empty()) {
                tempImage = image.clone(); // ��¡ͼ���Ա��޸�

                // ɾ�����һ����
                clickedPoints.pop_back();

                // ���ͼ��
                //tempImage.setTo(cv::Scalar(0, 0, 0)); // ��ͼ����Ϊ��ɫ

                // ���»������е�
                for (const auto& rect : clickedPoints) {
                    cv::rectangle(tempImage, rect, cv::Scalar(0, 0, 255), 2);
                }

                // ������ʾ�����е�ͼ��
                if(select>0)
                    cv::imshow("Display Window", tempImage(selectRoi));
                else
                    cv::imshow("Display Window", tempImage);
            }
            //select = -1;
        }

        //�������"z"�����򷵻ش�ͼ��ʾ
        else if (select<0)
        {
            cv::imshow("Display Window", tempImage);
            select = -1;
        }

        //�������ESC���˳�
        if (key == 27) { // ������� ESC �����˳�ѭ��
            break;
        }

        //�������s���򱣴浱ǰͼ��
        if (key == 's')
        {
            if (select < 0)
            {
                std::cout << "��ѡ��Ҫ����ľֲ�ͼ��" << std::endl;
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
                std::cout << "ͼƬ�ѱ���Ϊ:" << save_name << std::endl;
            }
        }
    }
}