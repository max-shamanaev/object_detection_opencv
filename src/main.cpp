#include "objectDetection.h"

int main()
{
    // ����� ���������� ������ �������������
    // ������ ObjectDetection ��� �������������
    // �������

    // ���������������� ���������
    constexpr bool runOnGPU         { true };
    constexpr bool showFPS          { true };
    constexpr bool showDetectTime   { true };

    // ������������� ������� ObjectDetection,
    // � ���� ������� ����������� ������ � ����
    // �� ��������� �����.
    // 
    // cv::Size(640, 480) - ����������� ������;
    // ������������� ��� ������ �����
    od::ObjectDetection od{ "../models/coco.names", "../models/yolov8s.onnx",
                            cv::Size(640, 480), runOnGPU };

    // ���������� ������ ����� ��� � �����-�����
    cv::VideoCapture inputVid{ "../sample_res/sample_vid.mp4" };

    // ��� � �� ������������
    cv::VideoCapture inputPic{ "../sample_res/sample_pic3.jpg" };

    // ��� � � �������� �������, ���������,
    // ��������, ���-������ ����������
    // (����������� ������ ����������)
    cv::VideoCapture inputWebcam{ 1 };

    // ��������������� ��� ������� �������������.
    // ������������� ��������� ����� �������� q/ESC
    od.run(inputPic, showDetectTime, showFPS);

    return 0;
}