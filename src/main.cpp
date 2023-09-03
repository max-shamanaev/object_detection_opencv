#include "objectDetection.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

using namespace cv;
using namespace od;

int main()
{
    bool runOnGPU = true;

    ObjectDetection od("../models/coco.names", "../models/yolov8s.onnx", cv::Size(640, 480), runOnGPU);

    std::vector<std::string> imageNames;
    imageNames.push_back("../sample_res/sample_pic3.jpg");

    cv::VideoCapture cap("../sample_res/sample_vid.mp4");
    //cv::VideoCapture cap(1);
    cv::Mat frame;

    //for (auto& img : imageNames)
    while (cap.read(frame))
    {
        //frame = cv::imread(img);

        // Inference starts here
        auto startTime = getTickCount();

        od.detect(frame);

        // Inference ends here...
        auto endTime = getTickCount();
        auto timeElapsed = (endTime - startTime) / getTickFrequency();

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = od.getNet().getPerfProfile(layersTimes) / freq;
        std::string labelTime = format("Inference time: %.2f ms", t);
        std::string labelFps = "FPS: " + std::to_string(int(1 / timeElapsed));
        cv::putText(frame, labelTime, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        cv::putText(frame, labelFps, Point(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));


        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
        cv::imshow("Inference", frame);

        char c = cv::waitKey(1);
        if (c == 113 || c == 27) //'q' or ESC
        {
            cap.release();
            cv::destroyAllWindows();
            break;
        }
    }

    return 0;
}