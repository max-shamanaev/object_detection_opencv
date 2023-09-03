#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <vector>
#include <string>
#include <string_view>

namespace od
{
    struct Detection
    {
        int classID{ 0 };
        std::string className{};
        float confidence{ 0.0f };
        cv::Scalar color{};
        cv::Rect box{};
    };

    struct ModelThresholds
    {
        float confidenceThreshold  { 0.25f };
        float scoreThreshold       { 0.45f };
        float NMSThreshold         { 0.50f };
    };

    class ObjectDetection
    {
    public:
        ObjectDetection(std::string_view classesPath, std::string_view modelPath,
                        const cv::Size& modelInputShape = { 640, 640 },  bool runWithCuda = false);

        void run(cv::VideoCapture& cap, bool showTimeProfile = false, bool showFPS = false);
        void detect(const cv::Mat& input);

    private:
        void loadClassesFromFile();
        void loadNet();

        cv::Mat formatToSquare(const cv::Mat& source);

    private:
        std::string m_classesPath{};
        std::vector<std::string> m_classes{};

        std::string m_modelPath{};
        cv::Size2f m_modelShape{};
        ModelThresholds m_thresholds{};

        bool m_cudaEnabled{ false };

        cv::dnn::Net m_net{};
    };
}


