#include "objectDetection.h"

#include <fstream>
#include <iostream>
#include <map>
#include <random>

namespace od
{
    ObjectDetection::ObjectDetection(std::string_view classesPath, std::string_view modelPath,
        const cv::Size& modelInputShape, bool runWithCuda)
        : m_classesPath { classesPath }
        , m_modelPath   { modelPath }
        , m_modelShape  { modelInputShape }
        , m_cudaEnabled { runWithCuda }
    {
        loadClassesFromFile(); 
        loadNet();
    }

    void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size modelSize)
    {
        cv::Mat blob{};

        // Создание 4D-blob-а из кадра
        if (modelSize.width <= 0) modelSize.width = frame.cols;
        if (modelSize.height <= 0) modelSize.height = frame.rows;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, modelSize, cv::Scalar(), true, false);
        
        net.setInput(blob);
    }

    void postprocess(std::vector<cv::Mat>& outs, cv::Size2f resizeFactors, const std::vector<std::string>& classes,
        const ModelThresholds& thresholds, std::vector<Detection>& outDetections)
    {
        CV_Assert(outs.size() > 0);

        int rows{ outs[0].size[1] };
        int dimensions{ outs[0].size[2] };

        // На данный момент поддерживается только модель Yolov8,
        // имеющая выход вида (batchSize, 84,  8400) (Num classes + box[x,y,w,h]).
        // Так что следующий блок Yolov8-specific и корректность его работы не
        // гарантируется для других моделей
        if (dimensions > rows)
        {
            rows = outs[0].size[2];
            dimensions = outs[0].size[1];

            outs[0] = outs[0].reshape(1, dimensions);
            cv::transpose(outs[0], outs[0]);
        }
        float* data{ (float*)outs[0].data };

        // Векторы для хранения значений во время развернывания детектов
        std::vector<int> class_ids{};
        std::vector<float> confidences{};
        std::vector<cv::Rect> boxes{};

        // Итерирование по всем детектам
        for (size_t i{ 0 }; i < rows; ++i)
        {
            float* classesScores{ data + 4 };

            // Создание 1х85 матрицы с оценками по 80 классам 
            cv::Mat scores(1, classes.size(), CV_32FC1, classesScores);

            // Вычисление индекса класса с лучшей оценкой 
            cv::Point class_id{};
            double maxClassScore{};
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > thresholds.scoreThreshold)
            {
                // Если он подходит - сохраняем id класса и уверенность 
                // предсказания в определенных выше векторах
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                // Центр
                float cx{ data[0] };
                float cy{ data[1] };

                // Размерность бокса
                float w{ data[2] };
                float h{ data[3] };

                // Координаты рамки бокса
                int left    { int((cx - 0.5 * w) * resizeFactors.width) };
                int top     { int((cy - 0.5 * h) * resizeFactors.height) };
                int width   { int(w * resizeFactors.width) };
                int height  { int(h * resizeFactors.height) };

                // Сохраняем боксы хороших детектов
                boxes.push_back(cv::Rect(left, top, width, height));
            }

            // Переход к следующему столбцу
            data += dimensions;
        }

        // Non Maximum Supression
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, thresholds.scoreThreshold, thresholds.NMSThreshold, nmsResult);

        // Заполнение выходного массива найденными и обработанными детектами
        for (auto res : nmsResult)
        {
            Detection result{};
            result.classID = class_ids[res];
            result.confidence = confidences[res];

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            result.color = cv::Scalar(dis(gen),
                dis(gen),
                dis(gen));

            result.className = classes[result.classID];
            result.box = boxes[res];

            outDetections.push_back(result);
        }
    }

    void drawLabel(cv::Mat& frame, const Detection& detection)
    {
        auto& dBox{ detection.box };
        auto& dColor{ detection.color };

        // Прямоугольник обнаруженного объекта
        cv::rectangle(frame, dBox, dColor, 2);

        // Ярлык обнаруженного объекта
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(dBox.x, dBox.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(frame, textBox, dColor, cv::FILLED);
        cv::putText(frame, classString, cv::Point(dBox.x + 5, dBox.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }

    void ObjectDetection::detect(const cv::Mat& input)
    {
        cv::Mat modelInput{ input };
        if (m_modelShape.width == m_modelShape.height)
            modelInput = formatToSquare(modelInput);        

        // Подготовка входных данных (кадра) для модели
        preprocess(modelInput, m_net, m_modelShape);

        // Forward propagate модели
        std::vector<cv::Mat> outputs{};
        m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());

        // Множители для ресайза
        float xfactor{ modelInput.cols / m_modelShape.width };
        float yfactor{ modelInput.rows / m_modelShape.height };
        cv::Size2f resize{ xfactor, yfactor };

        // Распознавание объектов
        std::vector<Detection> detections{};
        postprocess(outputs, resize, m_classes, m_thresholds, detections);

        std::cout << "Objects detected: " << detections.size() << '\n';

        for (const auto& detection : detections)
            drawLabel(modelInput, detection);
    }

    void ObjectDetection::loadClassesFromFile()
    {
        std::ifstream inputFile{ m_classesPath };
        if (inputFile.is_open())
        {
            std::string classLine{};
            while (std::getline(inputFile, classLine))
                m_classes.push_back(classLine);

            inputFile.close();
        }
    }

    void ObjectDetection::loadNet()
    {
        m_net = cv::dnn::readNet(m_modelPath);

        std::cout << "\nCUDA: ";
        if (m_cudaEnabled)
        {
            std::cout << "enabled" << std::endl;
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        else
        {
            std::cout << "disabled" << std::endl;
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }

    cv::Mat ObjectDetection::formatToSquare(const cv::Mat& source)
    {
        int col{ source.cols };
        int row{ source.rows };
        int max{ MAX(col, row) };

        cv::Mat result{ cv::Mat::zeros(max, max, CV_8UC3) };
        source.copyTo(result(cv::Rect(0, 0, col, row)));

        return result;
    }
}