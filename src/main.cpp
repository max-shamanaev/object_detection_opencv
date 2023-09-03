#include "objectDetection.h"

int main()
{
    // ƒалее приводитс€ пример использовани€
    // класса ObjectDetection дл€ распознавани€
    // образов

    //  онфигурационные константы
    constexpr bool runOnGPU         { true };
    constexpr bool showFPS          { true };
    constexpr bool showDetectTime   { true };

    // »нициализаци€ объекта ObjectDetection,
    // в ходе которой загружаютс€ классы и сеть
    // по указанным пут€м.
    // 
    // cv::Size(640, 480) - размерность модели;
    // индивидуальна дл€ разных сетей
    od::ObjectDetection od{ "../models/coco.names", "../models/yolov8s.onnx",
                            cv::Size(640, 480), runOnGPU };

    // –аспознать образы можно как в видео-файле
    cv::VideoCapture inputVid{ "../sample_res/sample_vid.mp4" };

    // так и на изображени€х
    cv::VideoCapture inputPic{ "../sample_res/sample_pic3.jpg" };

    // так и в реальном времени, использу€,
    // например, веб-камеру устройства
    // (указываетс€ индекс устройства)
    cv::VideoCapture inputWebcam{ 1 };

    // Ќепосредственно сам процесс распознавани€.
    // ѕринудительно завершить можно хотке€ми q/ESC
    od.run(inputPic, showDetectTime, showFPS);

    return 0;
}