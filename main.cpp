#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

int main() {

    // 0 is the id of the video device (webcam)
    cv::VideoCapture webcam = cv::VideoCapture(0);

    //check if video stream was opened successfully
    if (!webcam.isOpened()) {
        std::cerr << "ERROR: Cannot open video stream" << std::endl;
    }
    
    // Loop until the user presses any key
    while (true) {
        cv::Mat cameraFrame;
        webcam.read(cameraFrame);
        cv::imshow("Webcam", cameraFrame);
        if (cv::waitKey(30) >= 0)
            break;
    }
    return 0;
}
