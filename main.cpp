#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
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

        // Get current frame
        cv::Mat frame, out;
        webcam.read(frame);

        // Vector of keypoints
        std::vector<cv::KeyPoint> kp;

        int hessian = 1500;
        cv::SurfFeatureDetector surf(hessian);
        surf.detect(frame, kp);
        cv::drawKeypoints(frame, kp, out, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        cv::imshow("Webcam with keypoints", out);
        if (cv::waitKey(30) >= 0)
            break;
    }
    return 0;
}
