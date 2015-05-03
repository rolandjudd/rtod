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

    // Feature detection
    int hessian = 1500;
    cv::SurfFeatureDetector surf(hessian);

    // Descriptor extractor
    cv::SurfDescriptorExtractor extractor;
    
    // Flann matcher
    cv::FlannBasedMatcher matcher;
    
    // Load the image of the textbook
    cv::Mat textbook = cv::imread("images/textbook.jpg");
    cv::Mat textbook_out;
    
    std::vector<cv::KeyPoint> textbook_kp;
    cv::Mat textbook_descriptors;
    surf.detect(textbook, textbook_kp);
    extractor.compute(textbook, textbook_kp, textbook_descriptors);
    cv::drawKeypoints(textbook, textbook_kp, textbook_out, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("Textbook", cv::WINDOW_AUTOSIZE );
    cv::imshow("Textbook", textbook_out);
    
    // Loop until the user presses any key
    while (true) {

        // Get current frame
        cv::Mat frame, out;
        webcam.read(frame);

        // Keypoints and descriptors
        std::vector<cv::KeyPoint> kp;
        cv::Mat descriptors;
        
        surf.detect(frame, kp);
        extractor.compute(frame, kp, descriptors);
        
        cv::drawKeypoints(frame, kp, out, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        // Perform matching
        std::vector< std::vector< cv::DMatch > > matches;
        matcher.knnMatch(textbook_descriptors, descriptors, matches, 2);

        // Perform ratio test on matches
        std::vector< cv::DMatch > good_matches;

        float ratio = 0.70f;
        
        for(int i = 0; i < matches.size(); i++)
        {
            if (matches[i].size() < 2) {
                continue;
            }
            
            const cv::DMatch &m1 = matches[i][0];
            const cv::DMatch &m2 = matches[i][1];
            
            if(m1.distance <= ratio * m2.distance) {
                good_matches.push_back(m1);
            }
        }

        std::cout << "Found " << good_matches.size() << " matching points" << std::endl;

        cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE );
        cv::imshow("Webcam", out);
        if (cv::waitKey(30) >= 0)
            break;
    }
    return 0;
}
