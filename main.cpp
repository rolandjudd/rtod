#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>

int main() {

    // 0 is the id of the video device (webcam)
    cv::VideoCapture webcam = cv::VideoCapture(0);

    //check if video stream was opened successfully
    if (!webcam.isOpened()) {
        std::cerr << "ERROR: Cannot open video stream" << std::endl;
    }

    // Change resolution of webcam
    webcam.set(CV_CAP_PROP_FRAME_WIDTH,640);
    webcam.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    
    // Feature detection
    int hessian = 100;
    int octaves = 4;
    int octaveLayers = 2;
    bool extended = false;
    bool upright = true;
    cv::SurfFeatureDetector surf(hessian, octaves, octaveLayers, extended, upright);

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
    //cv::drawKeypoints(textbook, textbook_kp, textbook_out, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //cv::namedWindow("Textbook", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Textbook", textbook_out);

    // Open a window to display the webcam video
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE );
    
    // Loop until the user presses any key
    while (true) {

        // Get current frame
        cv::Mat frame;
        webcam.read(frame);

        // Keypoints and descriptors
        std::vector<cv::KeyPoint> kp;
        cv::Mat descriptors;
        
        surf.detect(frame, kp);
        extractor.compute(frame, kp, descriptors);
        
        cv::Mat out(frame);
        // cv::drawKeypoints(frame, kp, out, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        // Perform matching
        std::vector< std::vector< cv::DMatch > > matches;
        matcher.knnMatch(textbook_descriptors, descriptors, matches, 2);

        // Perform ratio test on matches
        std::vector< cv::DMatch > good_matches;

        float ratio = 0.75f;
        
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

        std::cout << "Textbook: " << good_matches.size() << " matching points" << std::endl;

        if(good_matches.size() > 50) {
        
            std::vector<cv::Point2f> textbook_pt;
            std::vector<cv::Point2f> frame_pt;

            for(int i = 0; i < good_matches.size(); i++) {
                textbook_pt.push_back(textbook_kp[good_matches[i].queryIdx].pt);
                frame_pt.push_back(kp[good_matches[i].trainIdx].pt);
            }
            
            cv::Mat h = cv::findHomography(textbook_pt, frame_pt, CV_RANSAC);

            // Get the corners of the object
            std::vector<cv::Point2f> textbook_corners(4);
            textbook_corners[0] = cv::Point2f(0,0);
            textbook_corners[1] = cv::Point2f(textbook.cols, 0);
            textbook_corners[2] = cv::Point2f(textbook.cols, textbook.rows);
            textbook_corners[3] = cv::Point2f(0, textbook.rows);

            // Transform the image using the homography
            std::vector<cv::Point2f> frame_corners(4);
            cv::perspectiveTransform(textbook_corners, frame_corners, h);
            
            // Draw lines around the object
            cv::Scalar color(255, 0, 0);
            cv::line(out, frame_corners[0], frame_corners[1], color, 2);
            cv::line(out, frame_corners[1], frame_corners[2], color, 2);
            cv::line(out, frame_corners[2], frame_corners[3], color, 2);
            cv::line(out, frame_corners[3], frame_corners[0], color, 2); 
        }
        
        cv::imshow("Webcam", out);
        if (cv::waitKey(30) >= 0)
            break;
    }
    
    return 0;
}
