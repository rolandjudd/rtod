#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>

int main() {

    // 0 is the id of the video device (webcam)
    cv::VideoCapture webcam = cv::VideoCapture(0);

    //check if video stream was opened successfully
    if (!webcam.isOpened()) {
        std::cerr << "ERROR: Cannot open video stream" << std::endl;
    }

    // Change resolution of webcam
    webcam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    webcam.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    
    // SURF
    int minHessian = 25;
    cv::SurfFeatureDetector detector(minHessian);
    cv::SurfDescriptorExtractor extractor;
    
    // Matcher
    cv::FlannBasedMatcher matcher;
    
    // Load the image of the textbook
    cv::Mat textbook = cv::imread("images/textbook.jpg");
    cv::Mat textbook_gray;
    cv::cvtColor(textbook, textbook_gray, CV_BGR2GRAY); 
    cv::Mat textbook_out;
    
    std::vector<cv::KeyPoint> textbook_kp;
    cv::Mat textbook_descriptors;
    detector.detect(textbook, textbook_kp);
    extractor.compute(textbook, textbook_kp, textbook_descriptors);

    // Get the corners of the object
    std::vector<cv::Point2f> textbook_corners(4);
    textbook_corners[0] = cv::Point2f(1,1);
    textbook_corners[1] = cv::Point2f(textbook.cols - 1, 1);
    textbook_corners[2] = cv::Point2f(textbook.cols - 1, textbook.rows - 1);
    textbook_corners[3] = cv::Point2f(1, textbook.rows - 1);
    
    //cv::drawKeypoints(textbook, textbook_kp, textbook_out, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //cv::namedWindow("Textbook", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Textbook", textbook_out);

    // Open a window to display the webcam video
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE );

    // Store the webcam frames
    cv::Mat frame, gray, gray_prev;

    // Store current state of tracking
    bool tracking = false;
    std::vector<cv::Point2f> tracking_pts[3];
    int tracking_pts_count = 0;
    
    // Loop until the user presses any key
    while (true) {

        // Get current frame
        gray.copyTo(gray_prev);
        webcam.read(frame);
        cv::cvtColor(frame, gray, CV_BGR2GRAY); 

        cv::Mat out(frame);
        
        if(!tracking) {
            
            // Keypoints and descriptors
            std::vector<cv::KeyPoint> kp;
            cv::Mat descriptors;
        
            detector.detect(frame, kp);
            extractor.compute(frame, kp, descriptors);
            
            cv::drawKeypoints(frame, kp, out, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            
            // Perform matching
            std::vector< std::vector<cv::DMatch> > matches;
            matcher.knnMatch(textbook_descriptors, descriptors, matches, 2);
            
            // Perform ratio test on matches
            std::vector< cv::DMatch > good_matches;
            
            float ratio = 0.6f;
            
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
            
            //std::cout << "Textbook: " << good_matches.size() << " matching points" << std::endl;
            
            if(good_matches.size() > 25) {
                
                std::cout << "Textbook detected - tracking..." << std::endl;
                
                std::vector<cv::Point2f> textbook_pt;
                std::vector<cv::Point2f> frame_pt;
                
                for(int i = 0; i < good_matches.size(); i++) {
                    textbook_pt.push_back(textbook_kp[good_matches[i].queryIdx].pt);
                    frame_pt.push_back(kp[good_matches[i].trainIdx].pt);
                }
                
                cv::goodFeaturesToTrack(textbook_gray, tracking_pts[0], 200, 0.01, 10, cv::Mat(), 3, 0, 0.04);
                tracking_pts[0].insert(tracking_pts[0].end(), textbook_corners.begin(), textbook_corners.end());  
                cv::Mat h = cv::findHomography(textbook_pt, frame_pt, CV_RANSAC, 10);

                tracking_pts_count = tracking_pts[0].size();
                
                // Transform the tracking points using the homography
                cv::perspectiveTransform(tracking_pts[0], tracking_pts[1], h);
                
                tracking = true;
            }

        }
        else{
            
            std::vector<uchar> status;
            std::vector<float> err;
            cv::Size window(25, 25);
            cv::calcOpticalFlowPyrLK(gray_prev, gray, tracking_pts[1], tracking_pts[2], status, err, window, 3);
            
            // Delete points from tracking for which the flow cannot be calculated
            int deleted = 0;
            for(int i = 0; i < status.size(); i++) {
                
                if(status[i] != 1 || err[i] > 10) {
                    tracking_pts[0].erase(tracking_pts[0].begin() + i - deleted);
                    tracking_pts[2].erase(tracking_pts[2].begin() + i - deleted);
                    deleted++;
                }
            }
            
            if(tracking_pts[0].size() < tracking_pts_count * 0.5) {
                std::cout << "Textbook lost -  resuming detection..." << std::endl;
                tracking = false;
            }

            else{

                cv::Mat h = cv::findHomography(tracking_pts[0], tracking_pts[2], CV_RANSAC, 10);
            
                // Transform the image using the homography
                std::vector<cv::Point2f> frame_corners(4);
                cv::perspectiveTransform(textbook_corners, frame_corners, h);
                
                // Draw lines around the object
                cv::Scalar color(255, 0, 0);
                cv::line(out, frame_corners[0], frame_corners[1], color, 2);
                cv::line(out, frame_corners[1], frame_corners[2], color, 2);
                cv::line(out, frame_corners[2], frame_corners[3], color, 2);
                cv::line(out, frame_corners[3], frame_corners[0], color, 2); 
                
                for(int i = 0; i < tracking_pts[2].size(); i++) {
                    cv::circle(out, tracking_pts[2][i], 5, cv::Scalar(0, 0, 255), CV_FILLED, 8, 0);
                }
                tracking_pts[1] = tracking_pts[2];
            }
        }
        
        cv::imshow("Webcam", out);
        if (cv::waitKey(5) >= 0) {
            std::cout << "Key pressed - Exiting" << std::endl;
            break;
        }
    }
    
    return 0;
}
