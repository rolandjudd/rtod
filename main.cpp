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

// Global path for image loads; reconfigure to whatever directory you choose
std::string base_path = "images/";

// Target class to store data for object needing detection
class Target {
    public:
        std::string name;
        cv::Mat image;
        cv::Mat gray;
        std::vector<cv::KeyPoint> kp;
        cv::Mat descriptors;
        std::vector<cv::Point2f> corners;
        bool detected;
        std::vector<cv::Point2f> points;
        std::vector<cv::Point2f> points_current;
        std::vector<cv::Point2f> points_previous;
        int points_count;

    public:
        Target(std::string);
        void get_keypoints(cv::SurfFeatureDetector detector);
        void get_descriptors(cv::SurfDescriptorExtractor extractor);
		void detect(cv::Mat frame_descriptors, std::vector< cv::KeyPoint > frame_kp, cv::FlannBasedMatcher matcher);
		void track(cv::Mat frame_gray, cv::Mat frame_gray_prev);
};

// Constructor for Target class initializes image, name, gray, corners

Target::Target(std::string target_name) {

    name = target_name;
    std::string path = base_path + name + ".jpg";
    image = cv::imread(path);

    if(!image.data) {
        std::cerr << "Failed to load " << path << std::endl;
        exit(1);
    }

    detected = false;
    
    cv::cvtColor(image, gray, CV_BGR2GRAY);
    corners.push_back(cv::Point2f(1,1));
    corners.push_back(cv::Point2f(image.cols - 1, 1));
    corners.push_back(cv::Point2f(image.cols - 1, image.rows - 1));
    corners.push_back(cv::Point2f(1, image.rows - 1));
}

// get_keypoints and get_descriptors sets the Target's keypoints and descriptors 
// based on provided detector and extractor functions

void Target::get_keypoints(cv::SurfFeatureDetector detector) {
    detector.detect(image, kp);
}

void Target::get_descriptors(cv::SurfDescriptorExtractor extractor) {
    extractor.compute(image, kp, descriptors);
}

// Simple function to initialize a webcam given its id
cv::VideoCapture init_webcam(int id) {
    // 0 is the id of the video device (webcam)
    cv::VideoCapture webcam = cv::VideoCapture(id);

    //check if video stream was opened successfully
    if (!webcam.isOpened()) {
        std::cerr << "ERROR: Cannot open video stream" << std::endl;
    }

    // Change resolution of webcam
    webcam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    webcam.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    return webcam;
}

void Target::detect(cv::Mat frame_descriptors, std::vector< cv::KeyPoint > frame_kp, cv::FlannBasedMatcher matcher) {

    // Perform matching
    std::vector< std::vector<cv::DMatch> > matches;
    matcher.knnMatch(descriptors, frame_descriptors, matches, 2);
    
    // Perform ratio test on matches
    std::vector<cv::DMatch> good_matches;
    
    float ratio = 0.6f;
    
    for(int j = 0; j < matches.size(); j++)
    {
        if (matches[j].size() < 2) {
            continue;
        }
        
        const cv::DMatch &m1 = matches[j][0];
        const cv::DMatch &m2 = matches[j][1];
        
        if(m1.distance <= ratio * m2.distance) {
            good_matches.push_back(m1);
        }
    }
    
    std::cout << name << ": " << good_matches.size() << " matching points" << std::endl;
    
    if(good_matches.size() < 7) {
    	return;
    }
                            
    std::vector<cv::Point2f> target_pt;
    std::vector<cv::Point2f> frame_pt;
    
    for(int j = 0; j < good_matches.size(); j++) {
        target_pt.push_back(kp[good_matches[j].queryIdx].pt);
        frame_pt.push_back(frame_kp[good_matches[j].trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat h = cv::findHomography(target_pt, frame_pt, CV_RANSAC, 10, mask);

    float in = 0.0f;
    float total = (float)target_pt.size();
    for(int j = 0; j < target_pt.size(); j++){
        if(mask.at<double>(j) == 0){
            in++;
        }
    }

    float fraction = in / total;    
    std::cout << fraction << " fraction inliers" << std::endl;

    if(fraction > 0.1) {
    
        std::cout << name << " detected - tracking..." << std::endl;
        
        cv::goodFeaturesToTrack(gray, points, 25, 0.01, 10, cv::Mat(), 3, 0, 0.04);
        
        points_count = points.size();
        
        // Transform the tracking points using the homography
        cv::perspectiveTransform(points, points_previous, h);
        
        detected = true;
    }
    
    return;
}

void Target::track(cv::Mat frame_gray, cv::Mat frame_gray_prev) {

    std::vector<uchar> status;
    std::vector<float> err;
    cv::Size window(41, 41);
    cv::calcOpticalFlowPyrLK(frame_gray_prev, frame_gray, points_previous, points_current, status, err, window, 4);
    
    // Delete points from tracking for which the flow cannot be calculated
    int deleted = 0;
    for(int i = 0; i < status.size(); i++) {
        
        if(status[i] != 1 || err[i] > 10 ) {
            points.erase(points.begin() + i - deleted);
            points_current.erase(points_current.begin() + i - deleted);
            deleted++;
        }
    }
    
    if(points.size() < points_count * 0.25) {
        std::cout << name << " lost -  resuming detection..." << std::endl;
        detected = false;
    }

    else{

        cv::Mat h = cv::findHomography(points, points_current, CV_RANSAC, 10);
        
        // Transform the image using the homography
        std::vector<cv::Point2f> frame_corners(4);
        cv::perspectiveTransform(corners, frame_corners, h);
        
        // Draw lines around the object
        cv::Scalar color(255, 0, 0);
        cv::line(out, frame_corners[0], frame_corners[1], color, 2);
        cv::line(out, frame_corners[1], frame_corners[2], color, 2);
        cv::line(out, frame_corners[2], frame_corners[3], color, 2);
        cv::line(out, frame_corners[3], frame_corners[0], color, 2); 
        
        for(int i = 0; i < points_current.size(); i++) {
            cv::circle(out, points_current[i], 5, cv::Scalar(0, 0, 255), CV_FILLED, 8, 0);
        }
        points_previous = points_current;
    }
}

int main(int argc, char* argv[]) {
    
    std::vector<Target> targets;

    // Initialize webcam
    cv::VideoCapture webcam = init_webcam(0);

    // Initialize SURF
    cv::SurfFeatureDetector detector(400);
    cv::SurfDescriptorExtractor extractor;

    // Initialize Matcher
    cv::FlannBasedMatcher matcher;

    // Iterate through images provided as arguments to create new targets.
    for (int i = 1; i < argc; i++) {
        Target loaded = Target(std::string(argv[i]));
        loaded.get_keypoints(detector);
        loaded.get_descriptors(extractor);
        targets.push_back(loaded);
        
        cv::Mat out;
        cv::drawKeypoints(loaded.image, loaded.kp, out, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("opencv", cv::WINDOW_AUTOSIZE);
        cv::imshow(loaded.name, out);
    }
    
    // Open a window to display the webcam video
    cv::namedWindow("opencv", cv::WINDOW_AUTOSIZE );

    // Store the webcam frames
    cv::Mat frame, gray, gray_prev;

    // Loop until the user presses any key
    while (true) {

        // Get current frame
        gray.copyTo(gray_prev);
        webcam.read(frame);
        cv::cvtColor(frame, gray, CV_BGR2GRAY); 

        cv::Mat out(frame);
                    
        // Keypoints and descriptors
        std::vector<cv::KeyPoint> kp;
        cv::Mat descriptors;
    
        detector.detect(frame, kp);
        extractor.compute(frame, kp, descriptors);
            
        //cv::drawKeypoints(frame, kp, out, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            
        for (int i = 0; i < targets.size(); i++) {

            Target &target = targets[i];
            
            if(target.detected == false) {
            	target.detect(descriptors, kp, matcher);
            }
            
	        else {
	        	target.track(gray_prev, gray)
	        }
	    }
        cv::imshow("Webcam", out);
        if (cv::waitKey(5) >= 0) {
            std::cout << "Key pressed" << std::endl;
            //break;
        }
    }
    
    return 0;
}
