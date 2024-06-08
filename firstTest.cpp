#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Structure to store tracked objects
struct TrackedObject {
    Rect2d boundingBox;
    Point2f center;
    bool counted;
    Point2f lastPosition;
    double speed;
};

// Function to calculate Euclidean distance between two points
double calculateDistance(Point2f p1, Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Function to convert pixel distance to real-world distance
double pixelToRealWorldDistance(double pixelDistance, double pixelPerMeter) {
    return pixelDistance / pixelPerMeter;
}

int main() {
    // Open the video file
    VideoCapture cap("C:/Users/frass/OneDrive/Documents/Coding/OpenCV Project/TrafficTracker/TrafficFootage.mp4");
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file!" << endl;
        return -1;
    }

    // Create Background Subtractor object
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();

    Mat frame, fgMask;
    int carCount = 0;
    int nextID = 0;

    // Vector to store tracked objects
    vector<TrackedObject> trackedObjects;

    // Assuming a frame rate of 30 FPS (adjust based on your video)
    double frameRate = 30.0;
    double pixelPerMeter = 10.0; // Adjust this based on your calibration

    while (true) {
        // Read the next frame
        cap >> frame;
        if (frame.empty()) break;

        // Apply background subtraction
        pBackSub->apply(frame, fgMask);

        // Morphological operations to reduce noise
        erode(fgMask, fgMask, Mat());
        dilate(fgMask, fgMask, Mat());

        // Find contours in the foreground mask
        vector<vector<Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Vector to store current bounding boxes
        vector<TrackedObject> currentTrackedObjects;

        // Filter out small contours and draw bounding boxes around detected cars
        for (size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) < 500) continue; // Adjust the threshold based on your video

            Rect2d boundingBox = boundingRect(contours[i]);
            Point2f center = Point2f(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            rectangle(frame, boundingBox, Scalar(0, 255, 0), 2);

            bool newCar = true;
            // Check if the detected car is already tracked
            for (auto &obj : trackedObjects) {
                if (calculateDistance(obj.center, center) < 50) { // Adjust distance threshold based on your needs
                    double pixelDistance = calculateDistance(obj.lastPosition, center);
                    double realWorldDistance = pixelToRealWorldDistance(pixelDistance, pixelPerMeter);
                    double speed = realWorldDistance * frameRate; // Speed in meters per second

                    obj.boundingBox = boundingBox;
                    obj.center = center;
                    obj.lastPosition = center;
                    obj.speed = speed;
                    currentTrackedObjects.push_back(obj);
                    newCar = false;
                    break;
                }
            }

            // If it's a new car, add it to the tracked objects
            if (newCar) {
                TrackedObject newObj = {boundingBox, center, false, center, 0.0};
                currentTrackedObjects.push_back(newObj);
            }
        }

        // Check for cars moving in the specific direction and count them
        for (auto &obj : currentTrackedObjects) {
            if (!obj.counted) {
                for (const auto &prevObj : trackedObjects) {
                    if (calculateDistance(obj.center, prevObj.center) < 50 && obj.center.x > prevObj.center.x) {
                        obj.counted = true;
                        carCount++;
                        break;
                    }
                }
            }
        }

        // Update tracked objects
        trackedObjects = currentTrackedObjects;

        // Display the frame with bounding boxes, car count, and speed
        putText(frame, "Car Count: " + to_string(carCount), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        for (const auto &obj : trackedObjects) {
            putText(frame, "Speed: " + to_string(obj.speed) + " m/s", obj.center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
        }
        imshow("Car Tracking", frame);

        // Break the loop on 'q' key press
        if (waitKey(30) == 'q') break;
    }

    cout << "Total cars counted: " << carCount << endl;

    return 0;
}
