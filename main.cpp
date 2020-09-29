#include <opencv2/videoio.hpp>
#include "opencv2/imgcodecs.hpp"
#include "cannyEdgeDetector.h"

using namespace cv;
using namespace std;

void show(string str, Mat img) {
	img.convertTo(img, CV_8U);
	imshow(str, img);
}

int main(int, char**) {
	Mat frame;
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID, apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		<< "Press any key to terminate" << endl;

	//vector<vector<double>> kernel = Gaussian(1, 3);
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		resize(frame, frame, Size(frame.cols/2,frame.rows/2));
		// check if we succeeded
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		CannyEdgeDetector canny = CannyEdgeDetector();
		frame = canny.detect(frame, 5);
		show("Live", frame);
		if (waitKey(5) >= 0)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}