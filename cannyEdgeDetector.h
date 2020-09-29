#ifndef CANNYEDGEDETECTOR_H
#define CANNYEDGEDETECTOR_H

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
//#include <map>

using namespace cv;
using namespace std;

class CannyEdgeDetector {
private:
	Mat convolve2D(Mat img, vector<vector<double>>& kernel);

	void fast_sobel(Mat img, Mat& G, Mat& grad);

	vector<vector<double>> convolve2D(vector<vector<double>> img, vector<vector<double>>& kernel);

	vector<vector<double>> GaussianKernel(double sigma = 1, int size = 3);

	vector<int> boxesForGauss(float sigma, int n);

	void boxBlurH_4(Mat& scl, Mat& tcl, int w, int h, int r);

	void boxBlurT_4(Mat& scl, Mat& tcl, int w, int h, int r);

	void boxBlur_4(Mat& scl, Mat& tcl, int w, int h, int r);

	void gaussBlur_4(Mat& scl, Mat& tcl, int w, int h, int r);

	Mat LowPass(Mat source, int w, int h, int radius);

	void sobel(Mat& img, Mat& magnitude, Mat& gradient);

	Mat nms(Mat mag, Mat grad);

	Mat hysteresis(Mat img, double lowRatio, double highRatio);

public:
	/**
		* @brief Detect edges.
		blah blah blah
			* @param image: Source image.
			* @param kernel: Size of Gaussian blur filter
			* @param lowRatio: Lower bound for hysteresis
			* @param highRatio: Upper bound for hysteresis
			* @return image with edges.
	*/
	Mat detect(Mat image, int kernel = 3, float lowRatio = 0.5, float highRatio = 0.2);
};

#endif // !CANNYEDGEDETECTOR_H