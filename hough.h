#ifndef HOUGH_H
#define HOUGH_H

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>

using namespace cv;
using namespace std;
struct location
{
	int x;
	int y;
};
void Hough(Mat img, location& l, int radius);

#endif // !HOUGH_H