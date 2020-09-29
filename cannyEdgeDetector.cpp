#include "cannyEdgeDetector.h"
using namespace cv;
using namespace std;

Mat CannyEdgeDetector::convolve2D(Mat img, vector<vector<double>>& kernel) {
	int krows = kernel.size(), kcols = kernel[0].size();
	int irows = img.rows, icols = img.cols;
	//Mat img_pad(img.rows + krows, img.cols + kcols, 0.0f);
	vector<vector<double>>  img_pad(irows + krows, vector<double>(icols + kcols, 0));

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img_pad[i + krows / 2][j + kcols / 2] = img.at<double>(i, j);
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			double val = 0;
			for (int kx = 0; kx < krows; kx++) {
				for (int ky = 0; ky < kcols; ky++) {
					val += img_pad[i + kx][j + ky] * kernel[kx][ky];
				}
			}
			img.at<double>(i, j) = val;
		}
	}
	return img;
}

void CannyEdgeDetector::fast_sobel(Mat img, Mat& G, Mat& grad) {
	//int krows = 1, kcols = kernelX[0].size();
	int irows = img.rows, icols = img.cols;
	/*Mat G(irows + 3, icols + 3, 0.0f);
	Mat grad(irows + 3, icols + 3, 0.0f);*/
	//vector<vector<double>>  img_pad(irows + 3, vector<double>(icols + 3, 0));
	double maxMagnitude = 0;
	for (int i = 1; i < irows - 1; i++) {
		for (int j = 1; j < icols - 1; j++) {
			//if (i < 1 || j < 1 || (i + 1) == irows || (j + 1) == icols) continue;
			double gx = -img.at<double>(i - 1, j - 1) +
				img.at<double>(i + 1, j - 1) +
				img.at<double>(i - 1, j) * (-2.0f) +
				img.at<double>(i + 1, j) * (2.0f) -
				img.at<double>(i - 1, j + 1) +
				img.at<double>(i + 1, j + 1);

			double gy = -img.at<double>(i - 1, j - 1) +
				img.at<double>(i, j - 1) * (-2.0f) -
				img.at<double>(i + 1, j - 1) +
				img.at<double>(i - 1, j + 1) +
				img.at<double>(i, j + 1) * (2.0f) +
				img.at<double>(i + 1, j + 1);
			G.at<double>(i, j) = min(255.0f, max(0.0f, sqrtf(gx * gx + gy * gy)));
			grad.at<double>(i, j) = atan2(gy, gx);
			maxMagnitude = max(maxMagnitude, G.at<double>(i, j));
		}
	}

	for (int i = 0; i < irows; i++) {
		for (int j = 0; j < icols; j++) {
			G.at<double>(i, j) *= 255 / maxMagnitude;
		}
	}
}

vector<vector<double>>  CannyEdgeDetector::convolve2D(vector<vector<double>> img, vector<vector<double>>& kernel) {
	int krows = kernel.size(), kcols = kernel[0].size();
	int irows = img.size(), icols = img[0].size();
	vector<vector<double>>  img_pad(irows + krows, vector<double>(icols + kcols, 0));
	for (int i = 0; i < irows; i++) {
		for (int j = 0; j < icols; j++) {
			img_pad[i + krows / 2][j + kcols / 2] = img[i][j];
		}
	}

	for (int i = 0; i < irows; i++) {
		for (int j = 0; j < icols; j++) {
			double val = 0;
			for (int kx = 0; kx < krows; kx++) {
				for (int ky = 0; ky < kcols; ky++) {
					val += img_pad[i + kx][j + ky] * kernel[kx][ky];
				}
			}
			img[i][j] = val;
		}
	}
	return img;
}

vector<vector<double>> CannyEdgeDetector::GaussianKernel(double sigma, int size) {
	vector<vector<double>> kernel(size, vector<double>(size, 0));
	size = size / 2;
	for (int x = -size; x <= size; x++) {
		for (int y = -size; y <= size; y++) {
			double val = exp(-(x * x + y * y) / (2 * sigma * sigma)) / double(2 * M_PI * sigma * sigma);
			kernel[x + size][y + size] = val;
		}
	}
	return kernel;
}

vector<int> CannyEdgeDetector::boxesForGauss(float sigma, int n)  // standard deviation, number of boxes
{
	auto wIdeal = sqrt((12 * sigma * sigma / n) + 1);  // Ideal averaging filter width 
	int wl = floor(wIdeal);
	if (wl % 2 == 0)
		wl--;
	int wu = wl + 2;

	auto mIdeal = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4);
	int m = round(mIdeal);
	// var sigmaActual = Math.sqrt( (m*wl*wl + (n-m)*wu*wu - n)/12 );

	vector<int> sizes(n);
	for (auto i = 0; i < n; i++)
		sizes[i] = i < m ? wl : wu;
	return sizes;
}

void CannyEdgeDetector::boxBlurH_4(Mat& scl, Mat& tcl, int w, int h, int r) {
	float iarr = 1.f / (r + r + 1);
	for (auto i = 0; i < h; i++) {
		auto ti = i * w;
		auto li = ti, ri = ti + r;
		//auto fv = scl[ti], lv = scl[ti + w - 1];
		auto fv = scl.at<double>(ti / w, ti % w);
		auto lv = scl.at<double>(ti / w, w - 1);

		auto val = (r + 1) * fv;
		for (auto j = 0; j < r; j++) val += scl.at<double>((ti + j) / w, (ti + j) % w);
		for (auto j = 0; j <= r; j++) { val += scl.at<double>(ri / w, ri % w) - fv; tcl.at<double>(ti / w, ti % w) = round(val * iarr); ri++, ti++; }
		for (auto j = r + 1; j < w - r; j++) { val += scl.at<double>(ri / w, ri % w) - scl.at<double>(li / w, li % w);   tcl.at<double>(ti / w, ti % w) = round(val * iarr); ri++, li++, ti++; }
		for (auto j = w - r; j < w; j++) { val += lv - scl.at<double>(li / w, li % w);   tcl.at<double>(ti / w, ti % w) = round(val * iarr); li++, ti++; }
	}
}

void CannyEdgeDetector::boxBlurT_4(Mat& scl, Mat& tcl, int w, int h, int r) {
	float iarr = 1.f / (r + r + 1);
	for (auto i = 0; i < w; i++) {
		auto ti = i, li = ti, ri = ti + r * w;
		auto fv = scl.at<double>(ti / w, ti % w), lv = scl.at<double>((ti + w * (h - 1)) / w, (ti + w * (h - 1)) % w);
		auto val = (r + 1) * fv;
		for (auto j = 0; j < r; j++) val += scl.at<double>((ti + j * w) / w, (ti + j * w) % w);
		for (auto j = 0; j <= r; j++) { val += scl.at<double>(ri / w, ri % w) - fv;  tcl.at<double>(ti / w, ti % w) = round(val * iarr);  ri += w; ti += w; }
		for (auto j = r + 1; j < h - r; j++) { val += scl.at<double>(ri / w, ri % w) - scl.at<double>(li / w, li % w);  tcl.at<double>(ti / w, ti % w) = round(val * iarr);  li += w; ri += w; ti += w; }
		for (auto j = h - r; j < h; j++) { val += lv - scl.at<double>(li / w, li % w);  tcl.at<double>(ti / w, ti % w) = round(val * iarr);  li += w; ti += w; }
	}
}

void CannyEdgeDetector::boxBlur_4(Mat& scl, Mat& tcl, int w, int h, int r) {
	for (auto i = 0; i < scl.rows; i++)
		for (auto j = 0; j < scl.cols; j++)
			tcl.at<double>(i, j) = scl.at<double>(i, j);
	boxBlurH_4(tcl, scl, w, h, r);
	boxBlurT_4(scl, tcl, w, h, r);
}

void CannyEdgeDetector::gaussBlur_4(Mat& scl, Mat& tcl, int w, int h, int r) {
	auto bxs = boxesForGauss(r, 3);
	boxBlur_4(scl, tcl, w, h, (bxs[0] - 1) / 2);
	boxBlur_4(tcl, scl, w, h, (bxs[1] - 1) / 2);
	boxBlur_4(scl, tcl, w, h, (bxs[2] - 1) / 2);
}

Mat CannyEdgeDetector::LowPass(Mat source, int w, int h, int radius)
{
	Mat lowpass = source.clone(); // copy constructor
	Mat target(source.rows, source.cols, CV_64FC1, 0.0f);
	gaussBlur_4(lowpass, target, w, h, radius);

	return target;
}

void CannyEdgeDetector::sobel(Mat& img, Mat& magnitude, Mat& gradient) {
	vector<vector<double>> Kx = { {-1, 0, 1},{-2, 0, 2}, {-1, 0, 1} };
	vector<vector<double>> Ky = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1 } };
	Mat Ix = convolve2D(img.clone(), Kx);
	Mat Iy = convolve2D(img.clone(), Ky);
	double maxMagnitude = 0;
	for (int i = 0; i < Ix.rows; i++) {
		for (int j = 0; j < Ix.cols; j++) {
			magnitude.at<double>(i, j) = sqrt(Ix.at<double>(i, j) * Ix.at<double>(i, j) + Iy.at<double>(i, j) * Iy.at<double>(i, j));

			gradient.at<double>(i, j) = atan2(Iy.at<double>(i, j), Ix.at<double>(i, j)) * 180 / M_PI;
		}
	}

	for (int i = 0; i < Ix.rows; i++) {
		for (int j = 0; j < Ix.cols; j++) {
			maxMagnitude = max(maxMagnitude, magnitude.at<double>(i, j));
		}
	}
	for (int i = 0; i < Ix.rows; i++) {
		for (int j = 0; j < Ix.cols; j++) {
			magnitude.at<double>(i, j) *= 255 / maxMagnitude;
		}
	}
}

Mat CannyEdgeDetector::nms(Mat magnitude, Mat gradient) {
	Mat mag = magnitude.clone(), grad = gradient.clone();
	Mat Z(mag.rows, mag.cols, CV_64FC1);
	for (int i = 2; i < mag.rows - 2; i++) {
		for (int j = 2; j < mag.cols - 2; j++) {
			double q = 0, r = 0;
			if (grad.at<double>(i, j) < 0) grad.at<double>(i, j) += 180;
			double thisAngle = grad.at<double>(i, j);
			try {
				if (((thisAngle < 22.5) && (thisAngle > -22.5)) || (thisAngle > 157.5) && (thisAngle < 180)) {
					q = mag.at<double>(i + 1, j - 1);
					r = mag.at<double>(i - 1, j + 1);
				}
				if ((thisAngle >= 22.5) && (thisAngle < 67.5)) {
					q = mag.at<double>(i + 1, j);
					r = mag.at<double>(i - 1, j);
				}
				if (((thisAngle >= 112.5) && (thisAngle < 157.5))) {
					q = mag.at<double>(i - 1, j - 1);
					r = mag.at<double>(i + 1, j + 1);
				}
				if ((mag.at<double>(i, j) >= q) && (mag.at<double>(i, j) >= r)) Z.at<double>(i, j) = mag.at<double>(i, j);
				else Z.at<double>(i, j) = 0;
			}
			catch (...) {

			}

		}
	}
	return Z;
}

Mat CannyEdgeDetector::hysteresis(Mat image, double lowRatio, double highRatio) {
	assert(highRatio != 0);
	Mat img = image.clone();
	int M = img.rows, N = img.cols;
	double highThreshold = 255 * highRatio;
	double lowThreshold = highThreshold * lowRatio;
	double weak = 75;
	double strong = 255;
	Mat res(M, N, CV_64FC1, 0.0f);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (img.at<double>(i, j) >= highThreshold) { res.at<double>(i, j) = 255; }
			else if (img.at<double>(i, j) < lowThreshold) { res.at<double>(i, j) = 0; }
			else { res.at<double>(i, j) = weak; }

		}
	}
	for (int i = 3; i < M - 3; i++) {
		for (int j = 3; j < N - 3; j++) {
			try {
				if ((res.at<double>(i + 1, j - 1) == strong) || (res.at<double>(i + 1, j) == strong) || (res.at<double>(i + 1, j + 1) == strong)
					|| (res.at<double>(i, j - 1) == strong) || (res.at<double>(i, j + 1) == strong) || (res.at<double>(i - 1, j - 1) == strong)
					|| (res.at<double>(i - 1, j) == strong) || (res.at<double>(i - 1, j + 1) == strong))
				{
					img.at<double>(i, j) = strong;
				}
				else img.at<double>(i, j) = 0;
			}
			catch (...) {}
		}
	}
	return img;
}

Mat CannyEdgeDetector::detect(Mat image, int kernel, float lowRatio, float highRatio) {
	assert(kernel != 0); //check kernel size is not 0
	assert(highRatio != 0); //check highratio is not 0
	Mat img = image.clone(); //clone image to new variable to avoid changing original input
	if (image.channels() > 1) cvtColor(image, image, COLOR_BGR2GRAY); //convert image to grayscale
	
	image.convertTo(image, CV_64FC1); //change format from unsigned int to double for future calculations
	Mat smooth_image = LowPass(image, image.cols, image.rows, kernel); //apply approximate gaussian blur
	/*Mat smooth_image = convolve2D(image.clone(), kernel);*/ //apply true gaussian blur
	Mat magnitude(image.rows + 3, image.cols + 3, CV_64FC1, 0.0f); 
	Mat gradient(image.rows + 3, image.cols + 3, CV_64FC1, 0.0f);
	fast_sobel(smooth_image, magnitude, gradient); //apply fast sobel filter on blurred image and return magnitude and gradient
	Mat Z = nms(magnitude, gradient); // apply non-max suppression to reduce thickness of edges
	Mat res = hysteresis(Z, lowRatio, highRatio); //make edges coontinuous
	return res;
}



