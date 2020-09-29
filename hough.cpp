#include "hough.h"
using namespace cv;
void Hough(Mat img, location& l, int radius) {
	int irows = img.rows, icols = img.cols;
	vector<vector<int>> accumulator(irows, vector<int>(icols));
	vector<double>theta;
	for (int i = 0; i < 360; i++) {
		theta.push_back(i * M_PI / 180);
	}
	vector<vector<int>> bprint(2 * (radius + 1), vector<int>(2 * (radius + 1), 0));
	for (int row = 0; row < irows; row++) {
		for (int col = 0; col < icols; col++) {
			if (img.at<double>(row, col) == 255) {
				for (auto angle : theta) {
					int x = (int)radius * cos(angle);
					int y = (int)radius * sin(angle);
					bprint[x + radius][y + radius] ++;
				}
			}
		}
	}
	int temp = 0;
	for (int i = 20; i < bprint.size() - 3; i++)
		for (int j = 20; j < bprint[0].size() - 3; j++) {
			//cout << bprint[i][j] << "\t" << i<< " "<< j;
			if (temp < bprint[i][j])
			{
				temp = bprint[i][j];
				l.x = i + 1;
				l.y = j + 1;
			}
			cout << endl;
		}
	cout << "max  " << bprint[l.x][l.y] << "  " << l.x << "  " << l.y << "  " << endl;
}