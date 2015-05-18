#ifndef FIT_CURVES_H
#define FIT_CURVES_H

#include <opencv2\core\core.hpp>
#include <vector>

using namespace cv;
using namespace std;

int fitPolynomial(vector<Point2f> pts, int nd, Mat &dst, Mat &pic = Mat());
int drawPolynomial(Mat coef, int type, Mat &pic);
int drawOverlapped(Mat coef1, Mat coef2, Mat &pic);

#endif