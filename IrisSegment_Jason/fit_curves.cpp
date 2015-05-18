#include "fit_curves.h"
#include <cmath>

int fitPolynomial(vector<Point2f> pts, int nd, Mat &dst, Mat &pic)
{
	
	int np = pts.size();

	if (np < nd)
	{
		return -1;
	}

	Mat A(np, nd + 1, CV_32FC1);
	Mat b(np, 1, CV_32FC1);

//	dst = Mat(nd + 1, 1, CV_32FC1);

	for (int i = 0; i < np; i++)
	{
		for (int d = 0; d <= nd; d++)
		{
			A.at<float>(i, d) = static_cast<float>(pow(pts[i].x, nd - d));
		}
		//A.at<float>(0, i) = pts[i].x * pts[i].x;
		//A.at<float>(1, i) = pts[i].x;
		//A.at<float>(2, i) = 1;

		b.at<float>(i, 0) = static_cast<float>(pts[i].y);
	}

	int ret = solve(A, b, dst, DECOMP_SVD);
	if (ret == 1)
	{
		ret = 0;
	}
	else
	{
		ret = -2;
	}

	if (!pic.empty())
	{
		for (int i = 0; i < np; i++)
		{
			circle(pic, pts[i], 1, Scalar(255), 1);
		}
	}
    return ret;	
}

int drawPolynomial(Mat coef, int type, Mat &pic)
{
	if (pic.empty())
	{
		return 0;
	}

	int xmax = pic.cols;
	int ymax = pic.rows;
	for (int x = 0; x < xmax; x++)
	{
		Mat X(coef.rows, 1, CV_32FC1);
		//X.at<float>(0, 0) = x * x;
		//X.at<float>(0, 1) = x;
		//X.at<float>(0, 2) = 1;
		for (int j = 0; j < X.rows; j++)
		{
			X.at<float>(j, 0) = static_cast<float>(pow(x, (X.rows - 1) - j));
		}

		int y = cvRound(coef.dot(X));

		if (type == 0)
		{
			if (y < 0 || y >= ymax)
			{
				continue;
			}
			pic.at<unsigned char>(y, x) = 0;
		}
		else if (type == 1)
		{
			y = max(0, y);

			for (int j = y; j < ymax; j++)
			{
				pic.at<unsigned char>(j, x) = 0;
			}
		}
		else if (type == 2)
		{
			y = min(ymax - 1, y);

			for (int j = 0; j < y; j++)
			{
				pic.at<unsigned char>(j, x) = 0;
			}
		}
	}

	return 0;
}

int drawOverlapped(Mat coef1, Mat coef2, Mat &pic)
{
	if (pic.empty() || coef1.rows != coef2.rows)
	{
		return 0;
	}

	int xmax = pic.cols;
	int ymax = pic.rows;
	for (int x = 0; x < xmax; x++)
	{
		Mat X(coef1.rows, 1, CV_32FC1);

		for (int j = 0; j < X.rows; j++)
		{
			X.at<float>(j, 0) = static_cast<float>(pow(x, (X.rows - 1) - j));
		}

		int y1 = cvRound(coef1.dot(X));
		int y2 = cvRound(coef2.dot(X));

		y1 = max(0, y1);
		y2 = min(ymax - 1, y2);

		for (int y = y1; y <= y2; y++)
		{
			pic.at<unsigned char>(y, x) = 255;
		}
	}

	return 0;
}