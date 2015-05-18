#ifndef MY_IRIS_SEGMENTATION_H
#define MY_IRIS_SEGMENTATION_H

#include <opencv2\core\core.hpp>

#define IRIS_RG_PI 3.14159265

using namespace cv;

int iris_segment(const char *subjName, Mat src, Mat &dst);

int iris_segment(const char *subjName, Mat src, Mat &mask1, Mat &mask2);

//////Add by Huan
int RadialSymmetryTran(Mat src, int iMinR, int iMaxR, Point &center, int &rds);

//////Add by Huan
int FindIrisCircle(Mat src, cv::Point pupil_center, int pupil_rds, cv::Point& iris_center, int& iris_rds);

//////Add by Huan
int FineLocatePupil(Mat src, int times, Point& pupil_center, int& pupil_rds);

//////Add by Huan
int CircleDifference(Mat src, int min_rds, int max_rds, int search_Dis, Point &center, Point &best_center, int &best_rds);

//////Add by Huan
int LocatePupilandIris(Mat src, int times, int min_rds,int max_rds,cv::Point& pupil_center, int& pupil_rds, cv::Point& iris_center, int& iris_rds);

//////Add by Huan
int iris_segment_new(const char *subjName, Mat src, Mat &dst);

int pupil_coarse_location(Mat& src, int iMinR, int iMaxR, Point &pupil_center, int &pupil_rds);

#endif