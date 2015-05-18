#ifndef MY_IMAGE_UTILS_H
#define MY_IMAGE_UTILS_H

#include <opencv2\core\core.hpp>
#include <opencv2/gpu/gpu.hpp>

#define INF 0.0000001
#define PI 3.141592654

using namespace std;
using namespace cv;

int SSREnhancement(const cv::gpu::GpuMat src, Mat &dst);

int SSREnhancement(Mat src, Mat &dst);

int StructureMap(const char *subjName, Mat src, float lmbdax, float threshx, Mat &dstx);

int convolute(Mat src, Mat filter, Mat &dst);

int scale255(Mat src, Mat &dst);

int thin(Mat src, Mat &dst);
int thin2(Mat src, Mat &dst);

int calc_threshold(Mat src, float low_percents, float high_percents, int &low_thresh, int &high_thresh);

int reflection_removal(Mat src, Mat &mask, Mat &dest);

int resize_small(Mat src, Mat &dst);

int LeastSquares(vector<cv::Point> points, cv::Point& center, int& rds);

int interp2(Mat src, double* xo, double* yo, int width, int height, Mat& dst);

void findCenter(Mat edgemap, int* radiusRange, int points_gap, Point &center_point);

void find_radius(Mat edgemap, Point center_point, int* radiusRange, int searchRange, Point& output_center, int& output_radius);

void find_pupil_radius(Mat edgemap, Point center_point, int* radiusRange, int searchRange, Point& output_center, int& output_radius);

void mask_lower_region(Mat img, Point center, int radius, double* extend, Mat& mask, double* thresh_high, double* thresh_low, double& cir_correct);

void mask_upper_region(Mat img, Mat& mask, Point center, int radius, double* thresh);

void thresh_angle_range(Mat img, Point center, double* radius_range, double* angles, double& thresh, double& thresh_low, double& quality);

void get_sector_region(Point center, double* radii, double* angles, int* size, Mat& region);

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);

void cal_hist_thresh( Mat masked_img, double& thresh_high, double& thresh_low, double& quality);

void segment_angle_range(Mat img, Mat& mask, Point center, int radius, double* angles, double thresh, double* radius_range);

void get_pupil_region( Mat img, Mat reflection, Point center, int radius, double* thresh);

int fit_lower_eyelid( Mat img, Point iris_center, int iris_rds, Point pupil_center, int pupil_rds, double* range, Mat& coffs);

int fit_upper_eyelid( Mat& img, Point iris_center, int iris_rds, Point pupil_center, int pupil_rds, double* range, double offset, bool save_image, Mat& coffs);

int process_ES_region( Mat img, Mat& mask, Point center, int rds, Mat coffs, double* thresh);

void soble_double_direction(Mat img, Mat& result);

#endif