#include "stdio.h"
#include "iostream"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>

#include <ctime>

#include "my_image_utils.h"
#include "my_iris_segmentation.h"

#include "fit_curves.h"



using namespace cv;
using namespace std;

void removeSmallBlobs(cv::Mat& im, double size);
void cal_canny_high_thresd(Mat src, float ratio, int& high_thresd);

void main()
{
	Mat img = imread("C:\\Users\\hkpuadmin\\Desktop\\66_Jun_Liu_iris_15_04_22_04_49_49.jpg",1);

 	if (img.type() != CV_8UC1)
 	{
 		cvtColor(img,img,CV_BGR2GRAY);
 	}
	
	time_t cur_time, last_time;
	last_time = clock();

	//resize(img,img,Size(200,150));

	Mat img_enhanced;
	equalizeHist(img,img);

 	medianBlur(img,img_enhanced,7);

	int high_threshold = 0;
	int low_threshold = 0;

 	calc_threshold(img_enhanced,0,0.9,low_threshold,high_threshold);

	Mat bw_img, bw_big;
	threshold(img,bw_img,high_threshold,255,0);

	bw_big = bw_img.clone();

	removeSmallBlobs(bw_big,1000);

	for(int i = 0; i < bw_img.rows; i++)
		for(int j = 0; j < bw_img.cols; j++)
		{
			if (bw_big.at<unsigned char>(i,j) != 0)
			{
				bw_img.at<unsigned char>(i,j) = 0;
			}
		}

	int dilation_type = MORPH_ELLIPSE;
	int dilation_size = 15;

	Mat element = getStructuringElement( MORPH_ELLIPSE, Size(dilation_size,dilation_size));

  /// Apply the dilation operation
	Mat dil_img;
	dilate( bw_img, dil_img, element );

	GaussianBlur(img_enhanced,img_enhanced,Size(7,7),0);

	int canny_h_thresd = 0;
	cal_canny_high_thresd(img_enhanced,0.3,canny_h_thresd);

	int canny_l_threshold = canny_h_thresd*0.5;
	cout<<"1"<<endl;
	Mat canny_ret_img;
	Canny( img_enhanced, canny_ret_img, canny_l_threshold, canny_h_thresd);
//	Canny( img_enhanced, canny_ret_img, 50, 70);

	for(int i = 0; i < img_enhanced.rows; i++)
		for(int j = 0; j < img_enhanced.cols; j++)
		{
			if (dil_img.at<unsigned char>(i,j) != 0)
			{
				canny_ret_img.at<unsigned char>(i,j) = 0;
			}
		}

	//circle(img,Point(93,60),23,cv::Scalar(255));

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	Mat canny_ret_copy = canny_ret_img.clone();

	int points_gap = 10;
	int radiusRange[2] = {90, 140};
	int searchRange = 10;

	//int points_gap = 5;
	//int radiusRange[2] = {10, 33};
	//int searchRange = 5;

	Point center_point(0,0);

	findCenter(canny_ret_copy,radiusRange,points_gap,center_point);

	cout<<"center_point = "<<center_point<<endl;

	Point iris_center(0,0);
	int iris_rds = 0;
	 
	find_radius(canny_ret_copy,center_point,radiusRange, searchRange,iris_center,iris_rds);

	//circle(img,iris_center,iris_rds,cv::Scalar(255));

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	cout<<"iris_center = "<<iris_center<<endl;
	cout<<"iris_rds = "<< iris_rds<<endl;

	cur_time = clock();
	double time_cost = cur_time - last_time;
	cout<<"  time_cost = "<<time_cost<<endl;
	last_time = clock();

	Point left_top((iris_center.x - 1.2*iris_rds),(iris_center.y - 1.2*iris_rds));
	Point right_bottom((iris_center.x + 1.2*iris_rds),(iris_center.y + 1.2*iris_rds));
	Rect iris_area(left_top,right_bottom);
	Mat iris_img = img_enhanced(iris_area).clone();

	equalizeHist(iris_img,iris_img);
	GaussianBlur(iris_img,iris_img,Size(7,7),0);


	cal_canny_high_thresd(iris_img,0.4,canny_h_thresd);

	canny_l_threshold = canny_h_thresd*0.5;
	Canny( iris_img, canny_ret_img, canny_l_threshold, canny_h_thresd);

	for(int i = 0; i < iris_img.rows; i++)
		for(int j = 0; j < iris_img.cols; j++)
		{
			int distance = (i-iris_img.rows/2)*(i-iris_img.rows/2) + (j - iris_img.cols/2)*(j - iris_img.cols/2);
			if (abs(distance-iris_rds*iris_rds) < 0.2*0.2*iris_rds*iris_rds)
			{
				canny_ret_img.at<unsigned char>(i,j) = 0;
			}
		}

	Mat canny_show = canny_ret_img.clone();

	int pupil_radius_range[2] = {radiusRange[0]*0.1,radiusRange[1]*0.7};
	Point pupil_center(0,0);
	int pupil_rds = 0;
	
	int ret = 0;

	GaussianBlur(iris_img, iris_img, Size(5,5), 0 , 0);

	int i,j;

	/*****if eye image is too large, downsample eye image. the scale (times) is image's cols/200 *****/
	int times = iris_img.cols/200;
	if(times == 0)
	{
		times = 1;
	}

	/*****downsample eye image *****/
	int iMinR = pupil_radius_range[0]/times, iMaxR = pupil_radius_range[1]/times;

	Mat src_cut = Mat::zeros((iris_img.rows-iris_img.rows%times),(iris_img.cols-iris_img.cols%times),CV_8UC1);
	for(i = 0; i < src_cut.rows; i++)
		for(j = 0; j < src_cut.cols; j++)
		{
			src_cut.at<unsigned char>(i,j) = iris_img.at<unsigned char>(i,j);
			//src_cut.data[i*src_cut.cols + j] = iris_img.data[i*iris_img.cols + j];
		}

	Mat src_small = Mat::zeros(src_cut.rows/times,src_cut.cols/times, CV_8UC1);

	for(i = 0; i< src_small.rows; i++)
		for(j = 0; j < src_small.cols; j++)
		{
			src_small.data[i*src_small.cols + j] = src_cut.data[i*times*src_cut.cols +j*times];
		}

	GaussianBlur(src_small, src_small, Size(5,5), 0 , 0);

	pupil_coarse_location(src_small, iMinR, iMaxR, pupil_center, pupil_rds);
	//find_pupil_radius(canny_ret_img,Point(canny_ret_img.rows/2,canny_ret_img.cols/2), pupil_radius_range, 5, pupil_center, pupil_rds);
	 
	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  pupil_coarse_location time_cost = "<<time_cost<<endl;

	Point rst_iris_center = iris_center, rst_pupil_center((pupil_center.x+left_top.x),(pupil_center.y+left_top.y));
	int rst_iris_rds = iris_rds;
	int rst_pupil_rds = pupil_rds;

	circle(img,rst_iris_center,rst_iris_rds,cv::Scalar(255));
	circle(img,rst_pupil_center,rst_pupil_rds,cv::Scalar(255));

	cvNamedWindow( "better_result", 1 );
	imshow("better_result", img);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );

	Mat coffs_lower(3,1,CV_32FC1), coffs_upper(3,1,CV_32FC1);
	double range[2] = {0.4,1.1};
	fit_lower_eyelid(img_enhanced,rst_iris_center,rst_iris_rds,rst_pupil_center,rst_pupil_rds,range,coffs_lower);

	//cout<<"Befer print!"<<endl;
	//cout<<"coffs_lower = "<<coffs_lower.ptr<float>(0)[0]<<", "<<coffs_lower.ptr<float>(1)[0]<<", "<<coffs_lower.ptr<float>(2)[0]<<endl;
	//cout<<"After print!"<<endl;

	cout<<"After fit_lower_eyelid!"<<endl;
	fit_upper_eyelid(img_enhanced,rst_iris_center,rst_iris_rds,rst_pupil_center,rst_pupil_rds,range,0,false,coffs_upper);

	cout<<"Befer print!"<<endl;
	cout<<"coffs_upper = "<<coffs_upper.ptr<float>(0)[0]<<", "<<coffs_upper.ptr<float>(1)[0]<<", "<<coffs_upper.ptr<float>(2)[0]<<endl;
	cout<<"After print!"<<endl;

	//drawPolynomial(coffs_lower,1,img);
	//drawPolynomial(coffs_upper,2,img);
	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	Mat mask;
	double thresh[2] = {0.1,0.8};
	process_ES_region(img_enhanced, mask, rst_iris_center, rst_iris_rds, coffs_upper, thresh);

	cvNamedWindow( "better_result", 1 );
	imshow("better_result", mask);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );

	cout<<"After fit_upper_eyelid!"<<endl;

	cout<<"coffs_lower = "<<coffs_lower.ptr<float>(0)[0]<<", "<<coffs_lower.ptr<float>(1)[0]<<", "<<coffs_lower.ptr<float>(2)[0]<<endl;
	cout<<"coffs_upper = "<<coffs_upper.ptr<float>(0)[0]<<", "<<coffs_upper.ptr<float>(1)[0]<<", "<<coffs_upper.ptr<float>(2)[0]<<endl;
	//imwrite("iris_location.jpg",img);

	//Mat mask = Mat::zeros(img.size(),CV_8UC1);
	//double thresh_high[3] = {0,0,0};
	//double thresh_low[3] = {0,0,0};
	//double cir_correct = 0;
	//double extend[2] = {0.7, 1.3};

	//mask_lower_region(img_enhanced,iris_center,iris_rds,extend,mask,thresh_high,thresh_low,cir_correct);

	//cout<<"thresh_high = "<<thresh_high[0]<<thresh_high[1]<<thresh_high[2]<<endl;
	//cout<<"thresh_low = "<<thresh_low[0]<<thresh_low[1]<<thresh_low[2]<<endl;

	//circle(iris_img,pupil_center,pupil_rds,cv::Scalar(255));
	//circle(iris_img,Point(canny_ret_img.rows/2,canny_ret_img.cols/2),iris_rds,cv::Scalar(255));

	cvNamedWindow( "better_result", 1 );
	cvNamedWindow( "img", 1 );
	imshow("better_result", img);
	//imshow("img", mask);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );
	cvDestroyWindow( "img" );

	printf("Hello world!\n");

//	getchar();
}

void removeSmallBlobs(cv::Mat& im, double size)
{
	// Only accept CV_8UC1
	if (im.channels() != 1 || im.type() != CV_8U)
		return;

	// Find all contours
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(im.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		// Calculate contour area
		double area = cv::contourArea(contours[i]);

		// Remove small objects by drawing the contour with black color
		if (area >= 0 && area <= size)
			cv::drawContours(im, contours, i, CV_RGB(0, 0, 0), -1);
	}
}

void cal_canny_high_thresd(Mat src, float ratio, int& high_thresd)
{
	const int channels[1]={0};
    const int histSize[1]={256};
    float hranges[2]={0,255};
    const float* ranges[1]={hranges};
	Mat hist = Mat::zeros(256,1,CV_8UC1);
    calcHist(&src,1,channels,Mat(),hist,1,histSize,ranges);

	float num_thresd = src.cols*src.rows*ratio;
	cout<<"num_thresd = "<<num_thresd<<endl;
	
	int num_of_pixel = 0;
	for (int i = 0; i < 256; i++)
	{
		num_of_pixel += hist.at<float>(i,0);
		if (num_of_pixel > num_thresd)
		{
			high_thresd = i;
			break;
		}
		
	}
	cout<<"high_thresd = "<<high_thresd<<endl;
}