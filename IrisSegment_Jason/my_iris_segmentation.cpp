
#include "my_iris_segmentation.h"
#include "my_image_utils.h"
#include "fit_curves.h"
// #include "stasm_lib.h"

//////Edit by Huan
// #include <QDateTime>

#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\photo\photo.hpp>
#include <ctime>

int irisPupilCircleDetect2(const char *subjName, Mat eye, Mat enhanced_eye, Mat struct_map,
										   int minPupilRds , int maxPupilRds,
										   Point &iris_center, int &iris_rds, 
										   Point &pupil_center, int &pupil_rds)
{
	if (enhanced_eye.empty() || struct_map.empty())
	{
		return -1;
	}

	Mat copy_struct_map = struct_map.clone();
	vector<vector<Point> > contours;

	findContours(copy_struct_map, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	//Mat para_space = struct_map.clone();

	int possible_center_xmin = eye.cols / 3, possible_center_xmax = eye.cols * 2 / 3;
	int possible_center_ymin = eye.rows / 3, possible_center_ymax = eye.rows * 2 / 3;

	minPupilRds = max(15, minPupilRds);
	maxPupilRds = min(45, maxPupilRds);

	Mat para_space = Mat::zeros(eye.rows, eye.cols, CV_8UC1);


	int step = 20 | 1; //must be odd
	for (auto contour : contours)
	{
		if (contour.size() < step)
		{
			continue;
		}

		for (int i = 0; i + step < contour.size(); i++)
		{
			vector<Point2f> pts;
			Point p1 = contour[i], p2 = contour[i + step - 1];
			Point cross = Point(cvRound((p1.x + p2.x) / 2.0), cvRound((p1.y + p2.y) / 2.0));
			
			
			float sin, cos;
			if (p1.x == p2.x)
			{
				sin = 0.0;
				cos = 1.0;
			}
			else if (p1.y == p2.y)
			{
				sin = 1.0;
				cos = 0.0;
			}
			else
			{
				float normal_a = static_cast<float>(p1.y - p2.y) / (p1.x - p2.x);
				normal_a = -1 / normal_a;

				float d = sqrt(1 + normal_a * normal_a);
				
				sin = normal_a / d;
				cos = 1 / d;
				if (normal_a < 0)
				{
					cos = -cos;
					sin = -sin;
				}
			}

			
			for (int ri = minPupilRds; ri < maxPupilRds; ri++)
			{
				
				int x = 0, y = 0;
					
				x = cvRound(cross.x - ri * cos);
				y = cvRound(cross.y - ri * sin);


				if (x >= 0 && x < para_space.cols && y >= 0 && y < para_space.rows)
				{
					para_space.at<unsigned char>(y, x) += 1;
				}

			}
		}
	}

	double maxd = 0;
	int estimated_radius = 0;
	Point estimated_center;

	blur(para_space, para_space, Size(3,3));

	Mat para_space_show;
	scale255(para_space, para_space_show);
	scaleAdd(struct_map, 1, para_space_show, para_space_show);
	
	imwrite(string(subjName) + "-votes.jpg", para_space_show);

	double mval = 0;
	Point pt;
	minMaxLoc(para_space, NULL, &mval, NULL, &estimated_center);


	Point best_center;
	int majority_dist = 0;
	int majority_dist_count = 0;
	
	double largest_portion = 0;

	for (int y = estimated_center.y - 10; y < estimated_center.y + 10; y++)
	{
		for (int x = estimated_center.x - 10; x < estimated_center.x + 10; x++)
		{

			vector<int> counts;
			counts.resize(maxPupilRds + 3);

			for (auto contour : contours)
			{
				
				for (auto p : contour)
				{
					Point pt = p;
					double dist = sqrt((pt.x - x) * (pt.x - x) + (pt.y - y) * (pt.y - y));

					if (dist < minPupilRds || dist > maxPupilRds)
					{
						continue;
					}

					int distInt = cvRound(dist);

					counts[distInt] += 1;

				}
			}

			int curr_majority_dist = 0;
			int curr_majority_dist_count = 0;
			for (int i = minPupilRds; i <= maxPupilRds; i++)
			{
				int avg = cvRound((counts[i - 2] + counts[i - 1] + counts[i]
				                  + counts[i + 1] + counts[i + 2]) / 5.0);

			    if (avg > curr_majority_dist_count)
				{
					curr_majority_dist_count = avg;
					curr_majority_dist = i;
				}
			}

			if (curr_majority_dist_count > majority_dist_count)
			{
				majority_dist_count = curr_majority_dist_count;
				majority_dist = curr_majority_dist;
				best_center.x = x;
				best_center.y = y;
			}

		}

	}

	if (best_center.x <= maxPupilRds || best_center.x >= enhanced_eye.cols - maxPupilRds
		|| best_center.y <= maxPupilRds || best_center.y >= enhanced_eye.rows - maxPupilRds)
	{
		printf("Poor best_center: %d, %d\n", best_center.x, best_center.y);
		return -2;
	}

	pupil_center = best_center;
	pupil_rds = majority_dist;

	int iris_scale = 5;
	int iris_rect_width = iris_scale * pupil_rds * 2;
	Rect iris_rect;
	iris_rect.x = max(0, pupil_center.x - iris_rect_width / 2);
	iris_rect.y = max(0, pupil_center.y - iris_rect_width / 2);
	iris_rect.width = min(iris_rect_width, eye.cols - iris_rect.x - 1);
	iris_rect.height = min(iris_rect_width, eye.rows - iris_rect.y - 1);

	//Rect iris_rect(pupil_center.x - iris_rect_width / 2, pupil_center.y -iris_rect_width / 2, iris_rect_width, iris_rect_width);
	//Rect iris_rect(0, 0, eye.cols, eye.rows);

	Mat iris_roi(eye, iris_rect); 
	iris_roi = iris_roi.clone();

	Mat enhanced_iris_roi;
	//SSREnhancement(iris_roi, enhanced_iris_roi);
	enhanced_iris_roi = iris_roi.clone();

	Mat iris_roi_struct;
	StructureMap((string(subjName) + "-iris-rect-").c_str(), enhanced_iris_roi, 10, 0.2f, iris_roi_struct);
	imwrite(string(subjName) + "-iris-rect-tv.jpg", iris_roi_struct);


	//Canny(enhanced_iris_roi, iris_roi_struct, 32, 80);
	//imwrite(string(subjName) + "-iris-rect-canny.jpg", iris_roi_struct);

	vector<vector<Point> > iris_rect_contours;

	findContours(iris_roi_struct, iris_rect_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	majority_dist = 0;
	majority_dist_count = 0;
	int minIrisRds = cvRound(pupil_rds * 1.1);
	int maxIrisRds = cvRound(pupil_rds * iris_scale);

	Point est_pupil_center = pupil_center - Point(iris_rect.x, iris_rect.y);

	for (int y = est_pupil_center.y - 10; y < est_pupil_center.y + 10; y++)
	{
		for (int x = est_pupil_center.x - 10; x < est_pupil_center.x + 10; x++)
		{

			vector<int> counts;
			counts.resize(maxIrisRds + 3);

			for (auto contour : iris_rect_contours)
			{
				
				for (auto p : contour)
				{
					Point pt = p;
					double dist = sqrt((pt.x - x) * (pt.x - x) + (pt.y - y) * (pt.y - y));

					if (dist <= minIrisRds || dist > maxIrisRds)
					{
						continue;
					}


					int distInt = cvRound(dist);
					counts[distInt] += 1;

				}
			}

			int curr_majority_dist = 0;
			int curr_majority_dist_count = 0;
			for (int ri = minIrisRds; ri <= maxIrisRds; ri++)
			{
				int avg = cvRound((counts[ri - 2] + counts[ri - 1] + counts[ri]
				                  + counts[ri + 1] + counts[ri + 2]) / 5.0);

			    if (avg > curr_majority_dist_count)
				{
					curr_majority_dist_count = avg;
					curr_majority_dist = ri;
				}
			}

			if (curr_majority_dist_count > majority_dist_count)
			{
				majority_dist_count = curr_majority_dist_count;
				majority_dist = curr_majority_dist;
				best_center.x = x;
				best_center.y = y;
			}

		}

	}

	iris_center = best_center + Point(iris_rect.x, iris_rect.y);
	iris_rds = majority_dist;

	return 0;
}

int doFindEllipses(const char *subjName, Mat eye_img, float rIris, float rPuip, Point contours[], Mat &eyelid)
{

	Mat equalized_eye_img;

	//GaussianBlur(eye_img, equalized_eye_img, Size(3,3), 0);
	//equalizeHist(equalized_eye_img, equalized_eye_img);
	//Canny(equalized_eye_img, eye_edge, 64, 160);

	Mat eye_edge;
	StructureMap(subjName, eye_img, 0.1f, 0.15f, eye_edge);

	vector<Point2f> uppers;
	uppers.push_back(contours[7]);
	uppers.push_back(contours[0]);
	uppers.push_back(contours[1]);

	vector<Point2f> lowers;
	lowers.push_back(contours[3]);
	lowers.push_back(contours[4]);
	lowers.push_back(contours[5]);

	Mat coef1, coef2;

	fitPolynomial(uppers, 2, coef1);
	fitPolynomial(lowers, 2, coef2);

	int d = max(eye_img.rows, eye_img.cols);
	Mat curve(d, d, CV_32FC1);
	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < d; j++)
		{
			curve.at<float>(j, i) = static_cast<float>(i);
		}
	}

	Mat curve_t;
	transpose(curve, curve_t);

	Mat mask1 = curve.mul(curve) * coef1.at<float>(0,0) + curve * coef1.at<float>(1,0) + coef1.at<float>(2,0)- curve_t;
	Mat mask2 = curve.mul(curve) * coef2.at<float>(0,0) + curve * coef2.at<float>(1,0) + coef2.at<float>(2,0)- curve_t;

	double mind1, maxd1, mind2, maxd2;
	minMaxLoc(mask1, &mind1, &maxd1);
	minMaxLoc(mask2, &mind2, &maxd2);



	// Calculate mean and dev;
	int xl = min(static_cast<int>(uppers[0].x), static_cast<int>(lowers[0].x));
	int xu = max(static_cast<int>(uppers[2].x), static_cast<int>(lowers[2].x));

	float dr = rIris / 3;
	dr = max(2.0f, dr);

	float sumDist1 = 0, sumSqDist1 = 0, sumDist2 = 0, sumSqDist2 = 0;
	int cntEdge1 = 0, cntEdge2 = 0;
	for (int i = 0; i < eye_edge.rows; i++)
	{
		for (int j = xl; j < xu; j++)
		{
			float pcd1 = abs(mask1.at<float>(i, j));
			if (pcd1 < dr && eye_edge.at<unsigned char>(i, j) > 0)
			{
				//show.at<unsigned char>(i, j) = 255;
				sumDist1 += pcd1;
				sumSqDist1 += pcd1 * pcd1;
				cntEdge1++;
				
			}

			float pcd2 = abs(mask2.at<float>(i, j));
			if (pcd2 < dr && eye_edge.at<unsigned char>(i, j) > 0)
			{
				//show.at<unsigned char>(i, j) = 255;
				sumDist2 += pcd2;
				sumSqDist2 += pcd2 * pcd2;
				cntEdge2++;
				
			}
		}
	}

	cntEdge1 = max(1, cntEdge1);
	float avgDist1 = sumDist1 / cntEdge1;
	float dev1 = (sumSqDist1 - sumDist1) / cntEdge1;
	dev1 = max(1.0f, dev1);
	
	cntEdge2 = max(1, cntEdge2);
	float avgDist2 = sumDist2 / cntEdge2;
	float dev2 = (sumSqDist2 - sumDist2) / cntEdge2;
	dev2 = max(1.0f, dev2);

	// Eliminate unqaulified edge points
	vector<Point2f> candidates1, candidates2;
	for (int i = 0; i < eye_edge.rows; i++)
	{
		for (int j = xl; j < xu; j++)
		{
			float pcd1 = abs(mask1.at<float>(i, j));
			if (pcd1 < dr && eye_edge.at<unsigned char>(i, j) > 0)
			{
				if (pow((pcd1 - avgDist1), 2) / dev1 < 3)
				{
					candidates1.push_back(Point2f(static_cast<float>(j), static_cast<float>(i)));
				}
			}

			float pcd2 = abs(mask2.at<float>(i, j));
			if (pcd2 < dr && eye_edge.at<unsigned char>(i, j) > 0)
			{
				if (pow((pcd2 - avgDist2), 2) / dev2 < 3)
				{
					candidates2.push_back(Point2f(static_cast<float>(j), static_cast<float>(i)));
				}
			}
		}
	}

	Mat show_img = eye_img.clone();
	Mat better_coef1, better_coef2;
	int ret = fitPolynomial(candidates1, 2, better_coef1, show_img);
	if (ret != 0)
	{
		better_coef1 =coef1;
	}

	ret = fitPolynomial(candidates2, 2, better_coef2, show_img);
	if (ret != 0)
	{
		better_coef2 = coef2;
	}

	eyelid = Mat::zeros(eye_img.rows, eye_img.cols, CV_8UC1);
	for (int i = 0; i < eye_img.rows; i++)
	{
		for (int j = 0; j < eye_img.cols; j++)
		{
			if (mask1.at<float>(i, j) > 0)
			{
				eyelid.at<unsigned char>(i, j) = 0;
			}

			if (mask2.at<float>(i, j) < 0)
			{
				eyelid.at<unsigned char>(i, j) = 0;
			}
		}
	}

	
	//drawPolynomial(coef1, 0, show_img);
	//drawPolynomial(coef2, 0, show_img);
	//imshow("Origin Poly", show_img);
	//drawPolynomial(better_coef1, 0, show_img);
	//drawPolynomial(better_coef2, 0, show_img);
	//imshow("Two Poly", show_img);
	//imshow("Edge", eye_edge);
	//imshow("Eydlid", eyelid);
	


	return 0;
}

int iris_segment(const char *subjName, Mat src, Mat &dst)
{
	//////save iris image before iris segment for test segment algorithm
//	imwrite("./iris_for_checking/resource/iris_" + string(iris_for_checking) + ".jpg", src);
//	imwrite("./iris_for_checking/left_iris_" + string(iris_for_checking) + "_resource.jpg", left_eye);


	Mat highlights_mask, no_highlights;

	Mat show2 = src.clone();

	reflection_removal(src, highlights_mask, no_highlights);
//	imwrite(string(subjName) + "-highlights.jpg", highlights_mask);

	Mat enhanced_img;
//	SSREnhancement(no_highlights, enhanced_img);
	enhanced_img = no_highlights.clone();
	


	Mat tv_struct;
	StructureMap(subjName, enhanced_img, 1, 0.2f, tv_struct);
//	imwrite(string(subjName) + "-tv-struct.jpg", tv_struct);

	Point iris_center, pupil_center;
	int iris_rds = 0, pupil_rds = 0;

	irisPupilCircleDetect2(subjName, no_highlights, enhanced_img, tv_struct, 
		10, 30, iris_center, iris_rds, pupil_center, pupil_rds);

	Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
    //circle(mask, iris_center, iris_rds, Scalar(255), -1);
    //circle(mask, pupil_center, pupil_rds, Scalar(0), -1);
	
//	imwrite(string(subjName) + "-iris-mask.jpg", mask);

	Mat show = no_highlights.clone();
	//circle(show, iris_center, 2, Scalar(255), 1);
	//circle(show, iris_center, iris_rds, Scalar(255));

	//circle(show, pupil_center, 2, Scalar(255), 1);
	//circle(show, pupil_center, pupil_rds, Scalar(255));
	//
	//
	circle(show2, iris_center, 2, Scalar(255), 1);
    circle(show2, pupil_center, 2, Scalar(0), 1);
	circle(show2, iris_center, iris_rds, Scalar(255), 1);
    circle(show2, pupil_center, pupil_rds, Scalar(0), 1);

//	imwrite(string(subjName) + "-iris-circle.jpg", show);	

	//Mat eyelid;
	//doFindEllipses(subjName, no_highlights, static_cast<float>(iris_rds), static_cast<float>(pupil_rds), contours, eyelid);

	Mat eyelid = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));
	Mat dst0;
	bitwise_and(mask, eyelid, dst0);
	bitwise_and(highlights_mask, dst0, dst0);
	//dst = dst0;
	dst = show2;

	//////save iris image after iris segment for checking
	char rds_pupil[20], rds_iris[20];
	sprintf(rds_pupil, "%d", pupil_rds);
	sprintf(rds_iris, "%d", iris_rds);
	char pos_pupil[20], pos_iris[20];
	sprintf(pos_pupil, "%d,%d", pupil_center.x,pupil_center.y);
	sprintf(pos_iris, "%d,%d", iris_center.x,iris_center.y);
	//imwrite("./iris_for_checking/result/" + string(iris_for_checking) + subjName + "_iris: " + string(pos_iris) + ", " + string(rds_iris) + "_pupil_rds: " + string(pos_pupil) + ", " + string(rds_pupil) + ".jpg", show2);
//	imwrite("./iris_for_checking/left_iris_" + string(iris_for_checking) + "_result.jpg", left_iris);

	return 0;
}


//////For stasm
// int iris_segment(const char *subjName, Mat src, Mat &right_iris, Mat &left_iris)
// {
// 	//////save iris image before iris segment for test segment algorithm
// 	//imwrite("./iris_for_checking/head/head_" + string(iris_for_checking) + ".jpg", src);
// 
// 	//////open record_for_irisSegment file for save cost time in iris segment
// 	FILE *record_for_irisSegment;
// 	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
// 
// 
// 	int foundface;
//     float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)
// 
// 	clock_t current_time = clock(), last_time;
// 
//     if (!stasm_search_single(&foundface, landmarks,
// 		(const char*)src.data, src.cols, src.rows, subjName, "data/stasm"))
//     {
//         printf("Error in stasm_search_single: %s\n", stasm_lasterr());
// 		return 0;
//     }
// 
// 	if (!foundface)
// 	{
// 		printf("No face found in %s\n", "");
// 		return 0;
// 	}
// 
// 	Point right_eye_center, ep0, ep1, ep2, ep3; // Right Eye, top corner, left corner, bottom corner, right corner
// 	Point left_eye_center, ep4, ep5, ep6, ep7;
// 	Point corners[18];
// 	stasm_force_points_into_image(landmarks, src.cols, src.rows);
// 
// 	last_time = current_time;
// 	current_time = clock();
// 	clock_t duration = current_time - last_time;
// 	printf("Landmarks detection cost: %f secs\n", (double)duration / CLOCKS_PER_SEC);
// 
// 	//////compute the time cost for Landmarks detection
// 	fprintf(record_for_irisSegment, "Landmarks detection cost: %0.2fms.\r\n", (double)duration);
// 	
// 	corners[0].x = cvRound(landmarks[38*2]);
// 	corners[0].y = cvRound(landmarks[38*2+1]);
// 	corners[1].x = cvRound(landmarks[32*2]);
// 	corners[1].y = cvRound(landmarks[32*2+1]);
// 	corners[2].x = cvRound(landmarks[31*2]);
// 	corners[2].y = cvRound(landmarks[31*2+1]);
// 	corners[3].x = cvRound(landmarks[30*2]);
// 	corners[3].y = cvRound(landmarks[30*2+1]);
// 	corners[4].x = cvRound(landmarks[37*2]);
// 	corners[4].y = cvRound(landmarks[37*2+1]);
// 	corners[5].x = cvRound(landmarks[36*2]);
// 	corners[5].y = cvRound(landmarks[36*2+1]);
// 	corners[6].x = cvRound(landmarks[35*2]);
// 	corners[6].y = cvRound(landmarks[35*2+1]);
// 	corners[7].x = cvRound(landmarks[34*2]);
// 	corners[7].y = cvRound(landmarks[34*2+1]);
// 	corners[8].x = cvRound(landmarks[33*2]);
// 	corners[8].y = cvRound(landmarks[33*2+1]);
// 
// 	corners[9].x = cvRound(landmarks[39*2]);
// 	corners[9].y = cvRound(landmarks[39*2+1]);
// 	corners[10].x = cvRound(landmarks[42*2]);
// 	corners[10].y = cvRound(landmarks[42*2+1]);
// 	corners[11].x = cvRound(landmarks[43*2]);
// 	corners[11].y = cvRound(landmarks[43*2+1]);
// 	corners[12].x = cvRound(landmarks[44*2]);
// 	corners[12].y = cvRound(landmarks[44*2+1]);
// 	corners[13].x = cvRound(landmarks[45*2]);
// 	corners[13].y = cvRound(landmarks[45*2+1]);
// 	corners[14].x = cvRound(landmarks[46*2]);
// 	corners[14].y = cvRound(landmarks[46*2+1]);
// 	corners[15].x = cvRound(landmarks[47*2]);
// 	corners[15].y = cvRound(landmarks[47*2+1]);
// 	corners[16].x = cvRound(landmarks[40*2]);
// 	corners[16].y = cvRound(landmarks[40*2+1]);
// 	corners[17].x = cvRound(landmarks[41*2]);
// 	corners[17].y = cvRound(landmarks[41*2+1]);
// 
// 	//Mat show_img = img.clone();
// 	//for (int i = 0; i < 18; i++)
// 	//{
// 	//	circle(show_img, corners[i], 2, Scalar(255, 0 , 0), 1);
// 	//}
// 
// 	//imwrite(path + "-landmarks.jpg", show_img);
// 	
// 	int ew = corners[3].x - corners[7].x;
// 	int eh = corners[5].y - corners[1].y;
//     Rect right_eye_roi(corners[7].x - ew / 3, corners[1].y - ew / 3, (corners[3].x - corners[7].x + ew * 2 / 3), (corners[5].y - corners[1].y + ew * 2 / 3));
// 	ew = corners[12].x - corners[16].x;
// 	eh = corners[14].y - corners[10].y;
// 	Rect left_eye_roi(corners[16].x - ew / 3, corners[10].y - ew / 3, (corners[12].x - corners[16].x + ew * 2 / 3), (corners[14].y - corners[10].y + ew * 2 / 3));
// 
// 	Point right_eye_4_corners[4];
// 	Point left_eye_4_corners[4];
// 	right_eye_4_corners[0] = corners[1];
// 	right_eye_4_corners[1] = corners[3];
// 	right_eye_4_corners[2] = corners[5];
// 	right_eye_4_corners[3] = corners[7];
// 
// 	left_eye_4_corners[0] = corners[10];
// 	left_eye_4_corners[1] = corners[12];
// 	left_eye_4_corners[2] = corners[14];
// 	left_eye_4_corners[3] = corners[16];
// 
// 	Point right_eye_8_corners[8];
// 	Point left_eye_8_corners[8];
// 	right_eye_8_corners[0] = corners[1];
// 	right_eye_8_corners[1] = corners[2];
// 	right_eye_8_corners[2] = corners[3];
// 	right_eye_8_corners[3] = corners[4];
// 	right_eye_8_corners[4] = corners[5];
// 	right_eye_8_corners[5] = corners[6];
// 	right_eye_8_corners[6] = corners[7];
// 	right_eye_8_corners[7] = corners[8];
// 
// 	
// 	left_eye_8_corners[0] = corners[10];
// 	left_eye_8_corners[1] = corners[11];
// 	left_eye_8_corners[2] = corners[12];
// 	left_eye_8_corners[3] = corners[13];
// 	left_eye_8_corners[4] = corners[14];
// 	left_eye_8_corners[5] = corners[15];
// 	left_eye_8_corners[6] = corners[16];
// 	left_eye_8_corners[7] = corners[17];
// 
// 
// 	for (int i = 0; i < 4; i++)
// 	{
// 		right_eye_4_corners[i].x -= right_eye_roi.x;
// 		right_eye_4_corners[i].y -= right_eye_roi.y;
// 		
// 		left_eye_4_corners[i].x -= left_eye_roi.x;
// 		left_eye_4_corners[i].y -= left_eye_roi.y;
// 	}
// 
// 	for (int i = 0; i < 8; i++)
// 	{
// 		right_eye_8_corners[i].x -= right_eye_roi.x;
// 		right_eye_8_corners[i].y -= right_eye_roi.y;
// 		
// 		left_eye_8_corners[i].x -= left_eye_roi.x;
// 		left_eye_8_corners[i].y -= left_eye_roi.y;
// 	}
// 
// 	Mat right_eye(src, right_eye_roi);
// 	right_eye = right_eye.clone();
// 
// 	Mat left_eye(src, left_eye_roi);
// 	left_eye = left_eye.clone();
// 
// 		
// 	//iris segment
// 	Mat m1, m2;
// 	int lRet;
// 
// 	//////change to new iris_segment method
// 	//lRet = iris_segment(subjName, right_eye, &right_eye_8_corners[0], m1);
// 	//lRet = iris_segment_new(subjName, right_eye, m1);
// 
// 	if(lRet != 0)
// 	{
// 		printf("Detect pupil or iris failed!\n");
// 		return 0;
// 	}
// 
// 	//////compute the time cost for right iris segment
// 	last_time = current_time;
// 	current_time = clock();
// 	duration = current_time - last_time;
// 
// 	fprintf(record_for_irisSegment, "right iris segment time cost: %0.2fms.\r\n", (double)duration);
// 
// 	//////change to new iris_segment method
// 	//lRet = iris_segment(subjName, left_eye, &left_eye_8_corners[0], m2);
// 	//lRet = iris_segment_new(subjName, left_eye, m2);
// 
// 	if(lRet != 0)
// 	{
// 		printf("Detect pupil or iris failed!\n");
// 		return 0;
// 	}
// 
// 	//////compute the time cost for left iris segment
// 	last_time = current_time;
// 	current_time = clock();
// 	duration = current_time - last_time;
// 
// 	fprintf(record_for_irisSegment, "left iris segment time cost: %0.2fms.\r\n\r\n", (double)duration);
// 
// 	right_iris = m1.clone();
// 	left_iris = m2.clone();
// 
// 	//////save iris image before iris segment for test segment algorithm
// 	imwrite("./iris_for_checking/result/right_" + string(iris_for_checking) + ".jpg", right_iris);
// 	imwrite("./iris_for_checking/result/left_" + string(iris_for_checking) + ".jpg", left_iris);
// 
// 	//////close the record file
// 	fclose(record_for_irisSegment);
// }
// 
// int iris_segment_new(const char *subjName, Mat src, Mat &dst)
// {
// 	int times = 3;
// 	int pupil_min_rds = 3* times;
// 	int pupil_max_rds = 20 * times;
// 
// 	Point pupil_center;
// 	int pupil_rds = 0;
// 	
// 	RadialSymmetryTran(src, pupil_min_rds, pupil_max_rds, pupil_center, pupil_rds);
// 
// 	if(pupil_rds == 0)
// 	{
// 		return 1;
// 	}
// 
// 	Point iris_center;
// 	int iris_rds = 0;
// 	int iris_min_rds = cvRound(pupil_rds * 1.5), iris_max_rds = pupil_rds * 5;
// 	int search_dis = 2;
// 
// 	CircleDifference(src, iris_min_rds,iris_max_rds, search_dis, pupil_center, iris_center, iris_rds);
// 	//iris_rds = cvRound(pupil_rds*2);
// 	//iris_center.x = pupil_center.x;
// 	//iris_center.y = pupil_center.y;
// 
// 	if(iris_rds == 0)
// 	{
// 		return 2;
// 	}
// 
// 	dst = src.clone();
// 
// 	circle(dst, pupil_center, pupil_rds, Scalar(255));
// 	circle(dst, iris_center, iris_rds, Scalar(0));
// 
// 	return 0;
// }

int RadialSymmetryTran(Mat src, int iMinR, int iMaxR, Point &center, int &rds)
{
	FILE *record_for_irisSegment;

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "\r\nBegin.\r\n");
	fclose(record_for_irisSegment);
	int i,j,k,kk;

	int times = 2;
	iMinR = iMinR/times, iMaxR = iMaxR/times;

	Mat src_cut = Mat::zeros((src.rows-src.rows%times),(src.cols-src.cols%times),CV_8UC1);
	for(i = 0; i < src_cut.rows; i++)
		for(j = 0; j < src_cut.cols; j++)
		{
			src_cut.data[i*src_cut.cols + j] = src.data[i*src.cols + j];
		}

	Mat src_small = Mat::zeros(src_cut.rows/times,src_cut.cols/times, CV_8UC1);

	for(i = 0; i< src_small.rows; i++)
		for(j = 0; j < src_small.cols; j++)
		{
			src_small.data[i*src_small.cols + j] = src_cut.data[i*times*src_cut.cols +j*times];
		}

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "After resize image.\r\n");
	fclose(record_for_irisSegment);

	int iWidth = src_small.cols, iHeight = src_small.rows;

	double *pGradx;
	double *pGrady;
	double *pGradxy;
	double *pMk;
	double *pGk;
	double *pOk;
	double *pFk;
	double *pSk;

	pGradx=new double[iWidth*iHeight];
	pGrady=new double[iWidth*iHeight];
	pGradxy=new double[iWidth*iHeight];
	memset(pGradx, 0, iWidth*iHeight*sizeof(double));
	memset(pGrady, 0, iWidth*iHeight*sizeof(double));
	memset(pGradxy, 0, iWidth*iHeight*sizeof(double));

	int iTemp1,iTemp11,iTemp2;

	uchar *src_data = src_small.data;

	for(i=1;i<iHeight-1;i++)
	{
		iTemp1=i*iWidth;
		for(j=1;j<iWidth-1;j++)
		{
			iTemp2=iTemp1+j;
			pGradx[iTemp2] = src_data[iTemp2-iWidth-1]+2*src_data[iTemp2-1]+src_data[iTemp2+iWidth-1]-src_data[iTemp2-iWidth+1]-2*src_data[iTemp2+1]-src_data[iTemp2+iWidth+1];
			pGrady[iTemp2] = src_data[iTemp2-iWidth-1]+2*src_data[iTemp2-iWidth]+src_data[iTemp2-iWidth+1]-src_data[iTemp2+iWidth-1]-2*src_data[iTemp2+iWidth]-src_data[iTemp2+iWidth+1];
			pGradxy[iTemp2] = sqrt(pGradx[iTemp2]*pGradx[iTemp2] + pGrady[iTemp2]*pGrady[iTemp2]);
		}
	}
	int iGrayThreshold = 110;
	int iGrayThreshold1 = 115;
	for(i=iWidth+1;i<(iHeight-1)*iWidth-1;i++)
	{
		if(src_data[i] >=iGrayThreshold || src_data[i-1] >iGrayThreshold1 || src_data[i+1] >iGrayThreshold1 
			|| src_data[i+iWidth] >iGrayThreshold1 || src_data[i+iWidth+1] >iGrayThreshold1 || src_data[i+iWidth-1] >iGrayThreshold1 
			|| src_data[i-iWidth] >iGrayThreshold1 || src_data[i-iWidth+1] >iGrayThreshold1 || src_data[i-iWidth-1] >iGrayThreshold1)
		{
			pGradxy[i] =0;
		}
		else if(pGradxy[i] <0.01 || (pGrady[i] >0 && (fabs(pGradx[i])<pGradxy[i]/2)))
		{
			pGradxy[i] =0;
		}
	}
	int *pCutImage2 = new int[iWidth*iHeight];
	double dMaxppp=0;
	int iThreshold1=iGrayThreshold/3;
	for(i=0;i<iWidth*iHeight;i++)
	{
		if(pGradxy[i] !=0)
		{
			if(src_data[i]<iThreshold1)
				pCutImage2[i] = iGrayThreshold - src_data[i]*2;
			else
				pCutImage2[i] = (iGrayThreshold - src_data[i])/2;
		}
		else
			pCutImage2[i] = 0;
	}

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "After calculate grad.\r\n");
	fclose(record_for_irisSegment);

	int iIterativeNum = (iMaxR-iMinR)/2+1;
	pMk = new double[iWidth*iHeight];
	pGk = new double[iWidth*iHeight];
	pOk = new double[iWidth*iHeight*iIterativeNum];
	pFk = new double[iWidth*iHeight*iIterativeNum];	
	pSk = new double[iWidth*iHeight];
	memset(pSk, 0, iWidth*iHeight*sizeof(double));
	memset(pOk, 0, iWidth*iHeight*iIterativeNum*sizeof(double));
	int iXMove,iYMove;
	for(k=iMinR;k<=iMaxR;k=k+2)
	{
		memset(pMk, 0, iWidth*iHeight*sizeof(double));
		memset(pGk, 0, iWidth*iHeight*sizeof(double));
		kk=((k-iMinR)/2)*iWidth*iHeight;
		for(i=1;i<iHeight-1;i++)
		{
			iTemp1=i*iWidth;
			for(j=1;j<iWidth-1;j++)
			{
				iTemp11 = iTemp1+j;
				if(pGradxy[iTemp11] !=0)
				{
					iXMove=j+cvRound(k*pGradx[iTemp11]/pGradxy[iTemp11]);
					iYMove=i+cvRound(k*pGrady[iTemp11]/pGradxy[iTemp11]);
					if(iXMove>=0 && iXMove<iWidth && iYMove>=0 && iYMove<iHeight)
					{
						iTemp2=iYMove*iWidth+iXMove;
						pMk[iTemp2]= pMk[iTemp2]+pGradxy[iTemp11];
						pGk[iTemp2]= pGk[iTemp2]+pCutImage2[iTemp11];
						pOk[kk+iTemp2]= pOk[kk+iTemp2]+1;
					}
				}
			}
		}
		for(i=0;i<iWidth*iHeight;i++)
		{
			pFk[kk+i]=pMk[i]*pGk[i];
			pSk[i]=pSk[i]+pFk[kk+i];
		}
	}
//	lRet=GaussLowPassDouble(pSk, iWidth, iHeight, g_dGaussRST9, 9);

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "After calculate Fk and Sk.\r\n");
	fclose(record_for_irisSegment);

	int iMaxX,iMaxY;
	double dMaxV;
	dMaxV=-1000;
	for(i=4;i<iHeight-4;i++)
	{
		iTemp1=i*iWidth;
		for(j=4;j<iWidth-4;j++)
		{
			if(dMaxV < pSk[iTemp1+j])
			{
				dMaxV = pSk[iTemp1+j];
				iMaxX = j;
				iMaxY = i;
			}
		}
	}
	int iNearPixNum=4; 
	int iFilterLeft,iFilterRight,iFilterTop,iFilterBottom; 
	if(iMaxX-4<iNearPixNum)
	{
		iFilterLeft = 0;
	}
	else
	{
		iFilterLeft = iMaxX-4-iNearPixNum;
	}
	if(iMaxX+4>iWidth-iNearPixNum)
	{
		iFilterRight = iWidth-1;
	}
	else
	{
		iFilterRight = iMaxX+4+iNearPixNum;
	}
	
	if(iMaxY-4<iNearPixNum)
	{
		iFilterTop = 0;
	}
	else
	{
		iFilterTop = iMaxY-4-iNearPixNum;
	}
	if(iMaxY+4>iHeight-iNearPixNum)
	{
		iFilterBottom = iHeight-1;
	}
	else
	{
		iFilterBottom = iMaxY+4+iNearPixNum;
	}

	int iXLen,iYLen;
	iXLen=iFilterRight-iFilterLeft+1;
	iYLen=iFilterBottom-iFilterTop+1;
	double *pSmallFk = new double[iXLen*iYLen];
	double dMaxFk=-1000;
	int iMaxFkX,iMaxFkY;  
	int iMaxFkR; 
	for(k=0;k<iIterativeNum;k++)
	{
		kk=k*iWidth*iHeight;
	
		for(i=iFilterTop;i<=iFilterBottom;i++)
		{
			iTemp1=(i-iFilterTop)*iXLen;
			iTemp2=kk+i*iWidth;
			for(j=iFilterLeft;j<=iFilterRight;j++)
			{
				pSmallFk[iTemp1+(j-iFilterLeft)]=pFk[iTemp2+j];
			}
		}

		//lRet=GaussLowPassDouble(pSmallFk, iXLen, iYLen, g_dGaussRST9, 9);

		for(i=4;i<iYLen-4;i++)
		{
			iTemp1=i*iXLen;
			for(j=4;j<iXLen-4;j++)
			{
				if(dMaxFk < pSmallFk[iTemp1+j])
				{
					dMaxFk = pSmallFk[iTemp1+j];
					iMaxFkX = j;
					iMaxFkY = i;
					iMaxFkR = k; 
				}
			}
		}
	}

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "The end.\r\n");
	fclose(record_for_irisSegment);

	center.x = (iFilterLeft+iMaxFkX)*times;
	center.y = (iFilterTop+iMaxFkY)*times;
	rds = (iMinR+iMaxFkR*2)*times;

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "pupil_center: (%d, %d), pupil_rds = %d.\r\n\r\n", center.x, center.y, rds);
	fclose(record_for_irisSegment);

	delete []pCutImage2;
//	delete []pSmallOk;
	delete []pSmallFk;
	delete []pMk;
	delete []pGk;
	delete []pOk;
	delete []pFk;
	delete []pSk;
	delete []pGradx; 
	delete []pGrady; 
	delete []pGradxy; 	

	return 0;
}

int FindIrisCircle(Mat src, cv::Point pupil_center, int pupil_rds, cv::Point& iris_center, int& iris_rds)
{
	int i,j,k;

	Mat src_canny, canny_result, tmp_result;
	GaussianBlur(src,src_canny,cv::Size(9,9),2,2);
	Canny(src_canny,canny_result,5,30);

	int points_num = 0;
	Vector<cv::Point> canny_points;
	Vector<int> canny_R;
	int coarse_R[1000];
	memset(coarse_R,0,256*sizeof(int));
	for(i = 1; i < canny_result.rows-1; i++)
		for(j = 1; j < canny_result.cols-1; j++)
		{
			
			if(canny_result.data[i*canny_result.cols+j] != 0)
			{				
				int y_value = ((src.data[(i+1)*canny_result.cols+j]*2+src.data[(i+1)*canny_result.cols+j+1]+src.data[(i+1)*canny_result.cols+j-1])-(src.data[(i-1)*canny_result.cols+j]*2+src.data[(i-1)*canny_result.cols+j+1]+src.data[(i-1)*canny_result.cols+j-1]))/4;
				int x_value = ((src.data[(i)*canny_result.cols+j]*2+src.data[(i-1)*canny_result.cols+j+1]+src.data[(i+1)*canny_result.cols+j+1])-(src.data[(i)*canny_result.cols+j-1]*2+src.data[(i+1)*canny_result.cols+j-1]+src.data[(i-1)*canny_result.cols+j-1]))/4;
				if(x_value == 0)
				{
					canny_result.data[i*canny_result.cols+j] = 0;
				}
				else if(abs(y_value/x_value) > 3.732)
				{
					canny_result.data[i*canny_result.cols+j] = 0;
				}
				else
				{
					int tmp_R = sqrt((i-pupil_center.y)*(i-pupil_center.y)+(j-pupil_center.x)*(j-pupil_center.x));
					coarse_R[tmp_R]++;
					canny_points.push_back(cv::Point(j,i));
					canny_R.push_back(tmp_R);
				}
			}
			else
			{
				canny_result.data[i*canny_result.cols+j] = 0;
			}
		}
	
	double max_val = 0;
	for(i = pupil_rds+10; i < (8*pupil_rds>1000?1000:8*pupil_rds); i++)
	{
		coarse_R[i] = (coarse_R[i]*2+coarse_R[i+1] +coarse_R[i-1])/4;
		double tmp_rat = (double)coarse_R[i]/i*pupil_rds;
		if(tmp_rat > max_val)
		{
			max_val = tmp_rat;
			iris_rds = i;
		}
	}
	iris_center.x = pupil_center.x;
	iris_center.y = pupil_center.y;

	cv::Point tmp_center;
	int tmp_rds[500];
	max_val = 0;
	int best_iris_rds = 0;
	cv::Point best_iris_point;
	for(i = -3; i < 4; i++)
	{
		for(j = -3; j < 4; j++)
		{
			memset(tmp_rds,0,500*sizeof(int));
			tmp_center.x = pupil_center.x + j;
			tmp_center.y = iris_center.y + i;

			for(k = 0; k < canny_R.size(); k++)
			{
				if(abs(canny_R[k] - iris_rds) < 4)
				{
					int tmp_R = sqrt((canny_points[k].y-tmp_center.y)*(canny_points[k].y-tmp_center.y)+(canny_points[k].x-tmp_center.x)*(canny_points[k].x-tmp_center.x));
					tmp_rds[tmp_R]++;
				}
			}
			for(k = iris_rds - 10; k < iris_rds + 10; k++)
			{
				
				tmp_rds[k] = (tmp_rds[k]*2+tmp_rds[k+1] +tmp_rds[k-1])/4;
				if(tmp_rds[k] > max_val)
				{
					max_val = tmp_rds[k];
					best_iris_rds = k;
					best_iris_point.x = tmp_center.x;
					best_iris_point.y = tmp_center.y;
				}
			}
		}
	}

	return 0;
}

int FineLocatePupil(Mat src, int times, Point& pupil_center, int& pupil_rds)
{
	FILE *record_for_irisSegment;

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "\r\nBegin fineLocatePupil.\r\n");
	fclose(record_for_irisSegment);

	int i,j,k;
	pupil_center.x = pupil_center.x * times;
	pupil_center.y = pupil_center.y * times;
	pupil_rds = pupil_rds * times;

	cv::Point best_pupil_center;
	int best_pupil_rds = pupil_rds;

	Point left_top, right_bottom;

	left_top.x = pupil_center.x - 2 * pupil_rds;
	left_top.y = pupil_center.y - 2 * pupil_rds;

	right_bottom.x = pupil_center.x + 2 * pupil_rds;
	right_bottom.y = pupil_center.y + 2 * pupil_rds;

	if(left_top.x < 5)
	{
		left_top.x = 5;
	}
	if(left_top.y < 5)
	{
		left_top.y = 5;
	}
	if(right_bottom.x > src.cols - 5)
	{
		right_bottom.x = src.cols - 5;
	}
	if(right_bottom.y > src.rows - 5)
	{
		right_bottom.y = src.rows - 5;
	}

	Rect pupil_rect(left_top.x,left_top.y,(right_bottom.x - left_top.x),(right_bottom.y - left_top.y));;

	Mat tmp_pupil, pupil_roi;
	
	tmp_pupil = src(pupil_rect);
	pupil_roi = tmp_pupil.clone();

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "After cut pupil roi.\r\n");
	fclose(record_for_irisSegment);

	pupil_center.x = pupil_center.x - left_top.x;
	pupil_center.y = pupil_center.y - left_top.y;

	GaussianBlur(pupil_roi, pupil_roi, Size(5,5), 0 , 0);

	Mat canny_result;
	Canny(pupil_roi, canny_result, 10, 30);
	
	int points_num = 0;
	Vector<cv::Point> canny_points;
	Vector<int> canny_R;
	int coarse_R[500];
	memset(coarse_R,0,500*sizeof(int));
	for(i = 1; i < canny_result.rows-1; i++)
		for(j = 1; j < canny_result.cols-1; j++)
		{
			if(canny_result.data[i*canny_result.cols+j] != 0)
			{	
				int tmp_R = sqrt((i-pupil_center.y)*(i-pupil_center.y)+(j-pupil_center.x)*(j-pupil_center.x));
				coarse_R[tmp_R]++;
				canny_points.push_back(cv::Point(j,i));
				canny_R.push_back(tmp_R);
			}
		}

	
	double max_val = 0;

	for(i = pupil_rds/2+1; i < 3*pupil_rds; i++)
	{
		coarse_R[i] = (coarse_R[i]*2+coarse_R[i+1] +coarse_R[i-1])/4;
		double tmp_rat = (double)coarse_R[i];
		if(tmp_rat > max_val)
		{
			max_val = tmp_rat;
			best_pupil_rds = i;
		}
	}
	
	pupil_rds = best_pupil_rds;

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "After first location.\r\n");
	fclose(record_for_irisSegment);

	cv::Point tmp_center;
	int tmp_rds[500];
	max_val = 0;
	
	for(i = -2; i < 3; i++)
	{
		for(j = -2; j < 3; j++)
		{
			memset(tmp_rds,0,500*sizeof(int));
			tmp_center.x = pupil_center.x + j;
			tmp_center.y = pupil_center.y + i;

			for(k = 0; k < canny_R.size(); k++)
			{
				if(abs(canny_R[k] - pupil_rds) < 4)
				{
					int tmp_R = sqrt((canny_points[k].y-tmp_center.y)*(canny_points[k].y-tmp_center.y)+(canny_points[k].x-tmp_center.x)*(canny_points[k].x-tmp_center.x));
					tmp_rds[tmp_R]++;
				}
			}
			
			for(k = 2; k < pupil_rds + 10; k++)
			{
				
				tmp_rds[k] = (tmp_rds[k]*2+tmp_rds[k+1] +tmp_rds[k-1])/4;
				if(tmp_rds[k] > max_val)
				{
					max_val = tmp_rds[k];
					best_pupil_rds = k;
					best_pupil_center.x = tmp_center.x;
					best_pupil_center.y = tmp_center.y;
				}
			}
		}
	}

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "The end.\r\n");
	fclose(record_for_irisSegment);
	
	pupil_center.x = (best_pupil_center.x + left_top.x)/times;
	pupil_center.y = (best_pupil_center.y + left_top.y)/times;
	pupil_rds = best_pupil_rds/times;

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "pupil_center: (%d, %d), pupil_rds = %d.\r\n\r\n", pupil_center.x, pupil_center.y, pupil_rds);
	fclose(record_for_irisSegment);

	return 0;
}

int LocatePupilandIris(Mat src, int times, int min_rds,int max_rds,cv::Point& pupil_center, int& pupil_rds, cv::Point& iris_center, int& iris_rds)
{
	//////open record_for_irisSegment file for save cost time in iris segment
	FILE *record_for_irisSegment;

	int i,j;
	int iMinR = min_rds/times, iMaxR = max_rds/times;

	pupil_center.x = 0;
	pupil_center.y = 0;
	pupil_rds = 0;
	iris_center.x = 0;
	iris_center.y = 0;
	iris_rds = 0;

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "Before ImageResize.\r\n");
	fclose(record_for_irisSegment);

	Mat src_cut = Mat::zeros((src.rows-src.rows%times),(src.cols-src.cols%times),CV_8UC1);
	for(i = 0; i < src_cut.rows; i++)
		for(j = 0; j < src_cut.cols; j++)
		{
			src_cut.data[i*src_cut.cols + j] = src.data[i*src.cols + j];
		}

	Mat src_small = Mat::zeros(src_cut.rows/times,src_cut.cols/times, CV_8UC1);

	for(i = 0; i< src_small.rows; i++)
		for(j = 0; j < src_small.cols; j++)
		{
			src_small.data[i*src_small.cols + j] = src_cut.data[i*times*src_cut.cols +j*times];
		}

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "Before SSREnhancement.\r\n");
	fclose(record_for_irisSegment);

	SSREnhancement(src_small,src_small);

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "Before GaussianBlur.\r\n");
	fclose(record_for_irisSegment);

	GaussianBlur(src_small, src_small, Size(5,5), 0 , 0);

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "Before RadialSymmetryTran.\r\n");
	fclose(record_for_irisSegment);

	RadialSymmetryTran(src_small, iMinR, iMaxR, pupil_center, pupil_rds);

	if(pupil_rds <= 0)
	{
		return -1;
	}

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "Before FineLocatePupil.\r\n");
	fclose(record_for_irisSegment);

	FineLocatePupil(src,times,pupil_center,pupil_rds);

	if(pupil_rds <= 0)
	{
		return -1;
	}

	record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	fprintf(record_for_irisSegment, "Before FindIrisCircle.\r\n");
	fclose(record_for_irisSegment);

	FindIrisCircle(src_small, pupil_center, pupil_rds, iris_center, iris_rds);

	if(iris_rds <= 0)
	{
		return -1;
	}
	
	//cvNamedWindow( "result", 1 );
	//imshow("result", src_small);
	//cvWaitKey(0);
	//cvDestroyWindow( "result" );

	return 0;
}

int CircleDifference(Mat src, int min_rds, int max_rds, int search_Dis, Point &center, Point &best_center, int &best_rds)
{
	int i, j;
	int AngleNum = 105, AngleStart = -45;
	int iGrayDiffThresh=5;
	int step = 1;

	int iWidth = src.cols, iHeight = src.rows;

	int temp_max_rds;
	if((center.x - search_Dis - max_rds) <= 0)
	{
		max_rds = center.x - search_Dis - 4;
	}
	if((center.x + search_Dis + max_rds) >= iWidth)
	{
		max_rds = iWidth - center.x - search_Dis - 4;
	}
	if((center.y - search_Dis - max_rds) <= 0)
	{
		max_rds = center.y - search_Dis - 4;
	}
	if((center.y + search_Dis + max_rds) >= iHeight)
	{
		max_rds = iHeight - center.y - search_Dis - 4;
	}

	if(min_rds >= max_rds)
	{
		printf("This iris image is not complete!\n");
		return 1;
	}

	int rdsNum = (int)(max_rds - min_rds)/step;

	for(i = 0; i < iHeight; i++)
		for(j = 0; j < iWidth; j++)
		{
			if(src.data[i*iWidth+j] > 235)
				src.data[i*iWidth+j] = 255;
		}

	double *pTheta=new double[AngleNum];
	double *pSinTheta=new double[AngleNum];
	double *pCosTheta=new double[AngleNum];

	for (i=0;i<AngleNum;i++)
	{
		pTheta[i]=(i+AngleStart)*IRIS_RG_PI/180;
		pSinTheta[i]=sin(pTheta[i]);
		pCosTheta[i]=cos(pTheta[i]);
	}

	int **pXSite=new int *[rdsNum];
	int **pYSite=new int *[rdsNum];
	int **ppiXYSite1=new int *[rdsNum];
	int **ppiXYSite2=new int *[rdsNum];
	int **ppiXYGray1=new int *[rdsNum];
	int **ppiXYGray2=new int *[rdsNum];

	for(i=0;i<rdsNum;i++)
	{
		pXSite[i]=new int[AngleNum];
		pYSite[i]=new int[AngleNum];
		ppiXYSite1[i]=new int[AngleNum];
		ppiXYSite2[i]=new int[AngleNum];
		ppiXYGray1[i]=new int[AngleNum];
		ppiXYGray2[i]=new int[AngleNum];
	}

	for (i=0;i<rdsNum;i++)
	{
		int ii=i*step;
		for (j=0;j<AngleNum;j++)
		{
			pXSite[i][j]=cvRound((min_rds+ii)*pCosTheta[j]);
			pYSite[i][j]=cvRound((min_rds+ii)*pSinTheta[j]);
			ppiXYSite1[i][j]=pYSite[i][j]*iWidth+pXSite[i][j];
			ppiXYSite2[i][j]=pYSite[i][j]*iWidth-pXSite[i][j]-pXSite[i][j];
		}
	}

	int move_disX, move_disY;
	int center_post;
	Point temp_center;
	int iRBios, dMax, dMaxDiff = -10000;
	for(move_disY = -search_Dis; move_disY < search_Dis; move_disY++)
	{
		temp_center.y = center.y + move_disY;
		for(move_disX = -search_Dis; move_disX < search_Dis; move_disX++)
		{
			temp_center.x = center.x + move_disX;
			center_post = temp_center.y*iWidth + temp_center.x;
			for (i=0;i<rdsNum;i++)
			{
				for (j=0;j<AngleNum;j++)
				{
					ppiXYGray1[i][j]=src.data[center_post+ppiXYSite1[i][j]];
					ppiXYGray2[i][j]=src.data[center_post+ppiXYSite2[i][j]];
				}
			}

			int iTemp1;
			int dTemp1;
			dMax=-1000;

			for(i = 0; i < rdsNum-2; i++)
			{
				int iGradsSum=0;
				int iGradsNum=0;
				int iOkDotNum=0;
				int iBrightDotNum=0;
				int iGradsSum1=0;
				int iGradsNum1=0;
				int iOkDotNum1=0;
				int iBrightDotNum1=0;
				for (j = 0; j < AngleNum; j++)
				{
					if(ppiXYGray1[i][j] != 255 && ppiXYGray1[i+2][j] != 255)
					{
						iTemp1 = ppiXYGray1[i+2][j] - ppiXYGray1[i][j];
						if(iTemp1 >iGrayDiffThresh)	
						{
							iGradsSum += iTemp1;	
							iOkDotNum++;
							iGradsNum++;
						}
						else if(iTemp1 > -10)
						{
							iGradsSum += iTemp1;
							iGradsNum++;
						}
						else
						{
							iBrightDotNum++;
						}
					}
					else
					{
						iBrightDotNum++;
					}

					if(ppiXYGray2[i][j] != 255 && ppiXYGray2[i+2][j] != 255)
					{
						iTemp1 = ppiXYGray2[i+2][j] - ppiXYGray2[i][j];
						if(iTemp1 >iGrayDiffThresh)	
						{
							iGradsSum1 += iTemp1;	
							iOkDotNum1++;
							iGradsNum1++;
						}
						else if(iTemp1 > -10)
						{
							iGradsSum1 += iTemp1;
							iGradsNum1++;
						}
						else
						{
							iBrightDotNum1++;
						}
					}
					else
					{
						iBrightDotNum1++;
					}
				}

				if(iGradsNum >2 && iGradsNum1 >2)
				{
					dTemp1 = cvRound((((double)(iGradsSum+iGradsSum1))/(iGradsNum+iGradsNum1+(double)(iBrightDotNum+iBrightDotNum1)/2))*(iOkDotNum+iOkDotNum1+(double)(iBrightDotNum+iBrightDotNum1)/8));
					//printf("dTemp1 = %d\n",dTemp1);
					if(dTemp1 >dMax)
					{
						dMax = dTemp1;
						iRBios = i;
					}
				}
			}

//			printf("dMax = %d, dMaxDiff = %d\n", dMax, dMaxDiff);

			if(dMax > dMaxDiff)
			{
//				printf("OK!\n");
				dMaxDiff=dMax;
				best_center.x = temp_center.x;
				best_center.y = temp_center.y;			
				best_rds = min_rds + iRBios*step;
			//	printf("best_rds = %d, min_rds = %d, iRBios = %d\n",best_rds, min_rds, iRBios);
			}
		}
	}

//	printf("best_center.x = %d, best_center.y = %d, best_rds = %d\n", best_center.x, best_center.y, best_rds);

	for(i=0;i<rdsNum;i++)
	{
		delete pXSite[i];
		delete pYSite[i];
		delete ppiXYSite1[i];
		delete ppiXYSite2[i];
		delete ppiXYGray1[i];
		delete ppiXYGray2[i];
	}

	delete pXSite;
	delete pYSite;
	delete ppiXYSite1;
	delete ppiXYSite2;
	delete ppiXYGray1;
	delete ppiXYGray2;
	delete []pTheta;
	delete []pSinTheta;
	delete []pCosTheta;

	return 0;
}

int pupil_coarse_location(Mat& src, int iMinR, int iMaxR, Point &pupil_center, int &pupil_rds)
{
	//FILE *record_for_irisSegment;

	//record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	//fprintf(record_for_irisSegment, "\r\nBegin.\r\n");
	//fclose(record_for_irisSegment);
	int i,j,k,kk;

	int times = 2;
	iMinR = iMinR/times, iMaxR = iMaxR/times;

	Mat src_cut = Mat::zeros((src.rows-src.rows%times),(src.cols-src.cols%times),CV_8UC1);
	for(i = 0; i < src_cut.rows; i++)
		for(j = 0; j < src_cut.cols; j++)
		{
			src_cut.data[i*src_cut.cols + j] = src.data[i*src.cols + j];
		}

	Mat src_small = Mat::zeros(src_cut.rows/times,src_cut.cols/times, CV_8UC1);

	for(i = 0; i< src_small.rows; i++)
		for(j = 0; j < src_small.cols; j++)
		{
			src_small.data[i*src_small.cols + j] = src_cut.data[i*times*src_cut.cols +j*times];
		}

	//record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	//fprintf(record_for_irisSegment, "After resize image.\r\n");
	//fclose(record_for_irisSegment);

	int iWidth = src_small.cols, iHeight = src_small.rows;

	double *pGradx;
	double *pGrady;
	double *pGradxy;
	double *pMk;
	double *pGk;
	double *pOk;
	double *pFk;
	double *pSk;

	pGradx=new double[iWidth*iHeight];
	pGrady=new double[iWidth*iHeight];
	pGradxy=new double[iWidth*iHeight];
	memset(pGradx, 0, iWidth*iHeight*sizeof(double));
	memset(pGrady, 0, iWidth*iHeight*sizeof(double));
	memset(pGradxy, 0, iWidth*iHeight*sizeof(double));

	int iTemp1,iTemp11,iTemp2;

	uchar *src_data = src_small.data;

	for(i=1;i<iHeight-1;i++)
	{
		iTemp1=i*iWidth;
		for(j=1;j<iWidth-1;j++)
		{
			iTemp2=iTemp1+j;
			pGradx[iTemp2] = src_data[iTemp2-iWidth-1]+2*src_data[iTemp2-1]+src_data[iTemp2+iWidth-1]-src_data[iTemp2-iWidth+1]-2*src_data[iTemp2+1]-src_data[iTemp2+iWidth+1];
			pGrady[iTemp2] = src_data[iTemp2-iWidth-1]+2*src_data[iTemp2-iWidth]+src_data[iTemp2-iWidth+1]-src_data[iTemp2+iWidth-1]-2*src_data[iTemp2+iWidth]-src_data[iTemp2+iWidth+1];
			pGradxy[iTemp2] = sqrt(pGradx[iTemp2]*pGradx[iTemp2] + pGrady[iTemp2]*pGrady[iTemp2]);
		}
	}
	int iGrayThreshold = 110;
	int iGrayThreshold1 = 115;
	for(i=iWidth+1;i<(iHeight-1)*iWidth-1;i++)
	{
		if(src_data[i] >=iGrayThreshold || src_data[i-1] >iGrayThreshold1 || src_data[i+1] >iGrayThreshold1 
			|| src_data[i+iWidth] >iGrayThreshold1 || src_data[i+iWidth+1] >iGrayThreshold1 || src_data[i+iWidth-1] >iGrayThreshold1 
			|| src_data[i-iWidth] >iGrayThreshold1 || src_data[i-iWidth+1] >iGrayThreshold1 || src_data[i-iWidth-1] >iGrayThreshold1)
		{
			pGradxy[i] =0;
		}
		else if(pGradxy[i] <0.01 || (pGrady[i] >0 && (fabs(pGradx[i])<pGradxy[i]/2)))
		{
			pGradxy[i] =0;
		}
	}
	int *pCutImage2 = new int[iWidth*iHeight];

	int iThreshold1=iGrayThreshold/3;
	for(i=0;i<iWidth*iHeight;i++)
	{
		if(pGradxy[i] !=0)
		{
			if(src_data[i]<iThreshold1)
				pCutImage2[i] = iGrayThreshold - src_data[i]*2;
			else
				pCutImage2[i] = (iGrayThreshold - src_data[i])/2;
		}
		else
			pCutImage2[i] = 0;
	}

	//record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	//fprintf(record_for_irisSegment, "After calculate grad.\r\n");
	//fclose(record_for_irisSegment);

	int iIterativeNum = (iMaxR-iMinR)/2+1;
	pMk = new double[iWidth*iHeight];
	pGk = new double[iWidth*iHeight];
	pOk = new double[iWidth*iHeight*iIterativeNum];
	pFk = new double[iWidth*iHeight*iIterativeNum];	
	pSk = new double[iWidth*iHeight];
	memset(pSk, 0, iWidth*iHeight*sizeof(double));
	memset(pOk, 0, iWidth*iHeight*iIterativeNum*sizeof(double));
	int iXMove,iYMove;
	for(k=iMinR;k<=iMaxR;k=k+2)
	{
		memset(pMk, 0, iWidth*iHeight*sizeof(double));
		memset(pGk, 0, iWidth*iHeight*sizeof(double));
		kk=((k-iMinR)/2)*iWidth*iHeight;
		for(i=1;i<iHeight-1;i++)
		{
			iTemp1=i*iWidth;
			for(j=1;j<iWidth-1;j++)
			{
				iTemp11 = iTemp1+j;
				if(pGradxy[iTemp11] !=0)
				{
					iXMove=j+cvRound(k*pGradx[iTemp11]/pGradxy[iTemp11]);
					iYMove=i+cvRound(k*pGrady[iTemp11]/pGradxy[iTemp11]);
					if(iXMove>=0 && iXMove<iWidth && iYMove>=0 && iYMove<iHeight)
					{
						iTemp2=iYMove*iWidth+iXMove;
						pMk[iTemp2]= pMk[iTemp2]+pGradxy[iTemp11];
						pGk[iTemp2]= pGk[iTemp2]+pCutImage2[iTemp11];
						pOk[kk+iTemp2]= pOk[kk+iTemp2]+1;
					}
				}
			}
		}
		for(i=0;i<iWidth*iHeight;i++)
		{
			pFk[kk+i]=pMk[i]*pGk[i];
			pSk[i]=pSk[i]+pFk[kk+i];
		}
	}

	//record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	//fprintf(record_for_irisSegment, "After calculate Fk and Sk.\r\n");
	//fclose(record_for_irisSegment);

	int iMaxX,iMaxY;
	double dMaxV;
	dMaxV=-1000;
	for(i=4;i<iHeight-4;i++)
	{
		iTemp1=i*iWidth;
		for(j=4;j<iWidth-4;j++)
		{
			if(dMaxV < pSk[iTemp1+j])
			{
				dMaxV = pSk[iTemp1+j];
				iMaxX = j;
				iMaxY = i;
			}
		}
	}
	int iNearPixNum=4; 
	int iFilterLeft,iFilterRight,iFilterTop,iFilterBottom; 
	if(iMaxX-4<iNearPixNum)
	{
		iFilterLeft = 0;
	}
	else
	{
		iFilterLeft = iMaxX-4-iNearPixNum;
	}
	if(iMaxX+4>iWidth-iNearPixNum)
	{
		iFilterRight = iWidth-1;
	}
	else
	{
		iFilterRight = iMaxX+4+iNearPixNum;
	}
	
	if(iMaxY-4<iNearPixNum)
	{
		iFilterTop = 0;
	}
	else
	{
		iFilterTop = iMaxY-4-iNearPixNum;
	}
	if(iMaxY+4>iHeight-iNearPixNum)
	{
		iFilterBottom = iHeight-1;
	}
	else
	{
		iFilterBottom = iMaxY+4+iNearPixNum;
	}

	int iXLen,iYLen;
	iXLen=iFilterRight-iFilterLeft+1;
	iYLen=iFilterBottom-iFilterTop+1;
	double *pSmallFk = new double[iXLen*iYLen];
	double dMaxFk=-1000;
	int iMaxFkX,iMaxFkY;  
	int iMaxFkR; 
	for(k=0;k<iIterativeNum;k++)
	{
		kk=k*iWidth*iHeight;
	
		for(i=iFilterTop;i<=iFilterBottom;i++)
		{
			iTemp1=(i-iFilterTop)*iXLen;
			iTemp2=kk+i*iWidth;
			for(j=iFilterLeft;j<=iFilterRight;j++)
			{
				pSmallFk[iTemp1+(j-iFilterLeft)]=pFk[iTemp2+j];
			}
		}

		for(i=4;i<iYLen-4;i++)
		{
			iTemp1=i*iXLen;
			for(j=4;j<iXLen-4;j++)
			{
				if(dMaxFk < pSmallFk[iTemp1+j])
				{
					dMaxFk = pSmallFk[iTemp1+j];
					iMaxFkX = j;
					iMaxFkY = i;
					iMaxFkR = k; 
				}
			}
		}
	}

	//record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	//fprintf(record_for_irisSegment, "The end.\r\n");
	//fclose(record_for_irisSegment);

	pupil_center.x = (iFilterLeft+iMaxFkX)*times;
	pupil_center.y = (iFilterTop+iMaxFkY)*times;
	pupil_rds = (iMinR+iMaxFkR*2)*times;

	//record_for_irisSegment = fopen("record_for_irisSegment.txt","ab");
	//fprintf(record_for_irisSegment, "pupil_center: (%d, %d), pupil_rds = %d.\r\n\r\n", center.x, center.y, rds);
	//fclose(record_for_irisSegment);

	delete []pCutImage2;
//	delete []pSmallOk;
	delete []pSmallFk;
	delete []pMk;
	delete []pGk;
	delete []pOk;
	delete []pFk;
	delete []pSk;
	delete []pGradx; 
	delete []pGrady; 
	delete []pGradxy; 	

	return 0;
}


