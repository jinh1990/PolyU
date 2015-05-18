#include "my_image_utils.h"
#include "fit_curves.h"

#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\photo\photo.hpp>

#include "iostream"

int SSREnhancement(const cv::gpu::GpuMat src, Mat &dst)
{
	FILE *record_for_GPU;
	
	if (src.empty())
	{
		return -1;
	}

	Mat tmp_img;

	record_for_GPU = fopen("record_for_GPU.txt","ab");
	fprintf(record_for_GPU, "Begin.\r\n");
	fclose(record_for_GPU);
	src.download(tmp_img);
	imwrite("Begin.jpg",tmp_img);

	cv::gpu::GpuMat img32f;

	src.convertTo(img32f, CV_32FC1);

	cv::gpu::GpuMat blurred_res, div_res, log_res, out;
	//Mat log_mat, out;

	record_for_GPU = fopen("record_for_GPU.txt","ab");
	fprintf(record_for_GPU, "Before GaussianBlur.\r\n");
	fclose(record_for_GPU);
	img32f.download(tmp_img);
	imwrite("Before_GaussianBlur.jpg",tmp_img);

	//cv::gpu::GaussianBlur(img32f,blurred_res,Size(img32f.rows-1 | 1, img32f.cols-1 | 1),0.1);
	//cv::gpu::GaussianBlur(img32f,blurred_res,cv::Size(9, 9),0.1);
	cv::gpu::GaussianBlur(img32f,blurred_res,cv::Size(31,31),0.1);

	//cv::gpu::GaussianBlur(img32f, blurred_res, Size(img32f.rows | 1, img32f.cols | 1), min(img32f.rows, img32f.cols), min(img32f.rows, img32f.cols));
	
	record_for_GPU = fopen("record_for_GPU.txt","ab");
	fprintf(record_for_GPU, "Before divide.\r\n");
	fclose(record_for_GPU);
	blurred_res.download(tmp_img);
	imwrite("Before_divide.jpg",tmp_img);

	cv::gpu::divide(img32f, blurred_res, div_res);

	record_for_GPU = fopen("record_for_GPU.txt","ab");
	fprintf(record_for_GPU, "Before log.\r\n");
	fclose(record_for_GPU);
	div_res.download(tmp_img);
	imwrite("Before_log.jpg",tmp_img);
	
	cv::gpu::log(div_res, log_res);

	record_for_GPU = fopen("record_for_GPU.txt","ab");
	fprintf(record_for_GPU, "Before minMaxLoc.\r\n");
	fclose(record_for_GPU);
	log_res.download(tmp_img);
	imwrite("Before_minMaxLoc.jpg",tmp_img);

	double minf, maxf;
	cv::gpu::minMaxLoc(log_res, &minf, &maxf);

	record_for_GPU = fopen("record_for_GPU.txt","ab");
	fprintf(record_for_GPU, "Before Loop.\r\n");
	fclose(record_for_GPU);

	if (maxf - minf > 0.00001)
	{
		double alpha, beta;
		alpha = 255.0 / (maxf - minf);
		beta = -minf * 255.0 / (maxf - minf);
		cv::gpu::multiply(log_res,static_cast<float>(alpha),out);
		cv::gpu::add(out,static_cast<float>(beta),out);
		//out = log_res * static_cast<float>(alpha) + static_cast<float>(beta);
		out.convertTo(out, CV_8UC1);
	}
	else
	{
		//equalizeHist(src, out);
		return -2;
	}

	record_for_GPU = fopen("record_for_GPU.txt","ab");
	fprintf(record_for_GPU, "Before medianBlur.\r\n");
	fclose(record_for_GPU);
	out.download(tmp_img);
	imwrite("Before_medianBlur.jpg",tmp_img);

	Mat dst_gpu;
	out.download(dst_gpu);

	medianBlur(dst_gpu, dst, 3);

	record_for_GPU = fopen("record_for_GPU.txt","ab");
	fprintf(record_for_GPU, "Before return.\r\n");
	fclose(record_for_GPU);

	imwrite("Before_return.jpg",dst);

	return 0;
}

int SSREnhancement(Mat src, Mat &dst)
{
	if (src.empty())
	{
		return -1;
	}

	Mat img32f;
	src.convertTo(img32f, CV_32FC1);

	Mat blurred_res, div_res, log_res, out;
	
	GaussianBlur(img32f, blurred_res, Size(img32f.rows|1, img32f.cols|1), min(img32f.rows, img32f.cols), min(img32f.rows, img32f.cols));
	//GaussianBlur(img32f, blurred_res, Size(5, 5), min(img32f.rows, img32f.cols), min(img32f.rows, img32f.cols));

	divide(img32f, blurred_res, div_res);

	log(div_res, log_res);

	double minf, maxf;
	minMaxLoc(log_res, &minf, &maxf);
	if (maxf - minf > 0.01)
	{
		double alpha, beta;
		alpha = 255.0 / (maxf - minf);
		beta = -minf * 255.0 / (maxf - minf);
		out = log_res * static_cast<float>(alpha) + static_cast<float>(beta);
		out.convertTo(out, CV_8UC1);
	}
	else
	{
		//equalizeHist(src, out);
		return -2;
	}

	medianBlur(out, dst, 3);
	return 0;
}

int StructureMap(const char *subjName, Mat src, float lmbdax, float threshx, Mat &dstx)
{
	Mat xdif, ydif;
	
	float std = 5;
	//must be odd
	int width = static_cast<int>(std * 2) + 1;

	Mat g(width, width, CV_32FC1), gx(width, width, CV_32FC1), gy(width, width, CV_32FC1);
	for (int i = 0; i < g.rows; i++)
	{
		for (int j = 0; j < g.cols; j++)
		{
			gx.at<float>(i, j) = static_cast<float>(j);
			gy.at<float>(i, j) = static_cast<float>(i);
		}
	}

	pow((gx - static_cast<float>(width / 2)), 2, gx);
	pow((gy - static_cast<float>(width / 2)), 2, gy);

	g = (gx + gy) / static_cast<float>(2 * pow(width / 2, 2));

	Mat allzero = Mat::zeros(g.rows, g.cols, CV_32FC1);
	
	subtract(allzero, g, g);
	
	exp(g, g);

	Sobel(src, xdif, CV_32F, 1, 0);
	Sobel(src, ydif, CV_32F, 0, 1);
	


	Mat Lx(xdif.rows, xdif.cols, CV_32FC1);
	Mat Ly(ydif.rows, ydif.cols, CV_32FC1);

	//convolute(xdif, g, Lx);

	//convolute(ydif, g, Ly);

	filter2D(xdif, Lx, -1, g);
	filter2D(ydif, Ly, -1, g);


	Mat dst0 = abs(Lx) * lmbdax + abs(Ly);

	Rect roi_rect(dst0.cols / 4, dst0.rows / 4, dst0.cols * 2 / 4, dst0.rows * 2 / 4);

	Mat roi(dst0, roi_rect);

	double mind = 0, maxd = 0;
	minMaxLoc(roi, &mind, &maxd, NULL, NULL);

	dst0 /= maxd;

	threshold(dst0, dst0, threshx, 255, THRESH_BINARY);
	dst0.convertTo(dst0, CV_8U);

	imwrite(string(subjName) + "-tv-nothin.jpg", dst0);

	thin2(dst0, dst0);
	scale255(dst0, dst0);
	dstx = dst0;

	//////

	//dst0 = abs(Lx)  + abs(Ly) * lmbday;
	//scale255(dst0, dst0);
	//threshold(dst0, dst0, 0.25 * 255, 255, THRESH_BINARY);
	//thin2(dst0, dst0);
	//scale255(dst0, dst0);
	//dsty = dst0;

	return 0;
}

int scale255(Mat src, Mat &dst)
{
	if (src.empty())
	{
		return -1;
	}

	Mat scaled;
	if (src.type() != CV_32FC1)
	{
		src.convertTo(scaled, CV_32FC1);
	}
	else
	{
		src.copyTo(scaled);
	}

	double minf, maxf;
	minMaxLoc(scaled, &minf, &maxf);
	if (maxf - minf > 0.00001)
	{
		double alpha, beta;
		alpha = 255.0 / (maxf - minf);
		beta = -minf * 255.0 / (maxf - minf);
		scaled = scaled * static_cast<float>(alpha) + static_cast<float>(beta);
		scaled.convertTo(dst, CV_8UC1);
	}
	else
	{
		//dst= src.clone();
		return -2;
	}

	return 0;
}

int convolute(Mat src, Mat mask, Mat &dst)
{
	if (src.empty() || src.type() != CV_32FC1 || mask.empty() || mask.type() != CV_32FC1
		|| (mask.rows & 1) == 0 || (mask.cols & 1) == 0)
	{
		return -1;
	}

	dst = Mat::zeros(src.rows, dst.cols, src.type());

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int half_width = mask.cols / 2;
			int half_height = mask.rows / 2;

			float val = 0;
			for (int h = -half_height; h <= half_height; h++)
			{
				for (int k = -half_width; k <= half_width; k++)
				{
					int srccol = max(0, j + k);
                    srccol = min(srccol, src.cols - 1);

					int srcrow = max(0, i + h);
					srcrow = min(srcrow, src.rows - 1);

					int mcol = k + half_width;
					int mrow = h + half_height;

					val += src.at<float>(srcrow, srccol) + mask.at<float>(mcol, mrow);
				}
			}

			dst.at<float>(i, j) = val;
		}
	}

	return 0;
}

void ThinSubiteration1(Mat & pSrc, Mat & pDst) {
        int rows = pSrc.rows;
        int cols = pSrc.cols;
        pSrc.copyTo(pDst);
        for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                        if(pSrc.at<float>(i, j) == 1.0f) {
                                /// get 8 neighbors
                                /// calculate C(p)
                                int neighbor0 = (int) pSrc.at<float>( i-1, j-1);
                                int neighbor1 = (int) pSrc.at<float>( i-1, j);
                                int neighbor2 = (int) pSrc.at<float>( i-1, j+1);
                                int neighbor3 = (int) pSrc.at<float>( i, j+1);
                                int neighbor4 = (int) pSrc.at<float>( i+1, j+1);
                                int neighbor5 = (int) pSrc.at<float>( i+1, j);
                                int neighbor6 = (int) pSrc.at<float>( i+1, j-1);
                                int neighbor7 = (int) pSrc.at<float>( i, j-1);
                                int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                                                 int(~neighbor3 & ( neighbor4 | neighbor5)) +
                                                 int(~neighbor5 & ( neighbor6 | neighbor7)) +
                                                 int(~neighbor7 & ( neighbor0 | neighbor1));
                                if(C == 1) {
                                        /// calculate N
                                        int N1 = int(neighbor0 | neighbor1) +
                                                         int(neighbor2 | neighbor3) +
                                                         int(neighbor4 | neighbor5) +
                                                         int(neighbor6 | neighbor7);
                                        int N2 = int(neighbor1 | neighbor2) +
                                                         int(neighbor3 | neighbor4) +
                                                         int(neighbor5 | neighbor6) +
                                                         int(neighbor7 | neighbor0);
                                        int N = min(N1,N2);
                                        if ((N == 2) || (N == 3)) {
                                                /// calculate criteria 3
                                                int c3 = ( neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
                                                if(c3 == 0) {
                                                        pDst.at<float>( i, j) = 0.0f;
                                                }
                                        }
                                }
                        }
                }
        }
}

void ThinSubiteration2(Mat & pSrc, Mat & pDst) {
        int rows = pSrc.rows;
        int cols = pSrc.cols;
        pSrc.copyTo( pDst);
        for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                        if (pSrc.at<float>( i, j) == 1.0f) {
                                /// get 8 neighbors
                                /// calculate C(p)
                            int neighbor0 = (int) pSrc.at<float>( i-1, j-1);
                            int neighbor1 = (int) pSrc.at<float>( i-1, j);
                            int neighbor2 = (int) pSrc.at<float>( i-1, j+1);
                            int neighbor3 = (int) pSrc.at<float>( i, j+1);
                            int neighbor4 = (int) pSrc.at<float>( i+1, j+1);
                            int neighbor5 = (int) pSrc.at<float>( i+1, j);
                            int neighbor6 = (int) pSrc.at<float>( i+1, j-1);
                            int neighbor7 = (int) pSrc.at<float>( i, j-1);
                                int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                                        int(~neighbor3 & ( neighbor4 | neighbor5)) +
                                        int(~neighbor5 & ( neighbor6 | neighbor7)) +
                                        int(~neighbor7 & ( neighbor0 | neighbor1));
                                if(C == 1) {
                                        /// calculate N
                                        int N1 = int(neighbor0 | neighbor1) +
                                                int(neighbor2 | neighbor3) +
                                                int(neighbor4 | neighbor5) +
                                                int(neighbor6 | neighbor7);
                                        int N2 = int(neighbor1 | neighbor2) +
                                                int(neighbor3 | neighbor4) +
                                                int(neighbor5 | neighbor6) +
                                                int(neighbor7 | neighbor0);
                                        int N = min(N1,N2);
                                        if((N == 2) || (N == 3)) {
                                                int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
                                                if(E == 0) {
                                                        pDst.at<float>(i, j) = 0.0f;
                                                }
                                        }
                                }
                        }
                }
        }
}

void normalizeLetter(Mat & inputarray, Mat & outputarray) {
        bool bDone = false;
        int rows = inputarray.rows;
        int cols = inputarray.cols;

        inputarray.convertTo(inputarray,CV_32FC1);

        inputarray.copyTo(outputarray);

        //outputarray.convertTo(outputarray,CV_32FC1);

        /// pad source
        Mat p_enlarged_src = Mat(rows + 2, cols + 2, CV_32FC1);
        for(int i = 0; i < (rows+2); i++) {
            p_enlarged_src.at<float>(i, 0) = 0.0f;
            p_enlarged_src.at<float>( i, cols+1) = 0.0f;
        }
        for(int j = 0; j < (cols+2); j++) {
                p_enlarged_src.at<float>(0, j) = 0.0f;
                p_enlarged_src.at<float>(rows+1, j) = 0.0f;
        }
        for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                        if (inputarray.at<float>(i, j) >= 20.0f) {
                                p_enlarged_src.at<float>( i+1, j+1) = 1.0f;
                        }
                        else
                                p_enlarged_src.at<float>( i+1, j+1) = 0.0f;
                }
        }

        /// start to thin
        Mat p_thinMat1 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
        Mat p_thinMat2 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
        Mat p_cmp = Mat::zeros(rows + 2, cols + 2, CV_8UC1);

        while (bDone != true) {
                /// sub-iteration 1
                ThinSubiteration1(p_enlarged_src, p_thinMat1);
                /// sub-iteration 2
                ThinSubiteration2(p_thinMat1, p_thinMat2);
                /// compare
                compare(p_enlarged_src, p_thinMat2, p_cmp, CV_CMP_EQ);
                /// check
                int num_non_zero = countNonZero(p_cmp);
                if(num_non_zero == (rows + 2) * (cols + 2)) {
                        bDone = true;
                }
                /// copy
                p_thinMat2.copyTo(p_enlarged_src);
        }
        // copy result
        for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                        outputarray.at<float>( i, j) = p_enlarged_src.at<float>( i+1, j+1);
                }
        }
}

int thin2(Mat src, Mat &dst)
{
	Mat src2 = src.clone();
	normalizeLetter(src2, dst);
	return 0;
}

int thin(Mat src, Mat &dst)
{
	Mat skel(src.size(), CV_8UC1, cv::Scalar(0));
	Mat img = src.clone();

	Mat temp;
	Mat eroded;
	Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
 
	bool done;		
	do
	{
	  cv::erode(img, eroded, element);
	  cv::dilate(eroded, temp, element); // temp = open(img)
	  cv::subtract(img, temp, temp);
	  cv::bitwise_or(skel, temp, skel);
	  eroded.copyTo(img);
 
	  done = (cv::countNonZero(img) == 0);
	} while (!done);

	dst = skel;

	return 0;
}

int calc_threshold(Mat src, float low_percents, float high_percents, int &low_thresh, int &high_thresh)
{
	int bins = 256;
	int histSize[] = {bins};
	float range[] = {0, 256};
	const float *ranges[] = {range};

	Mat hist;
	int dims = 1;
	int channels[] = {0};

	calcHist(&src, 1, channels, Mat(), hist, dims, histSize, ranges, true, false);

	int total = src.rows * src.cols;
	int low_part = cvRound(low_percents * total);
	int high_part = cvRound((1.0 - high_percents) * total);

	int curr_count = 0;
	int high_thresh_candidate = 256;
	while(curr_count < high_part && high_thresh_candidate > 0)
	{
		high_thresh_candidate--;
		curr_count += cvRound(hist.at<float>(high_thresh_candidate));
	}

	high_thresh = high_thresh_candidate + 1;
	curr_count = 0;
	int low_thresh_candiate = 0;
	while(curr_count < low_part && low_thresh_candiate < 255)
	{
		low_thresh_candiate++;
		curr_count += cvRound(hist.at<float>(low_thresh_candiate));
	}
	low_thresh = low_thresh_candiate - 1;

	return 0;
}

void keep_small_blocks(cv::Mat& im, double size)
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
        if (area > 0 && area >= size)
            cv::drawContours(im, contours, i, CV_RGB(0,0,0), -1);
    }
}

int reflection_removal(Mat src, Mat &mask, Mat &dest)
{
	int low_thresh = 0, high_thresh = 255;
	
	calc_threshold(src, 0, 0.9f, low_thresh, high_thresh);

	Mat mask0;
	threshold(src, mask0, high_thresh, 255, THRESH_BINARY); 

	keep_small_blocks(mask0, 10000);

	Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
	dilate(mask0, mask0, element);

	inpaint(src, mask0, dest, 3, INPAINT_TELEA);
	mask = 255 - mask0;

	return 0;
}

int resize_small(Mat src, Mat &dst)
{
	int iWidth = src.cols;
	int iHeight = src.rows;

	int small_iWidth = dst.cols;
	int small_iHeight = dst.rows;

	int w_times = iWidth/small_iWidth;
	int h_times = iHeight/small_iHeight;

	int i, j;

	for(i = 0; i < small_iHeight; i++)
		for(j = 0; j < small_iWidth; j++)
		{
			dst.data[i*small_iWidth+j] = src.data[i*h_times*iWidth+j*w_times];
		}
	
	return  0;
}

int LeastSquares(vector<cv::Point> points, cv::Point& center, int& rds)
{
	int i;
	int m_nNum = points.size();

     double X1=0;
     double Y1=0;
     double X2=0;
     double Y2=0;
     double X3=0;
     double Y3=0;
     double X1Y1=0;
     double X1Y2=0;
     double X2Y1=0;

     for (i=0;i<m_nNum;i++)
     {
		 X1 = X1 + points[i].x;
         Y1 = Y1 + points[i].y;
         X2 = X2 + points[i].x*points[i].x;
         Y2 = Y2 + points[i].y*points[i].y;
         X3 = X3 + points[i].x*points[i].x*points[i].x;
         Y3 = Y3 + points[i].y*points[i].y*points[i].y;
         X1Y1 = X1Y1 + points[i].x*points[i].y;
         X1Y2 = X1Y2 + points[i].x*points[i].y*points[i].y;
         X2Y1 = X2Y1 + points[i].x*points[i].x*points[i].y;
     }

     double C,D,E,G,H,N;
     double a,b,c;
     N = m_nNum;
     C = N*X2 - X1*X1;
     D = N*X1Y1 - X1*Y1;
     E = N*X3 + N*X1Y2 - (X2+Y2)*X1;
     G = N*Y2 - Y1*Y1;
     H = N*X2Y1 + N*Y3 - (X2+Y2)*Y1;
     a = (H*D-E*G)/(C*G-D*D);
     b = (H*C-E*D)/(D*D-G*C);
     c = -(a*X1 + b*Y1 + X2 + Y2)/N;

	 center.x = a/(-2);
     center.y = b/(-2);
     rds = sqrt(a*a+b*b-4*c)/2;

	 return 0;
}

int interp2(Mat src, double* xo, double* yo, int width, int height, Mat& dst)
{
	int i, j;
	int ndx;
	double s,t;

	for(i = 0; i < height; i++)
		for(j = 0; j < width; j++)
		{
			int tmp = i*width+j;
			if(xo[tmp] < 0 || xo[tmp] > src.cols-1 || yo[tmp] < 0 || yo[tmp] > src.rows-1)
			{
				dst.data[tmp] = 0;
			}
			else
			{
				ndx = (int)((floor(xo[tmp])-1) + (floor(yo[tmp])-1)*src.cols);
				s = xo[tmp] - floor(xo[tmp]);
				t = yo[tmp] - floor(yo[tmp]);

				dst.data[tmp] = (int)(src.data[ndx]*(1-t)+src.data[ndx+src.cols]*t)*(1-s)+(src.data[ndx+1]*(1-t)+src.data[ndx+src.cols+1]*t)*s;
			}
		}

	return 0;
}

void findCenter(Mat edgemap, int* radiusRange, int points_gap, Point &center_point)
{
	vector<vector<Point> > contours;
	findContours(edgemap, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	int num_of_lines = contours.size();

	Mat votes_left = Mat::ones(edgemap.rows, edgemap.cols, CV_8UC1);
	Mat votes_right = Mat::ones(edgemap.rows, edgemap.cols, CV_8UC1);
	Mat mark_mat = Mat::zeros(edgemap.rows, edgemap.cols, CV_8UC1);

	for (int i = 0; i < num_of_lines; i++)
	{
		vector<Point> Idx = contours[i];
		int len = Idx.size();
		if (len < 10)
		{
			continue;
		}
		for (int j = points_gap; j < len - points_gap; j++)
		{
			int y = Idx[j].y;
			int x = Idx[j].x;
			if (mark_mat.at<unsigned char>(y,x) != 0)
			{
				continue;
			}
			mark_mat.at<unsigned char>(y,x) = 1;
			Point P1 = Idx[j-points_gap];
			Point P2 = Idx[j+points_gap];
			if (P1.x == P2.x)
			{
				if (P1.y == P2.y)
				{
					continue;
				}
				else if (P1.y > P2.y)
				{
					if (x < P1.x)
					{
						Point tmp = P1;
						P1 = P2;
						P2 = tmp;
					}
				}
			}
			else
			{
				if (P1.x > P2.x)
				{
					Point tmp = P1;
					P1 = P2;
					P2 = tmp;
				}
				double slope1 = (y-P1.y)/(x-P1.x+INF);
				double slope2 = (P1.y-P2.y)/(P1.x-P2.x+INF);
				if (slope1 <= slope2)
				{
					continue;
				}
			}
			double dx = P2.x - P1.x;
			double dy = P2.y - P1.y;
			double distance = sqrt(dx*dx + dy*dy);
			double cosine = dx/distance;
			double sine = dy/distance;
			double cosine2 = sine;
			double sine2 = -cosine;
			if (cosine2 > 0.15)
			{
				for (int r = radiusRange[0]; r < radiusRange[1]; r++)
				{
					int xc = (int)(x + cosine2 * r + 0.5);
					int yc = (int)(y + sine2 * r + 0.5);
					if (xc < 1 || xc > edgemap.cols || yc < 1 || yc > edgemap.rows)
					{
						break;
					}
					else
					{
						votes_left.at<unsigned char>(yc,xc)++;
					}
				}
			}
			else if (cosine2 < -0.15)
			{
				for (int r = radiusRange[0]; r < radiusRange[1]; r++)
				{
					int xc = (int)(x + cosine2 * r + 0.5);
					int yc = (int)(y + sine2 * r + 0.5);
					if (xc < 1 || xc > edgemap.cols || yc < 1 || yc > edgemap.rows)
					{
						break;
					}
					else
					{
						votes_right.at<unsigned char>(yc,xc)++;
					}
				}
			}
		}
	}

	blur(votes_left, votes_left, cv::Size(3,3));
	blur(votes_right, votes_right, cv::Size(3,3));

	Mat votes = Mat::zeros(votes_left.size(),CV_8UC1);
	int max_value = 0;
	//Point max_position(0,0);
	for (int i = 0; i < votes.rows; i++)
	{
		for (int j = 0; j < votes.cols; j++)
		{
			votes.at<unsigned char>(i,j) = votes_left.at<unsigned char>(i,j)*votes_right.at<unsigned char>(i,j) - 1;
			if (max_value < votes.at<unsigned char>(i,j))
			{
				max_value = votes.at<unsigned char>(i,j);
				center_point.x = j;
				center_point.y = i;
			}
		}
	}

	//for (int i = 0; i < votes.rows; i++)
	//{
	//	for (int j = 0; j < votes.cols; j++)
	//	{
	//		votes.at<unsigned char>(i,j) = votes.at<unsigned char>(i,j)/max_value;
	//	}
	//}

	//Mat imvotes = votes.clone();

	//for(int i = 0; i < imvotes.rows; i++)
	//	for(int j = 0; j < imvotes.cols; j++)
	//	{
	//		if (edgemap.at<unsigned char>(i,j) != 0)
	//		{
	//			imvotes.at<unsigned char>(i,j) = 255;
	//		}
	//	}

	//if (max_position.x > 3 && max_position.y > 3)
	//{
	//	for (int i = -3; i < 4; i++)
	//	{
	//		for (int j = -3; j < 4; j++)
	//		{
	//			imvotes.at<unsigned char>(max_position.y+i,max_position.x+j) = 255;
	//		}
	//	}
	//}
	

	//cout<<"max_position.x = "<<max_position.x<<endl;
	//cout<<"max_position.y = "<<max_position.y<<endl;
}

void find_radius(Mat edgemap, Point center_point, int* radiusRange, int searchRange, Point& output_center, int& output_radius)
{
	int upper_bound = 1>(center_point.y - searchRange - 2)?1:(center_point.y - searchRange - 2);
	for (int i = 0; i < upper_bound; i++)
	{
		for (int j = 0; j < edgemap.cols; j++)
		{
			edgemap.at<unsigned char>(i,j) = 0;
		}		
	}
	Point center_output(0,0);
	int rds_outpoint = 0;
	int grade = 0;

	vector<int> X, Y;

	for (int i = 0; i < edgemap.rows; i++)
	{
		for (int j = 0; j < edgemap.cols; j++)
		{
			if (edgemap.at<unsigned char>(i,j) != 0)
			{
				Y.push_back(i);
				X.push_back(j);
			}
		}
	}

	int num = X.size();

	int* r = new int[num];
	memset(r,0,sizeof(int)*num);

	int len = radiusRange[1] - radiusRange[0] + 3;
	int* x_axis = new int[1000];
	int* y_axis = new int[1000];

	memset(x_axis,0,sizeof(int)*len);			
	memset(y_axis,0,sizeof(int)*len);

	for (int i = center_point.x - searchRange; i < center_point.x + searchRange; i++)
	{
		for (int j = center_point.y - searchRange; j < center_point.y + searchRange; j++)
		{
			//memset(r,0,sizeof(int)*num);
			for (int k = 0; k < num; k++)
			{
				r[k] = sqrt((i-(float)X[k])*(i-(float)X[k]) + (j-(float)Y[k])*(j-(float)Y[k]));
				//int index = (r[k] - radiusRange[0]) >= 0?(r[k] - radiusRange[0]):0;
				//index = (index<radiusRange[1])?index:(radiusRange[1]-1);
				//y_axis[index]++;
			}
		
			memset(x_axis,0,sizeof(int)*len);			
			memset(y_axis,0,sizeof(int)*len);
			float gap = (((float)radiusRange[1] - radiusRange[0] + 2)/(len-1)) == 0?INF:(((float)radiusRange[1] - radiusRange[0] + 2)/(len-1));
			x_axis[0] = radiusRange[0] - 1;
			x_axis[len-1] = radiusRange[1] + 1;
			for(int m = 1; m < len-1; m++)
			{
				x_axis[m] = x_axis[m-1] + gap;
			}
			for (int m = 0; m < num; m++)
			{
				int idex = (((int)r[m] - (int)x_axis[0])/gap)>(len-1)?(len-1):(((int)r[m] - (int)x_axis[0])/gap);
				y_axis[idex]++;
			}
			y_axis[0] = 0;
			y_axis[len-1] = 0;

			int grade_cur = 0;
			int index = 0;
			for (int m = 1; m < len-1; m++)
			{
				y_axis[m] = (y_axis[m] + y_axis[m-1] + y_axis[m+1])/3;
				if (grade_cur < y_axis[m])
				{
					grade_cur = y_axis[m];
					index = m;
				}
			}
 			if (grade_cur > grade)
			{
				output_radius = radiusRange[0] + index -2;
				output_center.x = i;
				output_center.y = j;
				grade = grade_cur;
			}
		}
	}

	delete[] r;
	//delete[] x_axis;
	//delete[] y_axis;
}

void find_pupil_radius(Mat edgemap, Point center_point, int* radiusRange, int searchRange, Point& output_center, int& output_radius)
{

	Point center_output(0,0);
	int rds_outpoint = 0;
	int grade = 0;

	vector<int> X, Y;

	for (int i = 0; i < edgemap.rows; i++)
	{
		for (int j = 0; j < edgemap.cols; j++)
		{
			if (edgemap.at<unsigned char>(i,j) != 0)
			{
				Y.push_back(i);
				X.push_back(j);
			}
		}
	}

	int num = X.size();

	int* r = new int[num];
	memset(r,0,sizeof(int)*num);

	int len = radiusRange[1] - radiusRange[0] + 3;
	int* x_axis = new int[len];
	int* y_axis = new int[len];

	memset(x_axis,0,sizeof(int)*len);			
	memset(y_axis,0,sizeof(int)*len);

	for (int i = center_point.x - searchRange; i < center_point.x + searchRange; i++)
	{
		for (int j = center_point.y - searchRange; j < center_point.y + searchRange; j++)
		{
			//memset(r,0,sizeof(int)*num);
			for (int k = 0; k < num; k++)
			{
				r[k] = sqrt((i-(float)X[k])*(i-(float)X[k]) + (j-(float)Y[k])*(j-(float)Y[k]));
				//int index = (r[k] - radiusRange[0]) >= 0?(r[k] - radiusRange[0]):0;
				//index = (index<radiusRange[1])?index:(radiusRange[1]-1);
				//y_axis[index]++;
			}
		
			memset(x_axis,0,sizeof(int)*len);			
			memset(y_axis,0,sizeof(int)*len);

			float gap = (((float)radiusRange[1] - radiusRange[0] + 2)/(len-1)) == 0?INF:(((float)radiusRange[1] - radiusRange[0] + 2)/(len-1));
			x_axis[0] = radiusRange[0] - 1;
			x_axis[len-1] = radiusRange[1] + 1;
			for(int m = 1; m < len-1; m++)
			{
				x_axis[m] = x_axis[m-1] + gap;
			}
			for (int m = 0; m < num; m++)
			{
				int idex = (((int)r[m] - (int)x_axis[0])/gap)>(len-1)?(len-1):(((int)r[m] - (int)x_axis[0])/gap);
				y_axis[idex]++;
			}
			y_axis[0] = 0;
			y_axis[len-1] = 0;

			int grade_cur = 0;
			int index = 0;
			for (int m = 1; m < len-1; m++)
			{
				y_axis[m] = (y_axis[m] + y_axis[m-1] + y_axis[m+1])/3;
				if (grade_cur < y_axis[m])
				{
					grade_cur = y_axis[m];
					index = m;
				}
			}

			if (grade_cur > grade)
			{
				output_radius = radiusRange[0] + index -2;
				output_center.x = i;
				output_center.y = j;
				grade = grade_cur;
			}
		}
	}

	cout<<"output_center = "<<output_center<<endl;
	cout<<"output_radius = "<<output_radius<<endl;

	delete[] r;
	delete[] x_axis;
	delete[] y_axis;
}

void mask_lower_region(Mat img, Point center, int radius, double* extend, Mat& mask, double* thresh_high, double* thresh_low, double& cir_correct)
{
	int cols = img.cols;
	int rows = img.rows;

	double angles[4] = {0, 0.25*PI, 0.75*PI, PI};

	//Mat thresh_high = Mat::zeros(3,1,CV_8UC1);
	//Mat thresh_low = Mat::zeros(3,1,CV_8UC1);
	//Mat quality = Mat::zeros(3,1,CV_8UC1);
	//double thresh_high[3] = {0,0,0};
	//double thresh_low[3] = {0,0,0};
	double quality[3] = {0,0,0};

	double r_range[2] = {extend[0]*radius,extend[1]*radius};

	cout<<"1"<<endl;
	double angles1[2] = {angles[0],angles[1]};
	thresh_angle_range(img,center,r_range,angles1,thresh_high[0],thresh_low[0],quality[0]);

	cout<<"2"<<endl;
	double angles2[2] = {angles[1],angles[2]};
	thresh_angle_range(img,center,r_range,angles2,thresh_high[1],thresh_low[1],quality[1]);

	cout<<"3"<<endl;
	double angles3[2] = {angles[2],angles[3]};
	thresh_angle_range(img,center,r_range,angles3,thresh_high[2],thresh_low[2],quality[2]);


	for (int i = 0; i < 3; i++)
	{
		double angles0[2] = {angles[i],angles[i+1]};
		double radius_ranges[2] = {0,1};
		segment_angle_range(img,mask,center,radius,angles0,thresh_high[i],radius_ranges);
	}
}

void mask_upper_region(Mat img, Mat& mask, Point center, int radius, double* thresh)
{
	double angles1[2] = {-0.5*PI, 0};
	double range[2] = {0,1};
	segment_angle_range(img,mask,center,radius,angles1,thresh[0],range);

	double angles2[2] = {PI, -0.4*PI};
	segment_angle_range(img,mask,center,radius,angles2,thresh[2],range);
}

void get_reflection_region( Mat img, Mat region, double* thresh )
{
	double max_thresh = 0;

	for (int i = 0; i < 3; i++)
	{
		if (max_thresh < thresh[i])
		{
			max_thresh = thresh[i];
		}
	}

	region = Mat::zeros(img.size(),CV_8UC1);
	
	threshold( img, region, max_thresh, 255,CV_THRESH_BINARY);

	int dilation_type = MORPH_ELLIPSE;
	int dilation_size = 7;

	Mat element = getStructuringElement( MORPH_ELLIPSE, Size(15,15));

  /// Apply the dilation operation
	dilate( region, region, element );
}
 
void thresh_angle_range(Mat img, Point center, double* radius_range, double* angles, double& thresh, double& thresh_low, double& quality)
{
	cout<<"1.1"<<endl;
	Mat region;
	int img_size[2] = {img.rows,img.cols};
	get_sector_region(center,radius_range,angles,img_size,region);
	cvNamedWindow( "better_result", 1 );
	imshow("better_result", region);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );
	cout<<"1.2"<<endl;
	Mat masked_img = Mat::zeros(img.size(),CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (region.at<unsigned char>(i,j) != 0)
			{
				masked_img.at<unsigned char>(i,j) = img.at<unsigned char>(i,j);
			}
		}
	}
	cout<<"1.3"<<endl;
	cal_hist_thresh(masked_img,thresh,thresh_low,quality);
	cout<<"1.4"<<endl;
}

void get_sector_region(Point center, double* radii, double* angles, int* size, Mat& region)
{
	Mat X = Mat::zeros(size[0],size[1],CV_8UC1);
	Mat Y = Mat::zeros(size[0],size[1],CV_8UC1);
	meshgrid(Range((1-center.x), (size[1] - center.x)),Range((1-center.y),(size[0]-center.y)),X,Y);

	Mat D = Mat::zeros(X.size(),CV_32FC1);

	for (int i = 0; i < X.rows; i++)
	{
		for (int j = 0; j < X.cols; j++)
		{
			D.ptr<float>(i)[j] = sqrt(X.ptr<int>(i)[j]*X.ptr<int>(i)[j] + Y.ptr<int>(i)[j]*Y.ptr<int>(i)[j]);
		}
	}
	Mat region1 = Mat::zeros(X.size(),CV_8UC1);
	Mat region2 = Mat::zeros(X.size(),CV_8UC1);
	region = Mat::zeros(X.size(),CV_8UC1);
	for (int i = 0; i < X.rows; i++)
	{
		for (int j = 0; j < X.cols; j++)
		{
			if (angles[0] <= PI/2)
			{
				region1.ptr<unsigned char>(i)[j] = Y.ptr<int>(i)[j] >= tan(angles[0])*X.ptr<int>(i)[j]?255:0;
			}
			else
			{
				region1.ptr<unsigned char>(i)[j] = Y.ptr<int>(i)[j] <= tan(angles[0])*X.ptr<int>(i)[j]?255:0;
			}
			if (angles[1] <= PI/2)
			{
				region2.ptr<unsigned char>(i)[j] = Y.ptr<int>(i)[j] <= tan(angles[1])*X.ptr<int>(i)[j]?255:0;
			}
			else
			{
				region2.ptr<unsigned char>(i)[j] = Y.ptr<int>(i)[j] >= tan(angles[1])*X.ptr<int>(i)[j]?255:0;
			}
			region.ptr<unsigned char>(i)[j] = (region1.ptr<unsigned char>(i)[j]&&region2.ptr<unsigned char>(i)[j]&&(D.ptr<float>(i)[j]>=radii[0])&&(D.ptr<float>(i)[j]<=radii[1]))?255:0;
		}
	}	
}

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)  
{  
    std::vector<int> t_x, t_y;  
    for(int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);  
    for(int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);  
  
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);  
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);  
} 

void cal_hist_thresh( Mat masked_img, double& thresh_high, double& thresh_low, double& quality)
{
	int threshold_value = 0;
	int threshold_type = 3;;
	int const max_value = 255;
	int const max_type = 4;
	int const max_BINARY_value = 255;

	const int channels[1]={0};
    const int histSize[1]={256};
    float hranges[2]={0,255};
    const float* ranges[1]={hranges};
	Mat hist;
    calcHist(&masked_img,1,channels,Mat(),hist,1,histSize,ranges);
	Mat bw_img;
	int tmp_thresh;
	int maxVal;
	threshold(masked_img,bw_img,thresh_high,max_BINARY_value,threshold_type);
	if (thresh_high == 0)
	{
		thresh_low = 0.2;
		thresh_high = 0.6;
		int idx = (int)(thresh_high*255+0.5);
		double sum_hist_low = 0, sum_hist_high = 0;
		for (int i = 0; i < 256; i++)
		{
			if (i < idx)
			{
				sum_hist_low += hist.ptr<float>(i)[1];
			}
			else
			{
				sum_hist_high += hist.ptr<float>(i)[1];
			}
		}

		quality = (double)sum_hist_low/sum_hist_high;
		if (quality > 1)
		{
			quality = 1/quality;
		}
		return;
	}
	thresh_high = (int)(thresh_high*255+0.5);

	double thresh_max;
	minMaxIdx(masked_img,&thresh_low,&thresh_max);
	double sum_hist_low = 0, sum_hist_high = 0;
	for (int i = 0; i < 256; i++)
	{
		if (i < thresh_high)
		{
			sum_hist_low += hist.ptr<float>(i)[1];
		}
		else
		{
			sum_hist_high += hist.ptr<float>(i)[1];
		}
	}
	quality = (double)sum_hist_low/sum_hist_high;
	if (quality > 1)
	{
		quality = 1/quality;
	}

	thresh_high = (double)thresh_high;
	thresh_low = (double)thresh_low;	
}

void segment_angle_range(Mat img, Mat& mask, Point center, int radius, double* angles, double thresh, double* radius_range)
{
	Mat region;
	double radii[2] = {radius*radius_range[0],radius*radius_range[1]};
	int img_size[2] = {img.rows,img.cols};
	get_sector_region(center,radii,angles,img_size,region);

	Mat bw_img(img.size(),CV_8UC1);
	double nnz_region = 0, nnz_bw = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (region.ptr<unsigned char>(i)[j] != 0)
			{
				nnz_region++;
				if (img.ptr<unsigned char>(i)[j] >= thresh*255)
				{
					nnz_bw++;
					bw_img.ptr<unsigned char>(i)[j] = 0;
				}
				else
				{
					bw_img.ptr<unsigned char>(i)[j] = 255;
				}		
			}
			else
			{
				bw_img.ptr<unsigned char>(i)[j] = 255;
			}
		}
	}

	if (nnz_bw < 0.2*nnz_region)
	{
		for (int i = 0; i < mask.rows; i++)
		{
			for (int j = 0; j < mask.cols; j++)
			{
				if (region.ptr<unsigned char>(i)[j] != 0)
				{
					mask.ptr<unsigned char>(i)[j] = 255;
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < mask.rows; i++)
		{
			for (int j = 0; j < mask.cols; j++)
			{
				if (region.ptr<unsigned char>(i)[j] != 0)
				{
					mask.ptr<unsigned char>(i)[j] = bw_img.ptr<unsigned char>(i)[j];
				}
			}
		}
	}
}

void get_pupil_region( Mat img, Mat reflection, Mat& region, Point center, int radius, double* thresh)
{
	double reflection_thresh = 0.6;

	int rows = img.rows;
	int cols = img.cols;

	Mat X = Mat::zeros(rows,cols,CV_8UC1);
	Mat Y = Mat::zeros(rows,cols,CV_8UC1);
	meshgrid(Range((1-center.x), (cols - center.x)),Range((1-center.y),(rows-center.y)),X,Y);

	Mat circle_mask = Mat::zeros(X.size(),CV_32FC1);

	double circle_rds = (radius-2)*(radius-2);
	for (int i = 0; i < X.rows; i++)
	{
		for (int j = 0; j < X.cols; j++)
		{
			if ((X.ptr<int>(i)[j]*X.ptr<int>(i)[j] + Y.ptr<int>(i)[j]*Y.ptr<int>(i)[j]) < circle_rds)
			{
				circle_mask.ptr<float>(i)[j] = 255;
			}
			else
			{
				circle_mask.ptr<float>(i)[j] = 0;
			}
		}
	}

	double min_thresh = 255;
	for (int i = 0; i < 3; i++)
	{
		if (min_thresh > thresh[i])
		{
			min_thresh = thresh[i];
		}
	}

	Mat bw_img;
	threshold(img, bw_img, min_thresh, 255, CV_THRESH_BINARY);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (bw_img.ptr<unsigned char>(i)[j] == 0 && circle_mask.ptr<unsigned char>(i)[j] != 0)
			{
				region.ptr<unsigned char>(i)[j] = 255;
			}
			else
			{
				region.ptr<unsigned char>(i)[j] = 0;
			}
		}
	}
}

int fit_lower_eyelid( Mat img, Point iris_center, int iris_rds, Point pupil_center, int pupil_rds, double* range, Mat& coffs)
{
	int score_thresh = 3;

	int rows = img.rows;
	int cols = img.cols;

	Mat Xmap = Mat::zeros(rows,cols,CV_8UC1);
	Mat Ymap = Mat::zeros(rows,cols,CV_8UC1);
	meshgrid(Range(1, cols),Range(1,rows),Xmap,Ymap);

	cout<<"1"<<endl;

	Mat struct_map(img.size(),CV_8UC1);
	Canny(img,struct_map,50,80);
//	soble_double_direction(img,struct_map);

	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		if (struct_map.ptr<unsigned char>(i)[j] < 80)
	//		{
	//			struct_map.ptr<unsigned char>(i)[j] = 0;
	//		}
	//		else
	//		{
	//			struct_map.ptr<unsigned char>(i)[j] = 255;
	//		}
	//		
	//	}
	//}

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", struct_map);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	Mat dymap = Mat::zeros(img.size(),CV_8UC1);

	for (int i = 1; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			dymap.ptr<unsigned char>(i)[j] = img.ptr<unsigned char>(i-1)[j] - img.ptr<unsigned char>(i)[j];
		}
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (dymap.ptr<unsigned char>(i)[j] <= 0)
			{
				struct_map.ptr<unsigned char>(i)[j] = 0;
			}		
		}
	}

	cvNamedWindow( "better_result", 1 );
	imshow("better_result", struct_map);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );

	//iris_center.y = iris_center.y - 4;
	cout<<"2"<<endl;
	Mat dis(img.size(),CV_32FC1);
	Mat iris(img.size(),CV_8UC1);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			dis.ptr<float>(i)[j] = sqrt(((float)j-iris_center.x)*(j-iris_center.x)+(i-iris_center.y)*(i-iris_center.y)) - iris_rds;
			iris.ptr<unsigned char>(i)[j] = (abs(dis.ptr<float>(i)[j]) < 3)?255:0;
			//iris.ptr<unsigned char>(i)[j] = (abs(dis.ptr<float>(i)[j]) < ((float)iris_rds)/20)?255:0;
		}
	}

	//pupil_center.y = pupil_center.y - 4;
	dis = Mat::zeros(img.size(),CV_32FC1);
	Mat pupil(img.size(),CV_8UC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			dis.ptr<float>(i)[j] = sqrt(((float)j-pupil_center.x)*(j-pupil_center.x)+(i-pupil_center.y)*(i-pupil_center.y)) - pupil_rds;
			pupil.ptr<unsigned char>(i)[j] = (abs(dis.ptr<float>(i)[j]) < ((float)pupil_rds)/5)?255:0;
		}
	}

	cout<<"3"<<endl;
	Mat upper_rect(img.size(),CV_8UC1);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if ((i >= iris_center.y + iris_rds*range[0]) && (i <= iris_center.y + iris_rds*range[1]) && (j >= iris_center.x - iris_rds) && (j <= iris_center.x + iris_rds) && (iris.ptr<unsigned char>(i)[j] == 0) && (pupil.ptr<unsigned char>(i)[j] == 0))
			//if ((i >= iris_center.y + iris_rds*range[0]) && (i <= iris_center.y + iris_rds*range[1]) && (j >= iris_center.x - iris_rds) && (j <= iris_center.x + iris_rds) && (pupil.ptr<unsigned char>(i)[j] == 0))
			{
				upper_rect.ptr<unsigned char>(i)[j] = 255;
			}
			else
			{
				upper_rect.ptr<unsigned char>(i)[j] = 0;
				struct_map.ptr<unsigned char>(i)[j] = 0;
			}
			if (img.ptr<unsigned char>(i)[j] >= 100)
			{
				struct_map.ptr<unsigned char>(i)[j] = 0;
			}
		}
	}


	
	cvNamedWindow( "better_result", 1 );
	imshow("better_result", struct_map);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );

	cout<<"4"<<endl;
	Vector<int> X, Y;
	int xmin = 1000;
	int ymin = 1000;
	int xmax = 0;
	int ymax = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (struct_map.ptr<unsigned char>(i)[j] != 0)
			{
				X.push_back(j);
				Y.push_back(i);
				xmin = xmin > j?j:xmin;
				xmax = xmax < j?j:xmax;
				ymin = ymin > i?i:ymin;
				ymax = ymax < i?i:ymax;
			}
		}
	}
	cout<<"5"<<endl;
	int num = Y.size();

	if (num <= iris_rds/1.5)
	{
		coffs = NULL;
	}

	int xc = (xmin + xmax)/2;
	int yc = (ymax + ymin)/2;

	int pts_x[3] = {xmin, xc, xmax};
	int pts_y[9] = {ymin, yc, yc, ymin, yc, ymin, yc, yc, ymin};

	vector<Point2f> uppers;
	vector<Point2f> middles;
	vector<Point2f> lowers;
	cout<<"6"<<endl;
	uppers.push_back(Point2f(xmin,ymin));
	uppers.push_back(Point2f(xc,yc));
	uppers.push_back(Point2f(xmax,yc));

	middles.push_back(Point2f(xmin,ymin));
	middles.push_back(Point2f(xc,yc));
	middles.push_back(Point2f(xmax,ymin));

	lowers.push_back(Point2f(xmin,yc));
	lowers.push_back(Point2f(xc,yc));
	lowers.push_back(Point2f(xmax,ymin));

	Mat coef1, coef2, coef3;

	fitPolynomial(uppers, 2, coef1);
	fitPolynomial(middles, 2, coef2);
	fitPolynomial(lowers, 2, coef3);

	//drawPolynomial(coef1,2,img);
	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	//drawPolynomial(coef2,2,img);
	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	//drawPolynomial(coef3,2,img);
	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	cout<<"7"<<endl;
	Mat D = Mat::zeros(num,3,CV_32FC1);
	Mat sum_D = Mat::zeros(1,3,CV_32FC1);

	for (int i = 0; i < num; i++)
	{
		D.ptr<float>(i)[0] = abs(Y[i] - (coef1.ptr<float>(0)[0]*X[i]*X[i] + coef1.ptr<float>(1)[0]*X[i] + coef1.ptr<float>(2)[0]));
		D.ptr<float>(i)[1] = abs(Y[i] - (coef2.ptr<float>(0)[0]*X[i]*X[i] + coef2.ptr<float>(1)[0]*X[i] + coef2.ptr<float>(2)[0]));
		D.ptr<float>(i)[2] = abs(Y[i] - (coef3.ptr<float>(0)[0]*X[i]*X[i] + coef3.ptr<float>(1)[0]*X[i] + coef3.ptr<float>(2)[0]));
		sum_D.ptr<float>(0)[0] += D.ptr<float>(i)[0];
		sum_D.ptr<float>(0)[1] += D.ptr<float>(i)[1];
		sum_D.ptr<float>(0)[2] += D.ptr<float>(i)[2];
	}

	double minsum = 0;
	Point ind(0,0);
	minMaxLoc(sum_D, &minsum,0,&ind);

	Mat D_new = Mat::zeros(num,1,CV_32FC1);

	for (int i = 0; i < num; i++)
	{
		D_new.ptr<float>(i)[0] = D.ptr<float>(i)[ind.x];
	}
	cout<<"8"<<endl;
	Mat mu_mat, sigma_mat;
	meanStdDev(D_new,mu_mat,sigma_mat);
	double mu = mu_mat.ptr<double>(0)[0];
	double sigma = sigma_mat.ptr<double>(0)[0];

	Mat score = Mat::zeros(num,1,CV_32FC1);
	vector<int> X_new, Y_new;
	for (int i = 0; i < num; i++)
	{
		score.ptr<float>(i)[0] = ((D_new.ptr<float>(i)[0] - mu)/sigma)*((D_new.ptr<float>(i)[0] - mu)/sigma);
		if (score.ptr<float>(i)[0] <= score_thresh)
		{
			X_new.push_back(X[i]);
			Y_new.push_back(Y[i]);
		}
	}
	cout<<"9"<<endl;

	vector<Point2f> final_points;

	if (X_new.size() < 3)
	{
		return -1;
	}

	cout<<"10"<<endl;

	for (int i = 0; i < X_new.size(); i++)
	{
		final_points.push_back(Point2f(X_new[i],Y_new[i]));
	}

	fitPolynomial(final_points, 2, coffs);

	//drawPolynomial(coffs,2,img);
	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	cout<<"coffs = "<<coffs.ptr<float>(0)[0]<<", "<<coffs.ptr<float>(1)[0]<<", "<<coffs.ptr<float>(2)[0]<<endl;

	if (coffs.ptr<float>(0)[0] < ((float)-1.0)/iris_rds || coffs.ptr<float>(0)[0] > 0 || X.size() < 3)
	{
		cout<<"10.1"<<endl;
		coffs = NULL;
		return -1;
	}
	cout<<"11"<<endl;

	cout<<"coffs = "<<coffs.ptr<float>(0)[0]<<", "<<coffs.ptr<float>(1)[0]<<", "<<coffs.ptr<float>(2)[0]<<endl;

	return 0;
}

void soble_double_direction(Mat img, Mat& result)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( img, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( img, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, result );
}

int fit_upper_eyelid( Mat& img, Point iris_center, int iris_rds, Point pupil_center, int pupil_rds, double* range, double offset, bool save_image, Mat& coffs)
{
	int score_thresh = 3;

	int rows = img.rows;
	int cols = img.cols;

	Mat Xmap = Mat::zeros(rows,cols,CV_8UC1);
	Mat Ymap = Mat::zeros(rows,cols,CV_8UC1);
	meshgrid(Range(1, cols),Range(1,rows),Xmap,Ymap);

	Mat struct_map(img.size(),CV_8UC1);
	//soble_double_direction(img,struct_map);
	Canny(img,struct_map,50,80);

	Mat dymap = Mat::zeros(img.size(),CV_8UC1);

	for (int i = 0; i < rows-1; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			dymap.ptr<unsigned char>(i)[j] = img.ptr<unsigned char>(i)[j] - img.ptr<unsigned char>(i+1)[j];
		}
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (dymap.ptr<unsigned char>(i)[j] <= 0)
			{
				struct_map.ptr<unsigned char>(i)[j] = 0;
			}		
		}
	}

	Mat dis(img.size(),CV_32FC1);
	//iris_center.y = iris_center.y - 4;

	//Mat dis(Xmap.size(),CV_32FC1);
	//Mat iris(img.size(),CV_8UC1);

	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		dis.ptr<float>(i)[j] = sqrt((Xmap.ptr<float>(i)[j]-iris_center.x)*(Xmap.ptr<float>(i)[j]-iris_center.x)+(Ymap.ptr<float>(i)[j]-iris_center.y)*(Ymap.ptr<float>(i)[j]-iris_center.y)) - iris_rds;
	//		iris.ptr<unsigned char>(i)[j] = (abs(dis.ptr<float>(i)[j]) < iris_rds/10)?255:0;
	//	}
	//}

	//pupil_center.y = pupil_center.y - 4;
	Mat pupil(img.size(),CV_8UC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			dis.ptr<float>(i)[j] = sqrt((j-pupil_center.x)*(j-pupil_center.x)+(i-pupil_center.y)*(i-pupil_center.y)) - pupil_rds;
			pupil.ptr<unsigned char>(i)[j] = (abs(dis.ptr<float>(i)[j]) < 4)?255:0;
		}
	}

	

	Mat upper_rect(img.size(),CV_8UC1);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if ((i >= iris_center.y - iris_rds*range[1]) && (i <= iris_center.y - iris_rds*range[0]) && (j >= iris_center.x - iris_rds) && (j <= iris_center.x + iris_rds) && (pupil.ptr<unsigned char>(i)[j] == 0))
			{
				upper_rect.ptr<unsigned char>(i)[j] = 255;
			}
			else
			{
				upper_rect.ptr<unsigned char>(i)[j] = 0;
				struct_map.ptr<unsigned char>(i)[j] = 0;
			}
		}
	}

	cvNamedWindow( "better_result", 1 );
	imshow("better_result", struct_map);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );

	cout<<"1"<<endl;

	Vector<int> X, Y;
	int xmin = 1000;
	int ymin = 1000;
	int xmax = 0;
	int ymax = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (struct_map.ptr<unsigned char>(i)[j] != 0)
			{
				X.push_back(j);
				Y.push_back(i);
				xmin = xmin > j?j:xmin;
				xmax = xmax < j?j:xmax;
				ymin = ymin > i?i:ymin;
				ymax = ymax < i?i:ymax;
			}
		}
	}

	int num = Y.size();

	cout<<"num = "<<num<<endl;

	if (num == 0)
	{
		int pts_x[3] = {iris_center.x - 2*iris_rds, iris_center.x, iris_center.x + 2*iris_rds};
		int pts_y[3] = {iris_center.y, iris_center.y - iris_rds, iris_center.y};
		vector<Point2f> fitting_points;
		for (int i = 0; i < 3; i++)
		{
			fitting_points.push_back(Point2f(pts_x[i],pts_y[i]));
		}
		fitPolynomial(fitting_points, 2, coffs);

		return -1;
	}

	int xc = (xmin+xmax)/2;
	int yc = (ymax + ymin)/2;

	int pts_x[3] = {xmin, xc, xmax};
	int pts_y[9] = {ymax, yc, yc, ymax, yc, ymax, yc, yc, ymax};

	vector<Point2f> uppers;
	vector<Point2f> middles;
	vector<Point2f> lowers;

	uppers.push_back(Point2f(xmin,ymax));
	uppers.push_back(Point2f(xc,yc));
	uppers.push_back(Point2f(xmax,yc));

	middles.push_back(Point2f(xmin,ymax));
	middles.push_back(Point2f(xc,yc));
	middles.push_back(Point2f(xmax,ymax));

	lowers.push_back(Point2f(xmin,yc));
	lowers.push_back(Point2f(xc,yc));
	lowers.push_back(Point2f(xmax,ymax));

	Mat coef1, coef2, coef3;
	cout<<"2"<<endl;
	fitPolynomial(uppers, 2, coef1);
	fitPolynomial(middles, 2, coef2);
	fitPolynomial(lowers, 2, coef3);

	cout<<"3"<<endl;
	Mat D = Mat::zeros(num,3,CV_32FC1);
	Mat sum_D = Mat::zeros(1,3,CV_32FC1);
	cout<<"coef1.ptr<unsigned char>(0)[0] = "<<coef1.ptr<float>(2)[0]<<endl;
	cout<<"coef2.ptr<unsigned char>(0)[0] = "<<coef2.ptr<float>(2)[0]<<endl;
	for (int i = 0; i < num; i++)
	{
		D.ptr<float>(i)[0] = abs(Y[i] - (coef1.ptr<float>(0)[0]*X[i]*X[i] + coef1.ptr<float>(1)[0]*X[i] + coef1.ptr<float>(2)[0]));
		D.ptr<float>(i)[1] = abs(Y[i] - (coef2.ptr<float>(0)[0]*X[i]*X[i] + coef2.ptr<float>(1)[0]*X[i] + coef2.ptr<float>(2)[0]));
		D.ptr<float>(i)[2] = abs(Y[i] - (coef3.ptr<float>(0)[0]*X[i]*X[i] + coef3.ptr<float>(1)[0]*X[i] + coef3.ptr<float>(2)[0]));
		sum_D.ptr<float>(0)[0] += D.ptr<float>(i)[0];
		sum_D.ptr<float>(0)[1] += D.ptr<float>(i)[1];
		sum_D.ptr<float>(0)[2] += D.ptr<float>(i)[2];
	}
	cout<<"4"<<endl;
	double minsum = 0;
	Point ind(0,0);
	minMaxLoc(sum_D, &minsum,0,&ind);

	Mat D_new = Mat::zeros(num,1,CV_8UC1);

	for (int i = 0; i < num; i++)
	{
		D_new.ptr<unsigned char>(i)[0] = D.ptr<unsigned char>(i)[ind.x];
	}

	Mat mu_mat, sigma_mat;
	meanStdDev(D_new,mu_mat,sigma_mat);
	double mu = mu_mat.ptr<double>(0)[0];
	double sigma = sigma_mat.ptr<double>(0)[0];

	Mat score = Mat::zeros(num,1,CV_8UC1);
	vector<int> X_new, Y_new;
	for (int i = 0; i < num; i++)
	{
		score.ptr<unsigned char>(i)[0] = ((D_new.ptr<unsigned char>(i)[0] - mu)/sigma)*((D_new.ptr<unsigned char>(i)[0] - mu)/sigma);
		if (score.ptr<unsigned char>(i)[0] <= score_thresh)
		{
			X_new.push_back(X[i]);
			Y_new.push_back(Y[i]);
		}
	}

	cout<<"X_new.size = "<<X_new.size()<<endl;

	vector<Point2f> final_points;

	for (int i = 0; i < X_new.size(); i++)
	{
		final_points.push_back(Point2f(X_new[i],Y_new[i]));
	}

	fitPolynomial(final_points, 2, coffs);

	coffs.ptr<float>(2)[0] += offset;

	cout<<"coffs.ptr<unsigned char>(0)[0] = "<<coffs.ptr<float>(2)[0]<<endl;

	//drawPolynomial(coffs,2,img);
	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	if (coffs.ptr<float>(0)[0] > 1/((float)iris_rds) || coffs.ptr<float>(0)[0] < ((float)-1.0)/20/iris_rds)
	{
		cout<<"In ex"<<endl;
		int pts_x[3] = {iris_center.x - 2*iris_rds, iris_center.x, iris_center.x + 2*iris_rds};
		int pts_y[3] = {iris_center.y, iris_center.y - iris_rds, iris_center.y};
		vector<Point2f> fitting_points;
		for (int i = 0; i < 3; i++)
		{
			fitting_points.push_back(Point2f(pts_x[i],pts_y[i]));
		}
		fitPolynomial(fitting_points, 2, coffs);

		return -1;
	}

	if (save_image)
	{
		Xmap = Mat::zeros(rows,cols,CV_8UC1);
		Ymap = Mat::zeros(rows,cols,CV_8UC1);
		meshgrid(Range(1, cols),Range(1,rows),Xmap,Ymap);

		Mat region = Mat::zeros(rows,cols,CV_8UC1);
		Mat imr = img.clone();
		Mat imb = img.clone();
		Mat imh = img.clone();

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				if (abs(coffs.ptr<unsigned char>(0)[0]*Xmap.ptr<unsigned char>(i)[j]*Xmap.ptr<unsigned char>(i)[j] + coffs.ptr<unsigned char>(1)[0]*Xmap.ptr<unsigned char>(i)[j] + coffs.ptr<unsigned char>(2)[0] - Ymap.ptr<unsigned char>(i)[j]) < 1.5)
				{
					region.ptr<unsigned char>(i)[j]  = 255;
					imr.ptr<unsigned char>(i)[j]  = 0;
					imb.ptr<unsigned char>(i)[j]  = 0;
					imh.ptr<unsigned char>(i)[j]  = 255;
				}
			}
		}

		Mat dst(img.size(),CV_8UC3);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				dst.ptr<unsigned char>(i,j)[0] = imr.ptr<unsigned char>(i)[j];
				dst.ptr<unsigned char>(i,j)[1] = imh.ptr<unsigned char>(i)[j];
				dst.ptr<unsigned char>(i,j)[2] = imb.ptr<unsigned char>(i)[j];
			}
		}
		imwrite("result_img.bmp",dst);
	}

	cout<<"Before return!"<<endl;
	cout<<"coffs = "<<coffs.ptr<float>(0)[0]<<", "<<coffs.ptr<float>(1)[0]<<", "<<coffs.ptr<float>(2)[0]<<endl;
	return 0;
}

int process_ES_region(Mat img, Mat& mask, Point center, int rds, Mat coffs, double* thresh)
{
	Mat tmp_mask = mask.clone();

	int rows = mask.rows;
	int cols = mask.cols;

	for (int i = 0; i < center.y; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			tmp_mask.ptr<unsigned char>(i)[j] = 0;
		}
	}

	Mat masked_img = img.clone();
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (tmp_mask.ptr<unsigned char>(i)[j] != 0)
			{
				masked_img.ptr<unsigned char>(i)[j] = 0;
			}
		}
	}

	int thresh_low = 0, thresh_high = 0;
	calc_threshold(masked_img, thresh[0], thresh[1], thresh_low, thresh_high);

	Mat X = Mat::zeros(img.size(),CV_8UC1);
	Mat Y = Mat::zeros(img.size(),CV_8UC1);
	meshgrid(Range(1, cols),Range(1,rows),X,Y);

	Mat PolyValue = Mat::zeros(img.size(),CV_8UC1);
	Mat ES_region = Mat::zeros(img.size(),CV_8UC1);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			PolyValue.ptr<unsigned char>(i)[j] = coffs.ptr<unsigned char>(0)[0]*X.ptr<unsigned char>(i)[j]*X.ptr<unsigned char>(i)[j] + coffs.ptr<unsigned char>(1)[0]*X.ptr<unsigned char>(i)[j] + coffs.ptr<unsigned char>(2)[0];
			if (Y.ptr<unsigned char>(i)[j] >= PolyValue.ptr<unsigned char>(i)[j] && Y.ptr<unsigned char>(i)[j] <= PolyValue.ptr<unsigned char>(i)[j] + 0.3*rds && mask.ptr<unsigned char>(i)[j] != 0)
			{
				ES_region.ptr<unsigned char>(i)[j] = 255;
			}
		}
	}

	Mat masked_img2 = img.clone();

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (ES_region.ptr<unsigned char>(i)[j] == 0)
			{
				masked_img2.ptr<unsigned char>(i)[j] = 0;
			}
		}
	}
	Mat bw_low, bw_high;
	threshold(masked_img2,bw_low,thresh_low,256,CV_THRESH_BINARY);
	threshold(masked_img2,bw_high,thresh_high,256,CV_THRESH_BINARY);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (ES_region.ptr<unsigned char>(i)[j] != 0 )
			{
				if ( bw_low.ptr<unsigned char>(i)[j] != 0 && bw_high.ptr<unsigned char>(i)[j] == 0)
				{
					masked_img2.ptr<unsigned char>(i)[j] = 255;
				}
				else
				{
					masked_img2.ptr<unsigned char>(i)[j] = 0;
				}
			}
			if(Y.ptr<unsigned char>(i)[j] < PolyValue.ptr<unsigned char>(i)[j])
			{
				masked_img2.ptr<unsigned char>(i)[j] = 0;
			}
		}
	}

	return 0;
}