
#include "iris_recognition.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <QDir>
#include "image_common.h"
#include <cstdio>

#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\photo\photo.hpp>
#include "my_iris_segmentation.h"
#include "my_image_utils.h"
#include <ctime>

//////temp
#include <io.h>
#include <iostream>
#include <stdio.h>
#include <time.h>

#include <opencv\highgui.h>
#include <opencv\cv.h>
#include <opencv\cxcore.h>

//#include "fit_curves.h"
//#include "stasm_lib.h"

//////Edit by Huan
#include <QDateTime>

#include <fstream>
#include "irisProcessUnit.h"


IrisRecognizer::IrisRecognizer()
{
}

int IrisRecognizer::initialize()
{
	min_pupil_rds_ = 10;
	max_pupil_rds_ = 60;

	template_height_ = 80;
	template_width_ = 360;

	save_image_index_ = 0;


    set_iris_matching_threshold(48);
    set_iris_quailty_threshold(128);


    return 0;
}

// int IrisRecognizer::load_all_iris_templates(QString &images_dir_path)
// {
//     QDir database_dir(images_dir_path);
//     //database_dir = database_dir.canonicalPath();
//     if (!database_dir.exists())
//     {
//         return -2;
//     }
// 
// 	foreach (auto item11, iris_gallery_)
// 	{
// 		int* buf = item11.second;
// 		if (!buf == NULL)
// 		{
// 			free(buf);
// 		}
// 	}
// 
// 	iris_gallery_.clear();
// 
// 	foreach (auto item11, mask_gallery_)
// 	{
// 		int* buf = item11.second;
// 		if (!buf == NULL)
// 		{
// 			free(buf);
// 		}
// 	}
// 
// 	mask_gallery_.clear();
// 
//     foreach(QString subject_name, database_dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
// 	{
// 
// 		QDir sub_iris_dir(database_dir.absoluteFilePath(subject_name) + "/iris/");
// 		QDir sub_mask_dir(database_dir.absoluteFilePath(subject_name) + "/mask/");
// 		/*
// 		 * Each subject
// 		 */
// 		
// 
// 		/*
// 		 * Construct inputFacesTemplate by extracting templates from images in database
// 		 * Each image of one subect
// 		 */
// //		Vector<int*> iris_template_vector;
// 		foreach(QString sample_iris_name, sub_iris_dir.entryList(QDir::Files | QDir::NoDotAndDotDot))
// 		{
// 			QString sample_iris_path = sub_iris_dir.absolutePath() + "/" + sample_iris_name;
// 
// 			string iris_fileName = sample_iris_path.toLocal8Bit();
// 
// 			int* new_iris_template = (int*) malloc(sizeof(int)*template_width_*template_height_*2);
// 
// 			iris_gallery_[subject_name][sample_iris_name] = new_iris_template;
// 		}
// 
// 		foreach(QString sample_mask_name, sub_mask_dir.entryList(QDir::Files | QDir::NoDotAndDotDot))
// 		{
// 			QString sample_mask_path = sub_iris_dir.absolutePath() + "/" + sample_mask_name;
// 
// 			string iris_fileName = sample_mask_path.toLocal8Bit();
// 
// 			int* new_mask_template = (int*) malloc(sizeof(int)*template_width_*template_height_*2);
// 
// 			mask_gallery_[subject_name][sample_mask_name] = new_mask_template;
// 		}
//     }
// 
//     return 0;
// }

int IrisRecognizer::load_all_iris_templates(QString &images_dir_path)
{	
 	save_image_index_ = 0;
    QDir database_dir(images_dir_path);

	if (!database_dir.exists())
    {
        return -2;
    }

	foreach (auto item11, iris_gallery_)
	{
		int* buf = item11.second;
		if (!buf == NULL)
		{
			free(buf);
		}
	}

	iris_gallery_.clear();

	foreach (auto item11, mask_gallery_)
	{
		int* buf = item11.second;
		if (!buf == NULL)
		{
			free(buf);
		}
	}

	mask_gallery_.clear();

    foreach(QString subject_name, database_dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
	{
		QDir sub_dir(database_dir.absoluteFilePath(subject_name));

		/*
		* Construct inputFacesTemplate by extracting templates from images in database
		* Each image of one subject
		*/
		QString file_name;
		foreach(QString sample_img_name, sub_dir.entryList(QDir::Files | QDir::NoDotAndDotDot))
		{	
			file_name = sample_img_name;
			QString sample_img_path = sub_dir.absolutePath() + "/" + sample_img_name;

			string fileName = sample_img_path.toLocal8Bit();

			Mat src = imread(fileName,1);	

			Mat src_gray;
			cvtColor(src, src_gray, CV_BGR2GRAY);

			equalizeHist(src_gray,src_gray);

			int* iris_template;
			int* mask;

			int template_width = 0, template_height = 0;

			int ret = iris_extraction(src_gray, &iris_template, &mask, &template_width, &template_height);
			if(ret != 0)
			{
				cout<<"Extract iris from image "<<fileName<<" Failed!"<<endl;
				continue;
			}

			iris_gallery_[file_name] = iris_template;
			mask_gallery_[file_name] = mask;
			cout<<"Extract iris from image "<<fileName<<" Succeed!"<<endl;
		}
	}

    return 0;
}

int IrisRecognizer::predict(Mat& left_eye, Mat& right_eye)
{
	int ret;
	
	Mat left_eye_iris_mask, right_eye_iris_mask;

	int* left_iris_template;
	int* left_mask;
	int* right_iris_template;
	int* right_mask;

	int template_width = 0, template_height = 0;

	ret = iris_extraction(left_eye, &left_iris_template, &left_mask, &template_width, &template_height);
	if(ret != 0)
	{
// 		cout<<"Left iris Location Failed!"<<endl;
		return -1;
	}
	ret = iris_extraction(right_eye, &right_iris_template, &right_mask, &template_width, &template_height);
	if(ret != 0)
	{
// 		cout<<"Right iris Location Failed!"<<endl;
		return -1;
	}

	left_iris_for_display_ = left_eye.clone();
	right_iris_for_display_ = right_eye.clone();

	best_iris_score_ = 1;
	best_iris_matching_ = "No Boday";
	
	foreach (auto subject_item, iris_gallery_)
	{

		int* iris_template_in_database = subject_item.second;
		int* mask_in_database = mask_gallery_[subject_item.first];

		double iris_match_score = gethammingdistance(left_iris_template,left_mask,iris_template_in_database,mask_in_database,1,template_width,template_height);

		if (iris_match_score > 0 && iris_match_score < best_iris_score_)
		{
			best_iris_score_ = iris_match_score;
			best_iris_matching_ = subject_item.first;
		}

		iris_match_score = gethammingdistance(right_iris_template,right_mask,iris_template_in_database,mask_in_database,1,template_width,template_height);

		if (iris_match_score > 0 && iris_match_score < best_iris_score_)
		{
			best_iris_score_ = iris_match_score;
			best_iris_matching_ = subject_item.first;
		}
	}

// 	if (best_iris_score_<0.35)
// 	{
// 		string temp_for_show = best_iris_matching_.toLocal8Bit();
// 		cout<<"In iris recognition, best_iris_score_ = "<<best_iris_score_<<endl;
// 		cout<<"In iris recognition, best_iris_matching_ = "<<temp_for_show<<endl;
// 	}	

	return 0;
}

int IrisRecognizer::set_iris_matching_threshold(int value)
{
    return 0;
}

int IrisRecognizer::set_iris_quailty_threshold(int value)
{
	return 0;
}

int IrisRecognizer::iris_mask_location(Mat& src, Mat& iris_mask)
{
	FILE *record_for_iris_recog;

	int i,j;

	cv::Point pupil_center(0,0), iris_center(0,0), upper_eyelid_center(0,0), lower_eyelid_center(0,0);
	int pupil_rds = 0, iris_rds = 0, upper_eyelid_rds = 0, lower_eyelid_rds = 0;

	int times = src.cols/200;
	if(times == 0)
	{
		times = 1;
	}

	int iMinR = min_pupil_rds_/times, iMaxR = max_pupil_rds_/times;

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

	SSREnhancement(src_small,src_small);

	GaussianBlur(src_small, src_small, Size(5,5), 0 , 0);

// 	record_for_iris_recog = fopen("record_for_iris_recog.txt","ab");
// 	fprintf(record_for_iris_recog, "Begin!\r\n");
// 	fclose(record_for_iris_recog);

	pupil_coarse_location(src_small, iMinR, iMaxR, pupil_center, pupil_rds);

// 	record_for_iris_recog = fopen("record_for_iris_recog.txt","ab");
// 	fprintf(record_for_iris_recog, "after pupil_coarse_location.\r\n");
// 	fclose(record_for_iris_recog);

	if(pupil_rds <= 0)
	{
		return -1;
	}

	pupil_accurate_location(src,times,pupil_center,pupil_rds);

	if(pupil_rds <= 0)
	{
		return -2;
	}

	iris_location(src_small, pupil_center, pupil_rds, iris_center, iris_rds);

// 	record_for_iris_recog = fopen("record_for_iris_recog.txt","ab");
// 	fprintf(record_for_iris_recog, "after iris_location.\r\n");
// 	fclose(record_for_iris_recog);

	if(iris_rds <= 0)
	{
		return -3;
	}

	Mat iris_mask = src_small.clone();

	iris_area_location(src_small,pupil_center,pupil_rds,iris_center,iris_rds,iris_mask);

	upper_eyelid_location(src_small,iris_mask,iris_center,iris_rds,upper_eyelid_center,upper_eyelid_rds);
	
// 	record_for_iris_recog = fopen("record_for_iris_recog.txt","ab");
// 	fprintf(record_for_iris_recog, "after upper_eyelid_location.\r\n");
// 	fclose(record_for_iris_recog);

	if(upper_eyelid_rds <= 0)
	{
		return -4;
	}

	lower_eyelid_location(src_small,iris_center,iris_rds,lower_eyelid_center,lower_eyelid_rds);

// 	record_for_iris_recog = fopen("record_for_iris_recog.txt","ab");
// 	fprintf(record_for_iris_recog, "after lower_eyelid_location.\r\n");
// 	fclose(record_for_iris_recog);

	if(lower_eyelid_rds <= 0)
	{
		return -5;
	}

	pupil_center.x = pupil_center.x * times;
	pupil_center.y = pupil_center.y * times;
	pupil_rds = pupil_rds * times;

	iris_center.x = iris_center.x * times;
	iris_center.y = iris_center.y * times;
	iris_rds = iris_rds * times;

	upper_eyelid_center.x = upper_eyelid_center.x*times;
	upper_eyelid_center.y = upper_eyelid_center.y*times;
	upper_eyelid_rds = upper_eyelid_rds*times;

	lower_eyelid_center.x = lower_eyelid_center.x*times;
	lower_eyelid_center.y = lower_eyelid_center.y*times;
	lower_eyelid_rds = lower_eyelid_rds*times;

	Vector<cv::Point> eyelash_points;

	eyelash_pixels_location(src,iris_center,iris_rds,pupil_center,pupil_rds,upper_eyelid_center,upper_eyelid_rds,eyelash_points);
	
	circle(src,iris_center,iris_rds,Scalar(255));
	circle(src,pupil_center,pupil_rds,Scalar(255));
	circle(src,upper_eyelid_center,upper_eyelid_rds,Scalar(255));
	circle(src,lower_eyelid_center,lower_eyelid_rds,Scalar(255));
	//Mat mask = Mat::zeros(src.rows,src.cols,src.type());
	//
	//for(i = 0; i < mask.rows; i++)
	//	for(j = 0; j < mask.cols; j++)
	//	{
	//		int dist_to_top_eyelid = (i - upper_eyelid_center.y)*(i - upper_eyelid_center.y)+(j - upper_eyelid_center.x)*(j - upper_eyelid_center.x);
	//		int dist_to_bottom_eyelid = (i - lower_eyelid_center.y)*(i - lower_eyelid_center.y)+(j - lower_eyelid_center.x)*(j - lower_eyelid_center.x);
	//		int dist_to_iris = (i - iris_center.y)*(i - iris_center.y)+(j - iris_center.x)*(j - iris_center.x);
	//		int dist_to_pupil = (i - pupil_center.y)*(i - pupil_center.y)+(j - pupil_center.x)*(j - pupil_center.x);
	//		if(dist_to_top_eyelid >= upper_eyelid_rds*upper_eyelid_rds || dist_to_bottom_eyelid >= lower_eyelid_rds*lower_eyelid_rds || dist_to_iris >= iris_rds*iris_rds || dist_to_pupil <= pupil_rds*pupil_rds)
	//		{
	//			mask.data[i*mask.cols+j] = 0;
	//		}
	//		else
	//		{
	//			mask.data[i*mask.cols+j] = 255;
	//		}
	//	}

	//for(i = 0; i < eyelash_points.size(); i++)
	//{
	//	mask.data[eyelash_points[i].y*mask.cols+eyelash_points[i].x] = 0;
	//}

	//Mat rectangle_iris;

	//iris_normalization(src,rectangle_iris,iris_mask,pupil_center,pupil_rds,iris_center,iris_rds,50,360);

	return 0;
}

int IrisRecognizer::pupil_coarse_location(Mat& src, int iMinR, int iMaxR, Point &pupil_center, int &pupil_rds)
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

int IrisRecognizer::pupil_accurate_location(Mat& src, int times, Point& pupil_center, int& pupil_rds)
{
	if (times == 0)
	{
		cout<<"In Function pupil_accurate_location, parameter times cannot be set as zero!"<<endl;
		return -1;
	}
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

	pupil_center.x = pupil_center.x - left_top.x;
	pupil_center.y = pupil_center.y - left_top.y;

	GaussianBlur(pupil_roi, pupil_roi, Size(5,5), 0 , 0);

	Mat canny_result;
	Canny(pupil_roi, canny_result, 10, 30);
	
//	int points_num = 0;
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
	
	pupil_center.x = (best_pupil_center.x + left_top.x)/times;
	pupil_center.y = (best_pupil_center.y + left_top.y)/times;
	pupil_rds = best_pupil_rds/times;

	return 0;
}

int IrisRecognizer::iris_location(Mat& src, cv::Point pupil_center, int pupil_rds, cv::Point& iris_center, int& iris_rds)
{

	if (pupil_rds <= 0)
	{
// 		cout<<"In Function iris_location, input pupil_rds cannot be as set zero!"<<endl;
		return -1;
	}
	int i,j,k;

	Mat src_canny, canny_result, tmp_result;
	GaussianBlur(src,src_canny,cv::Size(9,9),2,2);
	Canny(src_canny,canny_result,5,30);

// 	cout<<"After Canny"<<endl;

//	int points_num = 0;
	Vector<cv::Point> canny_points;
	Vector<int> canny_R;
	int coarse_R[4000];
	memset(coarse_R,0,4000*sizeof(int));
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

// 	cout<<"canny_points.size = "<<canny_points.size()<<endl;
	if (canny_points.size()< 10)
	{
// 		cout<<"Image quality is poor. Iris location Failed!"<<endl;
		return -2;
	}

// 	cout<<"After Grandence calculate"<<endl;
	
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

	if (iris_rds <= 5)
	{
// 		cout<<"Iris location failed!"<<endl;
		return -4;
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

// 	cout<<"After Difference"<<endl;

	return 0;
}

//int IrisRecognizer::upper_eyelid_location(Mat src, Point iris_center, int iris_rds, Point &upper_eyelid_center, int &upper_eyelid_rds)
//{
////	long lRet;
//	int i,j;
////	char szErr[512];
////	int width = src.cols, height = src.rows;
//	Point left_top, right_bottom;
//	left_top.x = iris_center.x - 2*iris_rds;
//	left_top.y = iris_center.y - 1.2*iris_rds;
//	right_bottom.x = iris_center.x + 2*iris_rds;
//	right_bottom.y = iris_center.y;
//
//	if(left_top.x < 5)
//	{
//		left_top.x = 5;
//	}
//	if(left_top.y < 5)
//	{
//		left_top.y = 5;
//	}
//	if(right_bottom.x > src.cols - 5)
//	{
//		right_bottom.x = src.cols - 5;
//	}
//
//	Rect upper_eyelid_rect(left_top,right_bottom);
//
//	Mat tmp_eyelid_img = src(upper_eyelid_rect);
//	Mat small_eyelid_img = tmp_eyelid_img.clone();
//	int iSmallWidth = small_eyelid_img.cols, iSmallHeight = small_eyelid_img.rows;
//
//	GaussianBlur(small_eyelid_img, small_eyelid_img, Size(9,9), 0 , 0);
//	GaussianBlur(small_eyelid_img, small_eyelid_img, Size(9,9), 0 , 0);
//
//
//	double *pGradx=new double[iSmallWidth*iSmallHeight]; 
//	double *pGrady=new double[iSmallWidth*iSmallHeight]; 
//	double *pGradxy=new double[iSmallWidth*iSmallHeight];
//	memset(pGradx,0,iSmallWidth*iSmallHeight*sizeof(double));
//	memset(pGrady,0,iSmallWidth*iSmallHeight*sizeof(double));
//	memset(pGradxy,0,iSmallWidth*iSmallHeight*sizeof(double));
//
//	int iPupilCenterX = iSmallWidth/2;
//	int iPupilCenterY = iSmallHeight;
//	int iPupilRadius = iris_rds;
//
//	int iYStar=4;
//	int iXStar=4;
//	int iYEnd=iSmallHeight-4;
//	int iXEnd=iSmallWidth-4;
//
//	int iTemp1,iTemp2;
//	for(i=iYStar;i<iYEnd;i++)
//	{
//		iTemp1=i*iSmallWidth;
//		for(j=iXStar;j<iXEnd;j++)
//		{
//			iTemp2=iTemp1+j;
//
//			pGradx[iTemp2] = (small_eyelid_img.data[iTemp2-iSmallWidth*2-2]+small_eyelid_img.data[iTemp2-iSmallWidth*2-1]+small_eyelid_img.data[iTemp2+iSmallWidth*2-2]+small_eyelid_img.data[iTemp2+iSmallWidth*2-1]) 
//				+ 2*(small_eyelid_img.data[iTemp2-iSmallWidth-2]+small_eyelid_img.data[iTemp2-iSmallWidth-1]+small_eyelid_img.data[iTemp2+iSmallWidth-2]+small_eyelid_img.data[iTemp2+iSmallWidth-1]) 
//				+ 3*(small_eyelid_img.data[iTemp2-2]+small_eyelid_img.data[iTemp2-1])
//				- (small_eyelid_img.data[iTemp2-iSmallWidth*2+2]+small_eyelid_img.data[iTemp2-iSmallWidth*2+1]+small_eyelid_img.data[iTemp2+iSmallWidth*2+2]+small_eyelid_img.data[iTemp2+iSmallWidth*2+1]) 
//				- 2*(small_eyelid_img.data[iTemp2-iSmallWidth+2]+small_eyelid_img.data[iTemp2-iSmallWidth+1]+small_eyelid_img.data[iTemp2+iSmallWidth+2]+small_eyelid_img.data[iTemp2+iSmallWidth+1]) 
//				- 3*(small_eyelid_img.data[iTemp2+2]+small_eyelid_img.data[iTemp2+1]);
//
//			pGrady[iTemp2] = (small_eyelid_img.data[iTemp2-2*iSmallWidth-2]+small_eyelid_img.data[iTemp2-iSmallWidth-2]+small_eyelid_img.data[iTemp2-2*iSmallWidth+2]+small_eyelid_img.data[iTemp2-iSmallWidth+2])
//				+ 2*(small_eyelid_img.data[iTemp2-2*iSmallWidth-1]+small_eyelid_img.data[iTemp2-iSmallWidth-1]+small_eyelid_img.data[iTemp2-2*iSmallWidth+1]+small_eyelid_img.data[iTemp2-iSmallWidth+1])
//				+ 3*(small_eyelid_img.data[iTemp2-2*iSmallWidth]+small_eyelid_img.data[iTemp2-iSmallWidth])
//				- (small_eyelid_img.data[iTemp2+2*iSmallWidth-2]+small_eyelid_img.data[iTemp2+iSmallWidth-2]+small_eyelid_img.data[iTemp2+2*iSmallWidth+2]+small_eyelid_img.data[iTemp2+iSmallWidth+2])
//				- 2*(small_eyelid_img.data[iTemp2+2*iSmallWidth-1]+small_eyelid_img.data[iTemp2+iSmallWidth-1]+small_eyelid_img.data[iTemp2+2*iSmallWidth+1]+small_eyelid_img.data[iTemp2+iSmallWidth+1])
//				- 3*(small_eyelid_img.data[iTemp2+2*iSmallWidth]+small_eyelid_img.data[iTemp2+iSmallWidth]);
//			
//
//			pGradxy[iTemp2] = sqrt(pGradx[iTemp2]*pGradx[iTemp2] + pGrady[iTemp2]*pGrady[iTemp2]);
//			//
//			if ((i<iPupilCenterY-9 && pGradxy[iTemp2] <200) || (i>=iPupilCenterY-9 && pGradxy[iTemp2] <400) || fabs(pGradx[iTemp2]) >pGradxy[iTemp2]/2)
//			{
//				pGradxy[iTemp2] = 0;
//			}
//			else if ((sqrt((double)((i-iPupilCenterY)*(i-iPupilCenterY)+(j-iPupilCenterX)*(j-iPupilCenterX)))-(double)iPupilRadius<=8 && sqrt((double)((i-iPupilCenterY)*(i-iPupilCenterY)+(j-iPupilCenterX)*(j-iPupilCenterX)))-(double)iPupilRadius>-8) && (double)(abs(j-iPupilCenterX))<((double)(iPupilCenterY-i))*0.8)
//			{
//				pGradxy[iTemp2] =0;
//			}
//		}
//	}
//	
//	Mat grad_image(iSmallHeight,iSmallWidth,CV_8UC1);
//	Mat grad_value_img(iSmallHeight,iSmallWidth,CV_8UC1);
//	double dGMax=-100000,dGMin=100000;
//	for(i=0;i<iSmallWidth*iSmallHeight;i++)
//	{
//		if(pGradxy[i]>dGMax)
//			dGMax=pGradxy[i];
//		if(pGradxy[i]<dGMin)
//			dGMin=pGradxy[i];
//	}
//
//	for(i = 0; i < iSmallHeight; i++)
//		for(j = 0; j < iSmallWidth; j++)
//		{
//			if(cvRound((pGradxy[i*iSmallWidth+j]-dGMin)*255/(dGMax-dGMin)) != 0)
//				grad_image.data[i*iSmallWidth+j] = 255;
//			else
//				grad_image.data[i*iSmallWidth+j] = 0;
//			grad_value_img.data[i*iSmallWidth+j] = (unsigned char)(cvRound((pGradxy[i*iSmallWidth+j]-dGMin)*255/(dGMax-dGMin)));
//		}
//
//	vector<vector<Point> > contours;
//	
//	 findContours(grad_image, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
//
//	 vector<Point> contmax;
//	 double maxArea = 0;
//	 int ids = 0, maxIds = 0;
//
//	 for (auto contour : contours)
//	 {
//		 double tempArea = fabs(contourArea(contour));
//		 if(tempArea > maxArea)
//		 {
//			 maxArea = tempArea;
//			 contmax = contour;
//			 maxIds = ids;
//		 }
//		 ids++;
//	 }
//
//	 
//	 int iMeanY = 0, iSumY = 0;
//	 int iNum = 0;
//
//	 for (auto points : contours[maxIds])
//	 {
//		 iSumY += points.y;
//		 iNum++;
//	 }
//	 iMeanY = iSumY/iNum;
//
//	 Vector<Point> eyelid_points;
//	 int index_num = 0;
//	 Mat eyelid_edge_img(grad_image.rows, grad_image.cols,CV_8UC1);
//
//	 for(i = 0; i < eyelid_edge_img.rows; i++)
//		 for(j = 0; j < eyelid_edge_img.cols; j++)
//		 {
//			 eyelid_edge_img.data[i*eyelid_edge_img.cols+j] = 0;
//		 }
//
//	 for (i = 0; i < contours[maxIds].size(); i++)
//	 {
//		 if(contours[maxIds][i].y < iMeanY)
//		 {
//			 eyelid_points.push_back(contours[maxIds][i]);
//			 index_num++;
//		 }
//	 }
//
//	 Point upperEyelid_center,best_upper_eyelid_center;
//	 int upperEyelid_rds,best_upper_eyelid_rds, num_of_points,sum_of_points = 0;
//	 for(j = 0; j < 10; j++)
//	 {
//		num_of_points = 0;
//		upperEyelid_rds = 0;
//		upperEyelid_center.y = 0;
//		while(upperEyelid_rds < 3*iris_rds || upperEyelid_center.y < iris_center.y)
//		{
//			std::vector<cv::Point> rand_points;
//			for(i = 0; i < 5; i++)
//			{
//				int index = rand()%eyelid_points.size();
//				rand_points.push_back(eyelid_points[index]);
//			}
//
//			LeastSquares(rand_points,upperEyelid_center,upperEyelid_rds);	
//		}
//
//		for(i = 0; i < eyelid_points.size(); i++)
//		{
//			int tmp_dist = (eyelid_points[i].x - upperEyelid_center.x)*(eyelid_points[i].x - upperEyelid_center.x)+(eyelid_points[i].y - upperEyelid_center.y)*(eyelid_points[i].y - upperEyelid_center.y) - upperEyelid_rds*upperEyelid_rds;
//			if(tmp_dist < 5)
//			{
//				num_of_points++;
//			}
//		}
//		if(sum_of_points < num_of_points)
//		{
//			sum_of_points = num_of_points;
//			best_upper_eyelid_center.x = upperEyelid_center.x;
//			best_upper_eyelid_center.y = upperEyelid_center.y;
//			best_upper_eyelid_rds = upperEyelid_rds;
//		}
//	 }
//
//	 circle(eyelid_edge_img, best_upper_eyelid_center, best_upper_eyelid_rds, Scalar(255));
//
//	 Point* full_lid_points = new Point[eyelid_edge_img.cols-40];
//	 memset(full_lid_points,0,(eyelid_edge_img.cols-40)*sizeof(Point));
//	 int full_index=0;
//	 for(i = 20; i< eyelid_edge_img.cols-20; i++)
//		 for(j = 20; j < eyelid_edge_img.rows-20; j++)
//		 {
//			 if(eyelid_edge_img.data[j*eyelid_edge_img.cols+i] != 0)
//			 {
//				 full_lid_points[full_index].x = i;
//				 full_lid_points[full_index++].y = j;
//				 break;
//			 }
//		 }
// 
//	 int grad_max = -1000, sum_OK = -1000;
//	 int max_move = 0;
//	 for(i=0; i < 20; i++)
//	 {
//		 int grad_sum = 0, OK_num = 0;
//		 for(j = 0; j < full_index; j++)
//		 {
//			 if(grad_value_img.data[(full_lid_points[j].y+i)*grad_value_img.cols+full_lid_points[j].x] > 100)
//			 {
//				 grad_sum += grad_value_img.data[(full_lid_points[j].y+i)*grad_value_img.cols+full_lid_points[j].x];
//				 OK_num++;
//			 }
//		 }
//		 sum_OK = grad_sum*OK_num;
//		 if(grad_max < sum_OK)
//		 {
//			 grad_max = sum_OK;
//			 max_move = i;
//		 }
//		 i++;
//	 }
//
//	 int m_fCenterX = best_upper_eyelid_center.x;
//	 int m_fCenterY = best_upper_eyelid_center.y + max_move;
//	 int m_fRadius = best_upper_eyelid_rds;
//
//	 upper_eyelid_center.x = m_fCenterX + left_top.x;
//	 upper_eyelid_center.y = m_fCenterY + left_top.y+5;
//	 upper_eyelid_rds = m_fRadius;
//
//	delete pGradx;
//	delete pGrady;
//	delete pGradxy;
//	return 0;
//}

int IrisRecognizer::upper_eyelid_location(Mat& src, Mat& iris_mask, Point iris_center, int iris_rds, Point &top_eyelid_center, int &top_eyelid_rds)
{
	int i,j,k,kk,m,n;
	Mat top_eyelid_img,top_eyelid_mask;

	Point left_top_point, right_bottom_point;

	left_top_point.x = iris_center.x - 2*iris_rds;
	left_top_point.y = iris_center.y - 1.2*iris_rds;
	right_bottom_point.x = iris_center.x + 2*iris_rds;
	right_bottom_point.y = iris_center.y;

	if(left_top_point.x < 5)
	{
		left_top_point.x = 5;
	}
	if(left_top_point.y < 5)
	{
		left_top_point.y = 5;
	}
	if(right_bottom_point.x > src.cols - 5)
	{
		right_bottom_point.x = src.cols - 5;
	}
	if(right_bottom_point.y > src.rows - 5)
	{
		right_bottom_point.y = src.rows - 5;
	}

	Rect top_eyelid_rect(left_top_point,right_bottom_point);

	Mat tmp_eyelid_img = src(top_eyelid_rect);
	top_eyelid_img = tmp_eyelid_img.clone();

	tmp_eyelid_img = iris_mask(top_eyelid_rect);
	top_eyelid_mask = tmp_eyelid_img.clone();

	GaussianBlur(top_eyelid_img,top_eyelid_img,Size(5,5),0,0);
	GaussianBlur(top_eyelid_img,top_eyelid_img,Size(5,5),0,0);

	Mat top_eyeled_canny_result;
	Canny(top_eyelid_img, top_eyeled_canny_result,5,40);

	std::vector<cv::Point> canny_points;

	for(i = 1; i < top_eyeled_canny_result.rows-1; i++)
		for(j = 1; j < top_eyeled_canny_result.cols-1; j++)
		{
			if(top_eyeled_canny_result.data[i*top_eyeled_canny_result.cols + j] != 0)
			{
				int y_value = ((top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j]*2+top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j+1]+top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j-1])-(top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j]*2+top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j+1]+top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j-1]))/4;
				int x_value = ((top_eyelid_img.data[(i)*top_eyeled_canny_result.cols+j]*2+top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j+1]+top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j+1])-(top_eyelid_img.data[(i)*top_eyeled_canny_result.cols+j-1]*2+top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j-1]+top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j-1]))/4;
				if(x_value == 0)
				{
					top_eyeled_canny_result.data[i*top_eyeled_canny_result.cols+j] = 0;
				}
				else if(abs(y_value/x_value) < 3.732)
				{
					top_eyeled_canny_result.data[i*top_eyeled_canny_result.cols+j] = 0;
				}
				else
				{
					top_eyeled_canny_result.data[i*top_eyeled_canny_result.cols+j] = 255;
				} 
			}
		}

	for(i = 1; i < top_eyeled_canny_result.cols-1; i++)
		for(j = 1; j < top_eyeled_canny_result.rows-1; j++)
		{
			if(top_eyeled_canny_result.data[j*top_eyeled_canny_result.cols+i] == 255 && top_eyelid_mask.data[j*top_eyeled_canny_result.cols+i] == 255)
			{
				int tmp_value = top_eyeled_canny_result.data[(j+1)*top_eyeled_canny_result.cols+(i+1)]/255+top_eyeled_canny_result.data[(j+1)*top_eyeled_canny_result.cols+(i)]/255+top_eyeled_canny_result.data[(j+1)*top_eyeled_canny_result.cols+(i-1)]/255+top_eyeled_canny_result.data[(j)*top_eyeled_canny_result.cols+(i+1)]/255+top_eyeled_canny_result.data[(j)*top_eyeled_canny_result.cols+(i-1)]/255+top_eyeled_canny_result.data[(j-1)*top_eyeled_canny_result.cols+(i+1)]/255+top_eyeled_canny_result.data[(j-1)*top_eyeled_canny_result.cols+(i-1)]/255+top_eyeled_canny_result.data[(j-1)*top_eyeled_canny_result.cols+(i)]/255;
				if(tmp_value >= 2)
				{
					canny_points.push_back(cv::Point((i+left_top_point.x),(j+left_top_point.y)));
					break;
				}
			}
		}
	if(canny_points.size() <= 5)
	{
		return -1;
	}
	
	int sum_of_points = 0, num_of_points = 0, best_top_eyelid_rds = 0;
	cv::Point best_top_eyelid_center(0,0);

	for(j = 0; j < 10; j++)
	{
		num_of_points = 0;
		top_eyelid_rds = 0;
		top_eyelid_center.y = 0;
	
		int iteration = 0;
		while(top_eyelid_rds < 3*iris_rds || top_eyelid_center.y < iris_center.y)
		{
			std::vector<cv::Point> rand_points;
			
			for(i = 0; i < 5; i++)
			{
				int index = rand()%canny_points.size();
				rand_points.push_back(canny_points[index]);
			}

			LeastSquares(rand_points,top_eyelid_center,top_eyelid_rds);	
			iteration++;
			if(iteration > 50)
			{
				break;
			}
		}
		
		for(i = 0; i < canny_points.size(); i++)
		{
			int tmp_dist = (canny_points[i].x - top_eyelid_center.x)*(canny_points[i].x - top_eyelid_center.x)+(canny_points[i].y - top_eyelid_center.y)*(canny_points[i].y - top_eyelid_center.y) - top_eyelid_rds*top_eyelid_rds;
			if(tmp_dist < 5)
			{
				num_of_points++;
			}
		}
		
		if(sum_of_points < num_of_points)
		{
			sum_of_points = num_of_points;
			best_top_eyelid_center.x = top_eyelid_center.x;
			best_top_eyelid_center.y = top_eyelid_center.y;
			best_top_eyelid_rds = top_eyelid_rds;
		}
	}

	top_eyelid_center.x = best_top_eyelid_center.x;
	top_eyelid_center.y = best_top_eyelid_center.y;
	top_eyelid_rds = best_top_eyelid_rds;

	return 0;
}

int IrisRecognizer::upper_eyelid_location_new(Mat& src, Point pupil_center, int pupil_rds, Point iris_center, int iris_rds, Point &top_eyelid_center, int &top_eyelid_rds)
{
	int i,j,k,kk,m,n;
	Mat top_eyelid_img,top_eyelid_mask,iris_mask = src.clone();

	iris_area_location(src,pupil_center,pupil_rds,iris_center,iris_rds,iris_mask);

	Point left_top_point, right_bottom_point;

	left_top_point.x = iris_center.x - 2*iris_rds;
	left_top_point.y = iris_center.y - 1.2*iris_rds;
	right_bottom_point.x = iris_center.x + 2*iris_rds;
	right_bottom_point.y = iris_center.y;

	if(left_top_point.x < 5)
	{
		left_top_point.x = 5;
	}
	if(left_top_point.y < 5)
	{
		left_top_point.y = 5;
	}
	if(right_bottom_point.x > src.cols - 5)
	{
		right_bottom_point.x = src.cols - 5;
	}
	if(right_bottom_point.y > src.rows - 5)
	{
		right_bottom_point.y = src.rows - 5;
	}

	Rect top_eyelid_rect(left_top_point,right_bottom_point);

	Mat tmp_eyelid_img = src(top_eyelid_rect);
	top_eyelid_img = tmp_eyelid_img.clone();

	tmp_eyelid_img = iris_mask(top_eyelid_rect);
	top_eyelid_mask = tmp_eyelid_img.clone();

	GaussianBlur(top_eyelid_img,top_eyelid_img,Size(5,5),0,0);
	GaussianBlur(top_eyelid_img,top_eyelid_img,Size(5,5),0,0);

	Mat top_eyeled_canny_result;
	Canny(top_eyelid_img, top_eyeled_canny_result,5,40);

	std::vector<cv::Point> canny_points;

	for(i = 1; i < top_eyeled_canny_result.rows-1; i++)
		for(j = 1; j < top_eyeled_canny_result.cols-1; j++)
		{
			if(top_eyeled_canny_result.data[i*top_eyeled_canny_result.cols + j] != 0)
			{
				int y_value = ((top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j]*2+top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j+1]+top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j-1])-(top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j]*2+top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j+1]+top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j-1]))/4;
				int x_value = ((top_eyelid_img.data[(i)*top_eyeled_canny_result.cols+j]*2+top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j+1]+top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j+1])-(top_eyelid_img.data[(i)*top_eyeled_canny_result.cols+j-1]*2+top_eyelid_img.data[(i+1)*top_eyeled_canny_result.cols+j-1]+top_eyelid_img.data[(i-1)*top_eyeled_canny_result.cols+j-1]))/4;
				if(x_value == 0)
				{
					top_eyeled_canny_result.data[i*top_eyeled_canny_result.cols+j] = 0;
				}
				else if(abs(y_value/x_value) < 3.732)
				{
					top_eyeled_canny_result.data[i*top_eyeled_canny_result.cols+j] = 0;
				}
				else
				{
					top_eyeled_canny_result.data[i*top_eyeled_canny_result.cols+j] = 255;
				} 
			}
		}

	for(i = 1; i < top_eyeled_canny_result.cols-1; i++)
		for(j = 1; j < top_eyeled_canny_result.rows-1; j++)
		{
			if(top_eyeled_canny_result.data[j*top_eyeled_canny_result.cols+i] == 255 && top_eyelid_mask.data[j*top_eyeled_canny_result.cols+i] == 255)
			{
				int tmp_value = top_eyeled_canny_result.data[(j+1)*top_eyeled_canny_result.cols+(i+1)]/255+top_eyeled_canny_result.data[(j+1)*top_eyeled_canny_result.cols+(i)]/255+top_eyeled_canny_result.data[(j+1)*top_eyeled_canny_result.cols+(i-1)]/255+top_eyeled_canny_result.data[(j)*top_eyeled_canny_result.cols+(i+1)]/255+top_eyeled_canny_result.data[(j)*top_eyeled_canny_result.cols+(i-1)]/255+top_eyeled_canny_result.data[(j-1)*top_eyeled_canny_result.cols+(i+1)]/255+top_eyeled_canny_result.data[(j-1)*top_eyeled_canny_result.cols+(i-1)]/255+top_eyeled_canny_result.data[(j-1)*top_eyeled_canny_result.cols+(i)]/255;
				if(tmp_value >= 2)
				{
					canny_points.push_back(cv::Point((i+left_top_point.x),(j+left_top_point.y)));
					break;
				}
			}
		}

	if(canny_points.size() == 0)
	{
		return -9;
	}

	int sum_of_points = 0, num_of_points = 0, best_top_eyelid_rds = 0;
	cv::Point best_top_eyelid_center(0,0);

	for(j = 0; j < 10; j++)
	{
		num_of_points = 0;
		top_eyelid_rds = 0;
		top_eyelid_center.y = 0;
		while(top_eyelid_rds < 3*iris_rds || top_eyelid_center.y < iris_center.y)
		{
			std::vector<cv::Point> rand_points;
			for(i = 0; i < 5; i++)
			{
				int index = rand()%canny_points.size();
				rand_points.push_back(canny_points[index]);
			}

			LeastSquares(rand_points,top_eyelid_center,top_eyelid_rds);	
		}

		for(i = 0; i < canny_points.size(); i++)
		{
			int tmp_dist = (canny_points[i].x - top_eyelid_center.x)*(canny_points[i].x - top_eyelid_center.x)+(canny_points[i].y - top_eyelid_center.y)*(canny_points[i].y - top_eyelid_center.y) - top_eyelid_rds*top_eyelid_rds;
			if(tmp_dist < 5)
			{
				num_of_points++;
			}
		}
		if(sum_of_points < num_of_points)
		{
			sum_of_points = num_of_points;
			best_top_eyelid_center.x = top_eyelid_center.x;
			best_top_eyelid_center.y = top_eyelid_center.y;
			best_top_eyelid_rds = top_eyelid_rds;
		}
	}

	top_eyelid_center.x = best_top_eyelid_center.x;
	top_eyelid_center.y = best_top_eyelid_center.y;
	top_eyelid_rds = best_top_eyelid_rds;

	return 0;
}

int IrisRecognizer::lower_eyelid_location(Mat& src, Point iris_center, int iris_rds, Point& lower_eyelid_center, int& lower_eyelid_rds)
{
 	FILE *recode_for_lower_eyelid_location;
// 	recode_for_lower_eyelid_location = fopen("recode_for_lower_eyelid_location.txt","ab");
// 	fprintf(recode_for_lower_eyelid_location, "In recode_for_lower_eyelid_location.\r\n");
// 	fclose(recode_for_lower_eyelid_location);

	int i,j;
	Mat lower_eyelid_img;

	Point left_top_point, right_bottom_point;

	left_top_point.x = iris_center.x - 2*iris_rds;
	left_top_point.y = iris_center.y + 0.5*iris_rds;
	right_bottom_point.x = iris_center.x + 2*iris_rds;
	right_bottom_point.y = iris_center.y + 2*iris_rds;

	if(left_top_point.x < 5)
	{
		left_top_point.x = 5;
	}
	if(left_top_point.y < 5)
	{
		left_top_point.y = 5;
	}
	if(right_bottom_point.x > src.cols - 5)
	{
		right_bottom_point.x = src.cols - 5;
	}
	if(right_bottom_point.y > src.rows - 5)
	{
		right_bottom_point.y = src.rows - 5;
	}

	Rect lower_eyelid_rect(left_top_point,right_bottom_point);

	Mat tmp_eyelid_img = src(lower_eyelid_rect);
	lower_eyelid_img = tmp_eyelid_img.clone();

// 	recode_for_lower_eyelid_location = fopen("recode_for_lower_eyelid_location.txt","ab");
// 	fprintf(recode_for_lower_eyelid_location, "After image crop.\r\n");
// 	fclose(recode_for_lower_eyelid_location);

	GaussianBlur(lower_eyelid_img,lower_eyelid_img,Size(5,5),0,0);
	GaussianBlur(lower_eyelid_img,lower_eyelid_img,Size(5,5),0,0);

	Mat lower_eyeled_canny_result;
	Canny(lower_eyelid_img, lower_eyeled_canny_result,5,40);

// 	recode_for_lower_eyelid_location = fopen("recode_for_lower_eyelid_location.txt","ab");
// 	fprintf(recode_for_lower_eyelid_location, "After Canny.\r\n");
// 	fclose(recode_for_lower_eyelid_location);

	std::vector<cv::Point> canny_points;

	for(i = 1; i < lower_eyeled_canny_result.rows-1; i++)
		for(j = 1; j < lower_eyeled_canny_result.cols-1; j++)
		{
			if(lower_eyeled_canny_result.data[i*lower_eyeled_canny_result.cols + j] != 0)
			{
				int y_value = ((lower_eyelid_img.data[(i+1)*lower_eyeled_canny_result.cols+j]*2+lower_eyelid_img.data[(i+1)*lower_eyeled_canny_result.cols+j+1]+lower_eyelid_img.data[(i+1)*lower_eyeled_canny_result.cols+j-1])-(lower_eyelid_img.data[(i-1)*lower_eyeled_canny_result.cols+j]*2+lower_eyelid_img.data[(i-1)*lower_eyeled_canny_result.cols+j+1]+lower_eyelid_img.data[(i-1)*lower_eyeled_canny_result.cols+j-1]))/4;
				int x_value = ((lower_eyelid_img.data[(i)*lower_eyeled_canny_result.cols+j]*2+lower_eyelid_img.data[(i-1)*lower_eyeled_canny_result.cols+j+1]+lower_eyelid_img.data[(i+1)*lower_eyeled_canny_result.cols+j+1])-(lower_eyelid_img.data[(i)*lower_eyeled_canny_result.cols+j-1]*2+lower_eyelid_img.data[(i+1)*lower_eyeled_canny_result.cols+j-1]+lower_eyelid_img.data[(i-1)*lower_eyeled_canny_result.cols+j-1]))/4;
				if(x_value == 0)
				{
					lower_eyeled_canny_result.data[i*lower_eyeled_canny_result.cols+j] = 0;
				}
				else if(abs(y_value/x_value) < 3.732)
				{
					lower_eyeled_canny_result.data[i*lower_eyeled_canny_result.cols+j] = 0;
				}
				else
				{
					lower_eyeled_canny_result.data[i*lower_eyeled_canny_result.cols+j] = 255;
				} 
			}
		}

	for(i = 1; i < lower_eyeled_canny_result.cols-1; i++)
		for(j = 1; j < lower_eyeled_canny_result.rows-1; j++)
		{
			if(lower_eyeled_canny_result.data[j*lower_eyeled_canny_result.cols+i] == 255)
			{
				int tmp_value = lower_eyeled_canny_result.data[(j+1)*lower_eyeled_canny_result.cols+(i+1)]/255+lower_eyeled_canny_result.data[(j+1)*lower_eyeled_canny_result.cols+(i)]/255+lower_eyeled_canny_result.data[(j+1)*lower_eyeled_canny_result.cols+(i-1)]/255+lower_eyeled_canny_result.data[(j)*lower_eyeled_canny_result.cols+(i+1)]/255+lower_eyeled_canny_result.data[(j)*lower_eyeled_canny_result.cols+(i-1)]/255+lower_eyeled_canny_result.data[(j-1)*lower_eyeled_canny_result.cols+(i+1)]/255+lower_eyeled_canny_result.data[(j-1)*lower_eyeled_canny_result.cols+(i-1)]/255+lower_eyeled_canny_result.data[(j-1)*lower_eyeled_canny_result.cols+(i)]/255;
				if(tmp_value != 0)
				{
					canny_points.push_back(cv::Point((i+left_top_point.x),(j+left_top_point.y)));
					break;
				}
			}
		}

	if(canny_points.size() == 0)
	{
		return -1;
	}

// 	recode_for_lower_eyelid_location = fopen("recode_for_lower_eyelid_location.txt","ab");
// 	fprintf(recode_for_lower_eyelid_location, "After modify Canny.\r\n");
// 	fclose(recode_for_lower_eyelid_location);

	int sum_of_points = 0, num_of_points = 0, best_top_eyelid_rds = 0;
	cv::Point best_top_eyelid_center(0,0);

	for(j = 0; j < 10; j++)
	{
		num_of_points = 0;
		lower_eyelid_rds = 0;
		lower_eyelid_center.y = 0;
		while(lower_eyelid_rds < 3*iris_rds || lower_eyelid_center.y > iris_center.y)
		{
			std::vector<cv::Point> rand_points;
			for(i = 0; i < 5; i++)
			{
				int index = rand()%canny_points.size();
				rand_points.push_back(canny_points[index]);
			}

			LeastSquares(rand_points,lower_eyelid_center,lower_eyelid_rds);	
		}

		for(i = 0; i < canny_points.size(); i++)
		{
			int tmp_dist = (canny_points[i].x - lower_eyelid_center.x)*(canny_points[i].x - lower_eyelid_center.x)+(canny_points[i].y - lower_eyelid_center.y)*(canny_points[i].y - lower_eyelid_center.y) - lower_eyelid_rds*lower_eyelid_rds;
			if(tmp_dist < 5)
			{
				num_of_points++;
			}
		}
		if(sum_of_points < num_of_points)
		{
			sum_of_points = num_of_points;
			best_top_eyelid_center.x = lower_eyelid_center.x;
			best_top_eyelid_center.y = lower_eyelid_center.y;
			best_top_eyelid_rds = lower_eyelid_rds;
		}
	}

// 	recode_for_lower_eyelid_location = fopen("recode_for_lower_eyelid_location.txt","ab");
// 	fprintf(recode_for_lower_eyelid_location, "After LeastSquares.\r\n");
// 	fclose(recode_for_lower_eyelid_location);

	return 0;
}

int IrisRecognizer::eyelash_pixels_location(Mat& src, Point iris_center, int iris_rds, Point pupil_center, int pupil_rds, Point upper_eyelid_center, int upper_eyelid_rds, Vector<cv::Point>& eyelash_points)
{
	int i, j;
	Vector<Point> ES_points, IR_points;

	for(i = upper_eyelid_center.y - upper_eyelid_rds; i < iris_center.y+0.3*iris_rds; i++)
		for(j = iris_center.x - iris_rds; j < iris_center.x + iris_rds; j++)
		{
			double dist_to_eyelid = sqrt(((double)i-upper_eyelid_center.y)*(i-upper_eyelid_center.y)+(j-upper_eyelid_center.x)*(j-upper_eyelid_center.x));
			double dist_to_iris = sqrt(((double)i-iris_center.y)*(i-iris_center.y)+(j-iris_center.x)*(j-iris_center.x));
			double dist_to_pupil = sqrt(((double)i-pupil_center.y)*(i-pupil_center.y)+(j-pupil_center.x)*(j-pupil_center.x));
			if(dist_to_eyelid < upper_eyelid_rds && dist_to_iris < iris_rds && dist_to_pupil > pupil_rds)
			{
				ES_points.push_back(cv::Point(j,i));
			}
		}

	for(i = iris_center.y; i < iris_center.y + 0.5*iris_rds + 0.5*pupil_rds; i++)
		for(j = iris_center.x - 0.5*iris_rds; j < iris_center.x + 0.5*iris_rds; j++)
		{
			double dist_to_iris = sqrt(((double)i-iris_center.y)*(i-iris_center.y)+(j-iris_center.x)*(j-iris_center.x));
			double dist_to_pupil = sqrt(((double)i-pupil_center.y)*(i-pupil_center.y)+(j-pupil_center.x)*(j-pupil_center.x));
			if(dist_to_iris < 0.5*iris_rds + 0.5*pupil_rds && dist_to_pupil > pupil_rds)
			{
				IR_points.push_back(cv::Point(j,i));
			}
		}

	if (IR_points.size() == 0 || ES_points.size() == 0)
	{
		return -1;
	}

	float sum=0,s=0,mean,stand;
	for(i = 0; i < IR_points.size(); i++)
	{
		sum += src.data[IR_points[i].y*src.cols+IR_points[i].x];
	}
	mean = sum/IR_points.size();

	for(i=0;i<IR_points.size();i++)
	{
		s += (src.data[IR_points[i].y*src.cols+IR_points[i].x] - mean)*(src.data[IR_points[i].y*src.cols+IR_points[i].x] - mean);
	}

	stand = sqrt(s/IR_points.size());
	
	int T_low = mean - 3.5*stand;
	int T_high = mean + 2.5*stand;

	for(i = 0; i < ES_points.size(); i++)
	{
		if(src.data[ES_points[i].y*src.cols+ES_points[i].x] < T_low || src.data[ES_points[i].y*src.cols+ES_points[i].x] > T_high)
		{
			eyelash_points.push_back(ES_points[i]);
		}
	}

	for(i = 0; i < IR_points.size(); i++)
	{
		if(src.data[IR_points[i].y*src.cols+IR_points[i].x] < T_low || src.data[IR_points[i].y*src.cols+IR_points[i].x] > T_high)
		{
			eyelash_points.push_back(IR_points[i]);
		}
	}

	return 0;
}

int IrisRecognizer::iris_area_location(Mat& src, Point pupil_center, int pupil_rds, Point iris_center, int iris_rds, Mat& iris_mask)
{
	int i, j;
	Vector<Point> ES_points, IR_points;

	for(i = iris_center.y; i < iris_center.y + 0.5*iris_rds; i++)
		for(j = iris_center.x - 0.5*iris_rds; j < iris_center.x + 0.5*iris_rds; j++)
		{
			double dist_to_iris = sqrt(((double)i-iris_center.y)*(i-iris_center.y)+(j-iris_center.x)*(j-iris_center.x));
			double dist_to_pupil = sqrt(((double)i-pupil_center.y)*(i-pupil_center.y)+(j-pupil_center.x)*(j-pupil_center.x));
			if(dist_to_iris < 0.5*iris_rds && dist_to_pupil > pupil_rds)
			{
				IR_points.push_back(cv::Point(j,i));
			}
		}

	float sum=0,s=0,mean,stand;
	for(i = 0; i < IR_points.size(); i++)
	{
		sum += src.data[IR_points[i].y*src.cols+IR_points[i].x];
	}
	mean = sum/IR_points.size();

	for(i=0;i<IR_points.size();i++)
	{
		s += (src.data[IR_points[i].y*src.cols+IR_points[i].x] - mean)*(src.data[IR_points[i].y*src.cols+IR_points[i].x] - mean);
	}

	stand = sqrt(s/IR_points.size());

	int T_low = mean - 3.5*stand;
	int T_high = mean + 2.5*stand;

	cv::Point left_top, right_bottom;
	left_top.x = (iris_center.x - 1.2*iris_rds)>5?(iris_center.x - 1.2*iris_rds):5;
	left_top.y = (iris_center.y - 1.2*iris_rds)>5?(iris_center.y - 1.2*iris_rds):5;
	right_bottom.x = (iris_center.x + 1.2*iris_rds)<(src.cols - 5)?(iris_center.x + 1.2*iris_rds):(src.cols - 5);
	right_bottom.y = (iris_center.y + 1.2*iris_rds)<(src.rows - 5)?(iris_center.y + 1.2*iris_rds):(src.rows - 5);
	
	for(i = left_top.y; i < right_bottom.y; i++)
		for(j = left_top.x; j < right_bottom.x; j++)
		{
			double dist_to_iris = sqrt(((double)i-iris_center.y)*(i-iris_center.y)+(j-iris_center.x)*(j-iris_center.x));
			double dist_to_pupil = sqrt(((double)i-pupil_center.y)*(i-pupil_center.y)+(j-pupil_center.x)*(j-pupil_center.x));
			if(dist_to_iris < iris_rds && dist_to_pupil > pupil_rds)
			{
				if(src.data[i*src.cols+j] < T_low || src.data[i*src.cols+j] > T_high)
				{
					iris_mask.data[i*src.cols+j] = 0;
				}
				else
				{
					iris_mask.data[i*src.cols+j] = 255;
				}
			}
		}

	return 0;
}

int IrisRecognizer::iris_normalization(Mat& src, Mat& dst, Mat& iris_mask, Mat& mask_dst, Point pupil_center, int pupil_rds, Point iris_center, int iris_rds, int radpixels, int angulardiv)
{
	int radiuspixels;
	int angledivisions;
	int width, height;
	double r;
	double *theta, *b, xcosmat, xsinmat, rmat;
	double *xo, *yo;
	int i, j;
	double x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil, ox, oy;
	int sgn;
	double phi;
	double a;
	int *x, *y, *xp, *yp;
	int len;
	double sum, avg;
	int count;

	double pi = 3.14159265;

	radiuspixels = radpixels + 2;
	angledivisions = angulardiv - 1;

	theta = (double*)malloc(sizeof(double)*(angledivisions+1));

	for (i = 0; i<angledivisions+1; i++)
		theta[i] = 2*i*pi/angledivisions;

	x_iris = (double)iris_center.x;
	y_iris = (double)iris_center.y;
	r_iris = (double)iris_rds;
	x_pupil = (double)pupil_center.x;
	y_pupil = (double)pupil_center.y;
	r_pupil = (double)pupil_rds;

	//calculate displacement of pupil center from the iris center
	ox = x_pupil - x_iris;
	oy = y_pupil - y_iris;

	if(ox <= 0)
		sgn = -1;
	else
		sgn = 1;

	if(ox == 0 && oy > 0)
		sgn = 1;

	a = ox*ox+oy*oy;

	if(ox == 0)
		phi = pi/2;
	else
		phi = atan(oy/ox);

	b = (double*)malloc(sizeof(double)*(angledivisions+1));

	width = angledivisions+1;
	height = radiuspixels-2;
	xo = (double*)malloc(sizeof(double)*(radiuspixels-2)*(angledivisions+1));
	yo = (double*)malloc(sizeof(double)*(radiuspixels-2)*(angledivisions+1));

	for(i = 0; i < angledivisions+1; i++)
	{
		b[i] = sgn*cos(pi-phi-theta[i]);
		r = sqrt(a)*b[i]+sqrt(a*b[i]*b[i]-(a-r_iris*r_iris));
		r -= r_pupil;

		// calculate cartesian location of each data point around the circular iris region
		xcosmat = cos(theta[i]);
		xsinmat = sin(theta[i]);
		/* exclude values at the boundary of the pupil iris border, and the iris scelra border
		   as these may not correspond to areas in the iris region and will introduce noise.		
		   ie don't take the outside rings as iris data.*/

		for (j = 0; j<radiuspixels; j++)
		{
			rmat = r*j/(radiuspixels-1);
			rmat += r_pupil;
			if (j>0 && j<radiuspixels-1)
			{
				xo[(j-1)*(angledivisions+1)+i] = rmat*xcosmat+x_pupil;
				yo[(j-1)*(angledivisions+1)+i] = -rmat*xsinmat+y_pupil;
			}
		}
	}


	dst = Mat::zeros(height,width,CV_8U);
	mask_dst = Mat::zeros(height,width,CV_8U);
	interp2(src,xo,yo,width,height,dst);
	interp2(iris_mask,xo,yo,width,height,mask_dst);

	//for(i = 0; i < height; i++)
	//	for(j = 0; j < width; j++)
	//	{
	//		if(mask_dst.data[i*width+j] == 0)
	//			dst.data[i*width+j] = 255;
	//		else if(_isnan(dst.data[i*width+j]))
	//			dst.data[i*width+j] = 255;
	//	}
	
	return 0;
}

int IrisRecognizer::blocproc(Mat& mat)
{
	//FILE *recode_for_enhacnce;
	//recode_for_enhacnce = fopen("recode_for_enhacnce.txt","ab");
	//fprintf(recode_for_enhacnce, "In blocproc\r\n");
	//fclose(recode_for_enhacnce);

	int sum = 0;
	for(int i=0;i<mat.rows;i++)  
	{  
		for(int j=0;j<mat.cols;j++)  
		{  
			sum = sum + mat.data[i*mat.cols+j];
		}
	}

	return sum;
}

void IrisRecognizer::enhacnce(Mat& mat,int n, Mat& result)
{
// 	FILE *recode_for_enhacnce;
// 	recode_for_enhacnce = fopen("recode_for_enhacnce.txt","ab");
// 	fprintf(recode_for_enhacnce, "cvmGet(mat,0,0)!\r\n");
// 	fclose(recode_for_enhacnce);

	if (n == 0)
	{
		cout<<"In Function enhance, parameter n cannot set as zero!"<<endl;
		return;
	}

	Mat bloc_sum_mat(mat.rows/n,mat.cols/n,CV_64FC1);
	Mat resize_mat(mat.rows,mat.cols, CV_64FC1 );
	Mat hisequl_mat(mat.rows,mat.cols, CV_8UC1);
	Mat minus_mat(mat.rows,mat.cols, CV_8UC1);
	Mat blocmat(n,n, CV_64FC1 );

// 	CvMat qstub;
	for(int i=0;i<mat.rows;i+=n)  
	{  
		for(int j=0;j<mat.cols;j+=n)  
		{  
			blocmat = mat(cvRect(j,i, n,n) );

			bloc_sum_mat.at<double>(i/n,j/n) = (double)blocproc(blocmat)/(double)(n*n);
		}
	}

	resize(bloc_sum_mat,resize_mat,cv::Size(mat.cols,mat.rows),CV_INTER_CUBIC);
	//cvResize( bloc_sum_mat, resize_mat, CV_INTER_CUBIC);

	for(int i=0;i<mat.rows;i++)  
	{  
		for(int j=0;j<mat.cols;j++)  
		{  
			minus_mat.at<uchar>(i,j) = mat.at<uchar>(i,j) - (int)resize_mat.at<double>(i,j);
		}
	}

	//CvMat* minus_mat_8  = cvCreateMat(mat.rows,mat.cols, CV_8UC1);
	//cvConvertScale(minus_mat, minus_mat_8, 1, 0 ); 
	equalizeHist(minus_mat,hisequl_mat);
	equalizeHist(hisequl_mat,result);


	//IplImage* img = cvCreateImage(cvGetSize(minus_mat_8),8,1);
	//cvGetImage(hisequl_mat,img);
	//cvSaveImage("enhance.bmp",img);

	//cvReleaseMat(&bloc_sum_mat);
	//cvReleaseMat(&resize_mat);
	//cvReleaseMat(&hisequl_mat);
	//cvReleaseMat(&minus_mat);
	//cvReleaseMat(&minus_mat_8);
	//cvReleaseMat(&blocmat);
}

int IrisRecognizer::iris_extraction(Mat& img, int** iris_template_result, int** mask_array_result,int* template_width, int* template_height)
{
	int ret = 0;

// 	save_image_index_++;

// 	equalizeHist(img, img);

	GaussianBlur(img, img, Size(5,5), 0 , 0);

	//	GaussianBlur(img, img, Size(5,5), 0 , 0);

	int i,j;

	cv::Point pupil_center(0,0), iris_center(0,0), upper_eyelid_center(0,0), lower_eyelid_center(0,0);
	int pupil_rds = 0, iris_rds = 0, upper_eyelid_rds = 0, lower_eyelid_rds = 0;

	/*****if eye image is too large, downsample eye image. the scale (times) is image's cols/200 *****/
	int times = img.cols/200;
	if(times == 0)
	{
		times = 1;
	}

	/*****downsample eye image *****/
	int iMinR = min_pupil_rds_/times, iMaxR = max_pupil_rds_/times;

	Mat src_cut = Mat::zeros((img.rows-img.rows%times),(img.cols-img.cols%times),CV_8UC1);
	for(i = 0; i < src_cut.rows; i++)
		for(j = 0; j < src_cut.cols; j++)
		{
			src_cut.data[i*src_cut.cols + j] = img.data[i*img.cols + j];
		}

	Mat src_small = Mat::zeros(src_cut.rows/times,src_cut.cols/times, CV_8UC1);

	for(i = 0; i< src_small.rows; i++)
		for(j = 0; j < src_small.cols; j++)
		{
			src_small.data[i*src_small.cols + j] = src_cut.data[i*times*src_cut.cols +j*times];
		}

	GaussianBlur(src_small, src_small, Size(5,5), 0 , 0);

	/*****Locate pupil position on downsample image, output are pupil center postion and pupil radius *****/
	ret = pupil_coarse_location(src_small, iMinR, iMaxR, pupil_center, pupil_rds);
	if(ret != 0 || pupil_center.x + pupil_center.y < 10)
	{
  		cout<<"pupil_coarse_location Failed!"<<endl;
		return -1;
	}

//   	cout<<"After pupil_coarse_location!"<<endl;

	/*****Locate iris position on downsample image, output are iris center postion and iris radius *****/
	ret = iris_location(src_small, pupil_center, pupil_rds, iris_center, iris_rds);

// 	cout<<"After iris location, but calculate iris_pupil_dist"<<endl;
	/********if the distance between the center of pupil and the center of iris is longer than 10 pixels, iris or pupil location failed and give up this frame.*********/
	double iris_pupil_dist = (pupil_center.x - iris_center.x)*(pupil_center.x - iris_center.x) + (pupil_center.y - iris_center.y)*(pupil_center.y - iris_center.y);
	if(ret != 0 || iris_pupil_dist*times*times > 100 || iris_center.x + iris_center.y < 10)
	{
		return -3;
	}

	circle(img,cv::Point(pupil_center.x*times,pupil_center.y*times),pupil_rds*times,cv::Scalar(255),2);
	circle(img,cv::Point(iris_center.x*times,iris_center.y*times),iris_rds*times,cv::Scalar(255),2);

//   	cout<<"After iris_location!"<<endl;


	Mat iris_mask = src_small.clone();

	ret = iris_area_location(src_small,pupil_center,pupil_rds,iris_center,iris_rds,iris_mask);

	if(ret != 0)
	{
//  		cout<<"iris_area_location Failed!"<<endl;
		return -4;
	}

//   	cout<<"After iris_area_location!"<<endl;

	/*****Using a circle to fit upper eyelid. Location the circle for upper eyelid, include center and radius *****/
	ret = upper_eyelid_location(src_small,iris_mask,iris_center,iris_rds,upper_eyelid_center,upper_eyelid_rds);

	/*****If the radius of upper eyelid smaller than iris radius, upper eyelid location failed.*****/
	if(ret != 0 || upper_eyelid_rds <= iris_rds)
	{
		cout<<"upper_eyelid_location Failed!"<<endl;
		return -5;
	}

//   	cout<<"After upper_eyelid_location!"<<endl;

	/*****Using a circle to fit lower eyelid. Location the circle for lower eyelid, include center and radius *****/
	ret = lower_eyelid_location(src_small,iris_center,iris_rds,lower_eyelid_center,lower_eyelid_rds);

	/*****If the radius of lower eyelid smaller than iris radius, lower eyelid location failed.*****/
 	if(ret != 0 || lower_eyelid_rds <= iris_rds)
 	{
   		cout<<"Error lower_eyelid_location!"<<endl;
 		return -6;
 	}

//   	cout<<"After lower_eyelid_location!"<<endl;
 
	pupil_center.x = pupil_center.x * times;
	pupil_center.y = pupil_center.y * times;
	pupil_rds = pupil_rds * times;

	iris_center.x = iris_center.x * times;
	iris_center.y = iris_center.y * times;
	iris_rds = iris_rds * times;

	upper_eyelid_center.x = upper_eyelid_center.x*times;
	upper_eyelid_center.y = upper_eyelid_center.y*times;
	upper_eyelid_rds = upper_eyelid_rds*times;

	lower_eyelid_center.x = lower_eyelid_center.x*times;
	lower_eyelid_center.y = lower_eyelid_center.y*times;
	lower_eyelid_rds = lower_eyelid_rds*times;

	Vector<cv::Point> eyelash_points;

	/*****Find pixels which don't belong iris. *****/
	ret = eyelash_pixels_location(img,iris_center,iris_rds,pupil_center,pupil_rds,upper_eyelid_center,upper_eyelid_rds,eyelash_points);
	if(ret != 0)
	{
 		cout<<"Error eyelash_pixels_location!"<<endl;
		return -7;
	}

//   	cout<<"After eyelash_pixels_location!"<<endl;

	Mat mask = Mat::zeros(img.rows,img.cols,img.type());

	for(i = 0; i < mask.rows; i++)
		for(j = 0; j < mask.cols; j++)
		{
			int dist_to_top_eyelid = (i - upper_eyelid_center.y)*(i - upper_eyelid_center.y)+(j - upper_eyelid_center.x)*(j - upper_eyelid_center.x);
			int dist_to_bottom_eyelid = (i - lower_eyelid_center.y)*(i - lower_eyelid_center.y)+(j - lower_eyelid_center.x)*(j - lower_eyelid_center.x);
			int dist_to_iris = (i - iris_center.y)*(i - iris_center.y)+(j - iris_center.x)*(j - iris_center.x);
			int dist_to_pupil = (i - pupil_center.y)*(i - pupil_center.y)+(j - pupil_center.x)*(j - pupil_center.x);
			if(dist_to_top_eyelid >= upper_eyelid_rds*upper_eyelid_rds || dist_to_bottom_eyelid >= lower_eyelid_rds*lower_eyelid_rds || dist_to_iris >= iris_rds*iris_rds || dist_to_pupil <= pupil_rds*pupil_rds)
			{
				mask.data[i*mask.cols+j] = 0;
			}
			else
			{
				mask.data[i*mask.cols+j] = 255;
			}
		}

	for(i = 0; i < eyelash_points.size(); i++)
	{
		mask.data[eyelash_points[i].y*mask.cols+eyelash_points[i].x] = 0;
	}

	/*****normalize iris from circle to rectangle *****/
	Mat iris_normalized_img;
	Mat iris_mask_img;
	ret = iris_normalization(img,iris_normalized_img,mask,iris_mask_img,pupil_center,pupil_rds,iris_center,iris_rds,80,360);
	if(ret != 0)
	{
 		cout<<"error iris_normalization!"<<endl;
		return -8;
	}

//   	cout<<"after iris_normalization!"<<endl;

	Mat enhanced_iris;

	//iris_recognizer_->enhacnce(iris_normalized_img,5,enhanced_iris);
	equalizeHist(iris_normalized_img,iris_normalized_img);

	int minWaveLength=18;
	int mult=1;
	double sigmaOnf=0.5;

	filter iris_normalized_filter;
	IMAGE iris_mask_IMG;

	// 		imwrite("iris_normalized.bmp",iris_normalized_img);

	iris_normalized_filter.hsize[1] = iris_normalized_img.cols;
	iris_normalized_filter.hsize[0] = iris_normalized_img.rows;

	iris_normalized_filter.data = (double *)malloc(sizeof(double)*(iris_normalized_img.cols)*(iris_normalized_img.rows));
	for (int i = 0; i < iris_normalized_img.rows; i++)
	{
		for (int j = 0; j < iris_normalized_img.cols; j++)
		{
			iris_normalized_filter.data[i*iris_normalized_img.cols+j] = (double)iris_normalized_img.at<uchar>(i,j);
		}
	}

	iris_mask_IMG.hsize[1] = iris_mask_img.cols;
	iris_mask_IMG.hsize[0] = iris_mask_img.rows;

	iris_mask_IMG.data = (unsigned char*)malloc(sizeof(unsigned char)*(iris_mask_img.cols)*(iris_mask_img.rows));
	for (int i = 0; i < iris_mask_img.rows; i++)
	{
		for (int j = 0; j < iris_mask_img.cols; j++)
		{
			iris_mask_IMG.data[i*iris_mask_img.cols+j] = iris_mask_img.at<uchar>(i,j);
		}
	}

	/*****Encode normalized iris image *****/
	encode(&iris_normalized_filter,&iris_mask_IMG,1,minWaveLength,mult,sigmaOnf,iris_template_result,mask_array_result,template_width,template_height);

// 	cout<<"After encode!"<<endl;

 	circle(img,pupil_center,pupil_rds,cv::Scalar(255),2);
 	circle(img,iris_center,iris_rds,cv::Scalar(255),2);
 	circle(img,upper_eyelid_center,upper_eyelid_rds,cv::Scalar(255),2);
 	circle(img,lower_eyelid_center,lower_eyelid_rds,cv::Scalar(255),2);

// 	string temp_save_index = "./" + to_string(save_image_index_) + ".jpg";
// 	imwrite(temp_save_index,img);

	free(iris_normalized_filter.data);
	free(iris_mask_IMG.data);

// 	cout<<"Out of iris_segment function!"<<endl;
	return 0;
}

int IrisRecognizer::get_iris_images_for_display(Mat& left_iris, Mat& right_iris)
{
	if (!left_iris_for_display_.empty() && !right_iris_for_display_.empty())
	{
		left_iris = left_iris_for_display_;
		right_iris = right_iris_for_display_;
	}
	else
	{
		return -1;
	}

	return 0;	
}


