// ObjectTracking.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <srt3d/azure_kinect_camera.h>
#include <srt3d/body.h>

#include <srt3d/common.h>
#include <srt3d/normal_viewer.h>
#include <srt3d/occlusion_renderer.h>
#include <srt3d/region_modality.h>
#include <srt3d/EdgeModality.h>

#include <srt3d/renderer_geometry.h>
#include <srt3d/tracker.h>
#include <srt3d/teture_render.h>
#include <srt3d/edge_model.h>
#include <srt3d/EdgeModality.h>
#include <srt3d/rbot_evaluator.h>

#include <Eigen/Geometry>
#include <filesystem>
#include <memory>
#include <string>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <opencv2/opencv_modules.hpp>

#include<opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ObjectTrackingClass.h"
#include <opencv2/imgproc/types_c.h>

#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/aruco.hpp>

#include "EdgeTrackingUtil.h"

#include <srt3d/LineModality.h>
#include <srt3d/line_model.h>

#include <opencv2/ml.hpp>

#include <stdio.h>
#include <time.h>
#include <opencv2/core/eigen.hpp>

using namespace cv;
void CalculateAlphaBlend_2(const cv::Mat &camera_image,
	const cv::Mat &renderer_image,
	cv::Mat *viewer_image, float opacity_);
cv::Matx44f rotationMatrix(float angle, cv::Vec3f axis)
{
	angle = (angle / 180)*CV_PI;

	float s = sin(angle);
	float c = cos(angle);
	float mc = 1.0f - c;

	float len = norm(axis);

	if (len == 0)
	{
		// avoid zero division error
		return cv::Matx44f::eye();
	}

	axis /= len;
	float x = axis[0];
	float y = axis[1];
	float z = axis[2];

	return cv::Matx44f(x * x * mc + c, x * y * mc - z * s, x * z * mc + y * s, 0,
		x * y * mc + z * s, y * y * mc + c, y * z * mc - x * s, 0,
		x * z * mc - y * s, y * z * mc + x * s, z * z * mc + c, 0,
		0, 0, 0, 1);
}
void TransVector2matrix(float tx, float ty, float tz, float a, float b, float c, srt3d::Transform3fA &matrix)
{
	cv::Matx44f body1_translationMatrix = cv::Matx44f(1, 0, 0, tx,
		0, 1, 0, ty,
		0, 0, 1, tz,
		0, 0, 0, 1);

	cv::Matx44f T_cm = body1_translationMatrix * rotationMatrix(a, cv::Vec3f(1, 0, 0)) * rotationMatrix(b, cv::Vec3f(0, 1, 0))*rotationMatrix(c, cv::Vec3f(0, 0, 1))
		*cv::Matx44f::eye();
	matrix.matrix() << T_cm(0, 0), T_cm(0, 1), T_cm(0, 2), T_cm(0, 3),
		T_cm(1, 0), T_cm(1, 1), T_cm(1, 2), T_cm(1, 3),
		T_cm(2, 0), T_cm(2, 1), T_cm(2, 2), T_cm(2, 3),
		T_cm(3, 0), T_cm(3, 1), T_cm(3, 2), T_cm(3, 3);
	//将世界坐标系到零件的转换关系进行求逆，得到模型到世界坐标系的转换矩阵
	//cout << matrix .matrix()<< endl;
	matrix = matrix.inverse();
}
void CalculateAlphaBlend(const cv::Mat &camera_image,
	const cv::Mat &renderer_image,
	cv::Mat *viewer_image, float opacity_) {
	// Declare variables
	int v, u;
	const cv::Vec3b *ptr_camera_image;
	const cv::Vec4b *ptr_renderer_image;
	cv::Vec3b *ptr_viewer_image;
	const uchar *val_camera_image;
	const uchar *val_renderer_image;
	uchar *val_viewer_image;
	float alpha, alpha_inv;
	float alpha_scale = opacity_ / 255.0f;

	// Iterate over all pixels
	for (v = 0; v < camera_image.rows; ++v) {
		ptr_camera_image = camera_image.ptr<cv::Vec3b>(v);
		ptr_renderer_image = renderer_image.ptr<cv::Vec4b>(v);
		ptr_viewer_image = viewer_image->ptr<cv::Vec3b>(v);
		for (u = 0; u < camera_image.cols; ++u) {
			val_camera_image = ptr_camera_image[u].val;
			val_renderer_image = ptr_renderer_image[u].val;
			val_viewer_image = ptr_viewer_image[u].val;

			// Blend images
			alpha = float(val_renderer_image[3]) * alpha_scale;
			alpha_inv = 1.0f - alpha;
			val_viewer_image[0] =
				char(val_camera_image[0] * alpha_inv + val_renderer_image[0] * alpha);
			val_viewer_image[1] =
				char(val_camera_image[1] * alpha_inv + val_renderer_image[1] * alpha);
			val_viewer_image[2] =
				char(val_camera_image[2] * alpha_inv + val_renderer_image[2] * alpha);
		}
	}
}
void CalculateAlphaBlend_2(const cv::Mat &camera_image,
	const cv::Mat &renderer_image,
	cv::Mat *viewer_image, float opacity_) {
	// Declare variables
	int v, u;
	const cv::Vec3b *ptr_camera_image;
	const cv::Vec4b *ptr_renderer_image;
	cv::Vec3b *ptr_viewer_image;
	const uchar *val_camera_image;
	const uchar *val_renderer_image;
	uchar *val_viewer_image;
	float alpha, alpha_inv;
	float alpha_scale = opacity_ / 255.0f;

	// Iterate over all pixels
	for (v = 0; v < camera_image.rows; ++v) {
		ptr_camera_image = camera_image.ptr<cv::Vec3b>(v);
		ptr_renderer_image = renderer_image.ptr<cv::Vec4b>(v);
		ptr_viewer_image = viewer_image->ptr<cv::Vec3b>(v);
		for (u = 0; u < camera_image.cols; ++u) {
			val_camera_image = ptr_camera_image[u].val;
			val_renderer_image = ptr_renderer_image[u].val;
			val_viewer_image = ptr_viewer_image[u].val;

			// Blend images
			alpha = float(val_renderer_image[3]) * alpha_scale;
			alpha_inv = 1.0f - alpha;
			val_viewer_image[0] =
				char(val_camera_image[0] * alpha_inv + val_renderer_image[2] * alpha);
			val_viewer_image[1] =
				char(val_camera_image[1] * alpha_inv + val_renderer_image[1] * alpha);
			val_viewer_image[2] =
				char(val_camera_image[2] * alpha_inv + val_renderer_image[0] * alpha);
}
	}
}

void Edge_fine()
{
	//读入图像
	cv::Mat img = cv::imread("contours_img.jpg");
	cv::Mat gray;
	cv::Mat Binary;
	//进行二值化
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::inRange(gray, 200, 255, Binary);
	cv::Mat CopyImg;
	int rows = Binary.rows;
	int cols = Binary.cols;


	std::vector<cv::Point2l> PointSaver1;
	//开始第一轮判断
	for (int i{ 1 }; i < rows - 1; i++)
	{
		uchar* high = Binary.ptr<uchar>(i - 1);
		uchar* mid = Binary.ptr<uchar>(i);
		uchar* low = Binary.ptr<uchar>(i + 1);

		for (int j{ 1 }; j < cols - 1; j++)
		{
			int a1 = mid[j];
			int a2 = high[j];
			int a3 = high[j + 1];
			int a4 = mid[j + 1];
			int a5 = low[j + 1];
			int a6 = low[j];
			int a7 = low[j - 1];
			int a8 = mid[j - 1];
			int a9 = high[j - 1];
			int a[9] = { a1,a2,a3,a4,a5,a6,a7,a8,a9 };

			bool req1 = true;
			bool req2 = true;
			bool req3 = true;
			bool req4 = true;

			//条件1 八领域的和
			if (a[0 == 255])
			{
				int req1_sum{ 0 };
				req1_sum = a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8];

				//std::cout << "req1_sum: " << req1_sum << std::endl;
				if (req1_sum >= 510 && req1_sum <= 1530)
				{
					//std::cout << "66666" << std::endl;
					req1 = true;
				}
				else req1 = false;

				//条件2  01的模式
				int req2_sum{ 0 };
				//	std::cout << "req2_sum: " << req2_sum << std::endl;
				for (int k = 2; k < 9; k++)
				{
					if (a[k] == 255 && a[k - 1] == 0)
					{
						req2_sum += 1;
					}
				}
				if (req2_sum == 1) req2 = true;
				else req2 = false;
				//条件三
				int req3_sum = a[1] * a[3] * a[5];
				//std::cout << "req3_sum: " << req3_sum << std::endl;

				if (req3_sum == 0) req3 = true;
				else req3 = false;
				//条件四
				int req4_sum = a[3] * a[5] * a[7];
				//	std::cout << "req3_sum: " << req3_sum << std::endl;

				if (req4_sum == 0) req4 = true;
				else req4 = false;
				if (req1 && req2 && req3 && req4)
				{
					PointSaver1.push_back(Point2l(i, j));
				}
			}
		}
	}
	for (int l = 0; l < PointSaver1.size(); l++)
	{
		uchar* ptr = Binary.ptr<uchar>(PointSaver1[l].x);
		std::cout << "PointSaver1[l].x: " << PointSaver1[l].x << "PointSaver1[l].y: " << PointSaver1[l].y << std::endl;
		ptr[PointSaver1[l].y] = 0;
	}

	//第二轮判断
	std::vector<cv::Point2l> PointSaver2;
	for (int i{ 1 }; i < rows - 1; i++)
	{
		uchar* high = Binary.ptr<uchar>(i - 1);
		uchar* mid = Binary.ptr<uchar>(i);
		uchar* low = Binary.ptr<uchar>(i + 1);

		for (int j{ 1 }; j < cols - 1; j++)
		{
			int a1 = mid[j];
			int a2 = high[j];
			int a3 = high[j + 1];
			int a4 = mid[j + 1];
			int a5 = low[j + 1];
			int a6 = low[j];
			int a7 = low[j - 1];
			int a8 = mid[j - 1];
			int a9 = high[j - 1];
			int a[9] = { a1,a2,a3,a4,a5,a6,a7,a8,a9 };

			bool req1 = true;
			bool req2 = true;
			bool req3 = true;
			bool req4 = true;

			//条件1 八领域的和
			if (a[0 == 255])
			{
				int req1_sum{ 0 };
				req1_sum = a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8];

				//std::cout << "req1_sum: " << req1_sum << std::endl;
				if (req1_sum >= 510 && req1_sum <= 1530)
				{
					//std::cout << "66666" << std::endl;
					req1 = true;
				}
				else req1 = false;

				//条件2  01的模式
				int req2_sum{ 0 };
				//	std::cout << "req2_sum: " << req2_sum << std::endl;
				for (int k = 2; k < 9; k++)
				{
					if (a[k] == 255 && a[k - 1] == 0)
					{
						req2_sum += 1;
					}
				}
				if (req2_sum == 1) req2 = true;
				else req2 = false;
				//条件三
				int req3_sum = a[1] * a[3] * a[5];
				//std::cout << "req3_sum: " << req3_sum << std::endl;

				if (req3_sum == 0) req3 = true;
				else req3 = false;
				//条件四
				int req4_sum = a[1] * a[3] * a[7];
				//	std::cout << "req3_sum: " << req3_sum << std::endl;

				if (req4_sum == 0) req4 = true;
				else req4 = false;
				if (req1 && req2 && req3 && req4)
				{
					PointSaver2.push_back(Point2l(i, j));
				}
			}
		}
	}
	for (int l = 0; l < PointSaver2.size(); l++)
	{
		uchar* ptr = Binary.ptr<uchar>(PointSaver2[l].x);
		std::cout << "PointSaver[l].x: " << PointSaver2[l].x << "PointSaver[l].y: " << PointSaver2[l].y << std::endl;
		ptr[PointSaver2[l].y] = 0;
	}

	cv::imshow("SrcImg", Binary);
	cv::imwrite("Binary.jpg", Binary);
	std::cout << "saver_size: " << PointSaver2.size() << std::endl;
	cv::waitKey(0);
}


// RGB to Gray scale
cv::Mat BGR2GRAY(cv::Mat img) {
	// get height and width
	int height = img.rows;
	int width = img.cols;
	int channel = img.channels();

	// prepare output
	cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

	// BGR -> Gray
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			out.at<uchar>(y, x) = (int)((float)img.at<cv::Vec3b>(y, x)[0] * 0.0722 + \
				(float)img.at<cv::Vec3b>(y, x)[1] * 0.7152 + \
				(float)img.at<cv::Vec3b>(y, x)[2] * 0.2126);
		}
	}
	return out;
}

float clip(float value, float min, float max) {
	return fmin(fmax(value, 0), 255);
}

// gaussian filter
cv::Mat gaussian_filter(cv::Mat img, double sigma, int kernel_size) {
	int height = img.rows;
	int width = img.cols;
	int channel = img.channels();

	// prepare output
	cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
	if (channel == 1) {
		out = cv::Mat::zeros(height, width, CV_8UC1);
	}

	// prepare kernel
	int pad = floor(kernel_size / 2);
	int _x = 0, _y = 0;
	double kernel_sum = 0;

	// get gaussian kernel
	float kernel[5][5];

	for (int y = 0; y < kernel_size; y++) {
		for (int x = 0; x < kernel_size; x++) {
			_y = y - pad;
			_x = x - pad;
			kernel[y][x] = 1 / (2 * CV_PI * sigma * sigma) * exp(-(_x * _x + _y * _y) / (2 * sigma * sigma));
			kernel_sum += kernel[y][x];
		}
	}

	for (int y = 0; y < kernel_size; y++) {
		for (int x = 0; x < kernel_size; x++) {
			kernel[y][x] /= kernel_sum;
		}
	}

	// filtering
	double v = 0;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// for BGR
			if (channel == 3) {
				for (int c = 0; c < channel; c++) {
					v = 0;
					for (int dy = -pad; dy < pad + 1; dy++) {
						for (int dx = -pad; dx < pad + 1; dx++) {
							if (((x + dx) >= 0) && ((y + dy) >= 0) && ((x + dx) < width) && ((y + dy) < height)) {
								v += (double)img.at<cv::Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
							}
						}
					}
					out.at<cv::Vec3b>(y, x)[c] = (uchar)clip(v, 0, 255);
				}
			}
			else {
				// for Gray
				v = 0;
				for (int dy = -pad; dy < pad + 1; dy++) {
					for (int dx = -pad; dx < pad + 1; dx++) {
						if (((x + dx) >= 0) && ((y + dy) >= 0) && ((x + dx) < width) && ((y + dy) < height)) {
							v += (double)img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
						}
					}
				}
				out.at<uchar>(y, x) = (uchar)clip(v, 0, 255);
			}
		}
	}
	return out;
}

// Sobel filter
cv::Mat sobel_filter(cv::Mat img, int kernel_size, bool horizontal) {
	int height = img.rows;
	int width = img.cols;
	int channel = img.channels();

	// prepare output
	cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

	// prepare kernel
	double kernel[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

	if (horizontal) {
		kernel[0][1] = 0;
		kernel[0][2] = -1;
		kernel[1][0] = 2;
		kernel[1][2] = -2;
		kernel[2][0] = 1;
		kernel[2][1] = 0;
	}

	int pad = floor(kernel_size / 2);

	double v = 0;

	// filtering  
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			v = 0;
			for (int dy = -pad; dy < pad + 1; dy++) {
				for (int dx = -pad; dx < pad + 1; dx++) {
					if (((y + dy) >= 0) && ((x + dx) >= 0) && ((y + dy) < height) && ((x + dx) < width)) {
						v += (double)img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
					}
				}
			}
			out.at<uchar>(y, x) = (uchar)clip(v, 0, 255);
		}
	}
	return out;
}

// get edge
cv::Mat get_edge(cv::Mat fx, cv::Mat fy) {
	// get height and width
	int height = fx.rows;
	int width = fx.cols;

	// prepare output
	cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

	double _fx, _fy;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			_fx = (double)fx.at<uchar>(y, x);
			_fy = (double)fy.at<uchar>(y, x);

			out.at<uchar>(y, x) = (uchar)clip(sqrt(_fx * _fx + _fy * _fy), 0, 255);
		}
	}

	return out;
}

// get angle
cv::Mat get_angle(cv::Mat fx, cv::Mat fy) {
	// get height and width
	int height = fx.rows;
	int width = fx.cols;

	// prepare output
	cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

	double _fx, _fy;
	double angle;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			_fx = fmax((double)fx.at<uchar>(y, x), 0.000001);
			_fy = (double)fy.at<uchar>(y, x);

			angle = atan2(_fy, _fx);
			angle = angle / CV_PI * 180;

			if (angle < -22.5) {
				angle = 180 + angle;
			}
			else if (angle >= 157.5) {
				angle = angle - 180;
			}

			//std::cout << angle << " " ;

			// quantization
			if (angle <= 22.5) {
				out.at<uchar>(y, x) = 0;
			}
			else if (angle <= 67.5) {
				out.at<uchar>(y, x) = 45;
			}
			else if (angle <= 112.5) {
				out.at<uchar>(y, x) = 90;
			}
			else {
				out.at<uchar>(y, x) = 135;
			}
		}
	}

	return out;
}


// non maximum suppression
cv::Mat non_maximum_suppression(cv::Mat angle, cv::Mat edge) {
	int height = angle.rows;
	int width = angle.cols;
	int channel = angle.channels();

	int dx1, dx2, dy1, dy2;
	int now_angle;

	// prepare output
	cv::Mat _edge = cv::Mat::zeros(height, width, CV_8UC1);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			now_angle = angle.at<uchar>(y, x);
			// angle condition
			if (now_angle == 0) {
				dx1 = -1;
				dy1 = 0;
				dx2 = 1;
				dy2 = 0;
			}
			else if (now_angle == 45) {
				dx1 = -1;
				dy1 = 1;
				dx2 = 1;
				dy2 = -1;
			}
			else if (now_angle == 90) {
				dx1 = 0;
				dy1 = -1;
				dx2 = 0;
				dy2 = 1;
			}
			else {
				dx1 = -1;
				dy1 = -1;
				dx2 = 1;
				dy2 = 1;
			}

			if (x == 0) {
				dx1 = fmax(dx1, 0);
				dx2 = fmax(dx2, 0);
			}
			if (x == (width - 1)) {
				dx1 = fmin(dx1, 0);
				dx2 = fmin(dx2, 0);
			}
			if (y == 0) {
				dy1 = fmax(dy1, 0);
				dy2 = fmax(dy2, 0);
			}
			if (y == (height - 1)) {
				dy1 = fmin(dy1, 0);
				dy2 = fmin(dy2, 0);
			}

			// if pixel is max among adjuscent pixels, pixel is kept
			if (fmax(fmax(edge.at<uchar>(y, x), edge.at<uchar>(y + dy1, x + dx1)), edge.at<uchar>(y + dy2, x + dx2)) == edge.at<uchar>(y, x)) {
				_edge.at<uchar>(y, x) = edge.at<uchar>(y, x);
			}
		}
	}

	return _edge;
}


// Canny step 2
int Canny_step2(cv::Mat img) {
	// BGR -> Gray
	cv::Mat gray = BGR2GRAY(img);

	// gaussian filter
	cv::Mat gaussian = gaussian_filter(gray, 1.4, 5);

	// sobel filter (vertical)
	cv::Mat fy = sobel_filter(gaussian, 3, false);

	// sobel filter (horizontal)
	cv::Mat fx = sobel_filter(gaussian, 3, true);

	// get edge
	cv::Mat edge = get_edge(fx, fy);

	// get angle
	cv::Mat angle = get_angle(fx, fy);

	// edge non-maximum suppression
	edge = non_maximum_suppression(angle, edge);

	//cv::imwrite("out.jpg", out);
	cv::imshow("answer(edge)", edge);
	cv::imwrite("edge.jpg", edge);
	cv::imshow("answer(angle)", angle);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
void genaratePsf(Mat &psf, cv::Point &anchor, double len, double angle)
{
	//生成卷积核和锚点
	double half = len / 2;
	double alpha = (angle - floor(angle / 180) * 180) / 180 * CV_PI;
	double cosalpha = cos(alpha);
	double sinalpha = sin(alpha);
	int xsign;
	if (cosalpha < 0) {
		xsign = -1;
	}
	else {
		if (angle == 90) {
			xsign = 0;
		}
		else {
			xsign = 1;
		}
	}
	int psfwdt = 1;
	//模糊核大小
	int sx = (int)fabs(half*cosalpha + psfwdt * xsign - len * FLT_EPSILON);
	int sy = (int)fabs(half*sinalpha + psfwdt - len * FLT_EPSILON);
	cv::Mat_<double> psf1(sy, sx, CV_64F);

	//psf1是左上角的权值较大，越往右下角权值越小的核。
	//这时运动像是从右下角到左上角移动
	for (int i = 0; i < sy; i++) {
		double* pvalue = psf1.ptr<double>(i);
		for (int j = 0; j < sx; j++) {
			pvalue[j] = i * fabs(cosalpha) - j * sinalpha;

			double rad = sqrt(i*i + j * j);
			if (rad >= half && fabs(pvalue[j]) <= psfwdt) {
				double temp = half - fabs((j + pvalue[j] * sinalpha) / cosalpha);
				pvalue[j] = sqrt(pvalue[j] * pvalue[j] + temp * temp);
			}
			pvalue[j] = psfwdt + FLT_EPSILON - fabs(pvalue[j]);
			if (pvalue[j] < 0) {
				pvalue[j] = 0;
			}
		}
	}
	//    运动方向是往左上运动，锚点在（0，0）
	anchor.x = 0;
	anchor.y = 0;
	//    运动方向是往右上角移动，锚点一个在右上角
	//    同时，左右翻转核函数，使得越靠近锚点，权值越大
	if (angle < 90 && angle>0) {
		flip(psf1, psf1, 1);
		anchor.x = psf1.cols - 1;
		anchor.y = 0;
	}
	else if (angle > -90 && angle < 0) {    //同理：往右下角移动
		flip(psf1, psf1, -1);
		anchor.x = psf1.cols - 1;
		anchor.y = psf1.rows - 1;
	}
	else if (angle < -90) {   //同理：往左下角移动
		flip(psf1, psf1, 0);
		anchor.x = 0;
		anchor.y = psf1.rows - 1;
	}
	/*保持图像总能量不变，归一化矩阵*/
	double sum = 0;
	for (int i = 0; i < sy; i++) {
		for (int j = 0; j < sx; j++) {
			sum += psf1[i][j];
		}
	}
	psf = psf1 / sum;
}

void getImgsPoints(vector<Mat> imgs, vector<vector<Point2f>>& imgsPoints, Size board_size)
{
	for (int i = 0; i < imgs.size(); i++)
	{
		Mat img1 = imgs[i];
		Mat gray1;
		cvtColor(img1, gray1, COLOR_BGR2GRAY);
		vector<Point2f> img1_points;
		bool found = findChessboardCorners(gray1, board_size, img1_points);  //计算方格标定板角点
		if (found)
		{
			find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));  //细化方格标定板角点坐标
			imgsPoints.push_back(img1_points);
		}	
	}
}
// Euler rotation in XYZ format to rotation matrix
cv::Vec3f Rot2EulerXYZ(const cv::Matx33f &R)
{
	cv::Vec3f eulerxyz(0, 0, 0);

	if (R(0, 2) < 1)
	{
		if (R(0, 2) > -1)
		{
			eulerxyz[1] = asin(R(0, 2));
			eulerxyz[0] = atan2(-R(1, 2), R(2, 2));
			eulerxyz[2] = atan2(-R(0, 1), R(0, 0));
		}
		else
		{
			eulerxyz[1] = -CV_PI / 2;
			eulerxyz[0] = -atan2(R(1, 0), R(1, 1));
			eulerxyz[2] = 0;
		}
	}
	else
	{
		eulerxyz[1] = CV_PI / 2;
		eulerxyz[0] = atan2(R(1, 0), R(1, 1));
		eulerxyz[2] = 0;
	}
	eulerxyz[1] = eulerxyz[1] * 180 / CV_PI;
	eulerxyz[0] = eulerxyz[0] * 180 / CV_PI;
	eulerxyz[2] = eulerxyz[2] * 180 / CV_PI;
	return eulerxyz;
}
cv::Matx33f eulerXYZ2Rot(float x, float y, float z)
{
	cv::Matx33f R(1, 0, 0,
		0, 1, 0,
		0, 0, 1);

	x = x * CV_PI / 180;
	y = y * CV_PI / 180;
	z = z * CV_PI / 180;

	// Assuming the angles are in radians.
	float cx = cos(x);
	float sx = sin(x);
	float cy = cos(y);
	float sy = sin(y);
	float cz = cos(z);
	float sz = sin(z);

	float m00, m01, m02, m10, m11, m12, m20, m21, m22;

	m00 = cy * cz;
	m01 = -cy * sz;
	m02 = sy;
	m10 = cz * sx*sy + cx * sz;
	m11 = cx * cz - sx * sy*sz;
	m12 = -cy * sx;
	m20 = -cx * cz*sy + sx * sz;
	m21 = cz * sx + cx * sy*sz;
	m22 = cx * cy;

	R(0, 0) = m00; R(0, 1) = m01; R(0, 2) = m02;
	R(1, 0) = m10; R(1, 1) = m11; R(1, 2) = m12;
	R(2, 0) = m20; R(2, 1) = m21; R(2, 2) = m22;

	return R;
}

bool ReadPosesRBOTDataset(
	const std::experimental::filesystem::path& path, std::vector<srt3d::Transform3fA>* poses, int num) {
	std::ifstream ifs{ path.string(), std::ios::binary };
	if (!ifs.is_open() || ifs.fail()) {
		ifs.close();
		std::cerr << "Could not open file stream " << path.string() << std::endl;
		return false;
	}

	poses->resize(num + 1);
	std::string parsed;
	std::getline(ifs, parsed);
	for (auto& pose : *poses) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::getline(ifs, parsed, ' ');
				//std::cout << parsed << std::endl;
				pose.matrix()(i, j) = stof(parsed);
			}
		}
		//std::cout<< pose.matrix() <<std::endl;
		std::getline(ifs, parsed, ' ');
		pose.matrix()(0, 3) = stof(parsed);
		std::getline(ifs, parsed, ' ');
		pose.matrix()(1, 3) = stof(parsed);
		std::getline(ifs, parsed);
		pose.matrix()(2, 3) = stof(parsed);
		//std::cout << pose.matrix() << std::endl;
	}
	return true;
}
bool ReadPosesRBOTDataset_2(
	const std::experimental::filesystem::path& path, std::vector<srt3d::Transform3fA>* poses, int num) {
	std::ifstream ifs{ path.string(), std::ios::binary };
	if (!ifs.is_open() || ifs.fail()) {
		ifs.close();
		std::cerr << "Could not open file stream " << path.string() << std::endl;
		return false;
	}

	poses->resize(num + 1);
	std::string parsed;
	std::getline(ifs, parsed);
	for (auto& pose : *poses) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::getline(ifs, parsed, '\t');
				//std::cout << parsed << std::endl;
				pose.matrix()(i, j) = stof(parsed);
			}
		}
		//std::cout<< pose.matrix() <<std::endl;
		std::getline(ifs, parsed, '\t');
		pose.matrix()(0, 3) = stof(parsed);
		std::getline(ifs, parsed, '\t');
		pose.matrix()(1, 3) = stof(parsed);
		std::getline(ifs, parsed);
		pose.matrix()(2, 3) = stof(parsed);
		//std::cout << pose.matrix() << std::endl;
	}
	return true;
}
bool Read2dPoints(
	const std::experimental::filesystem::path& path, std::vector<std::vector<Eigen::Vector2f>>& points_2d, int num) {
	std::ifstream ifs{ path.string(), std::ios::binary };
	if (!ifs.is_open() || ifs.fail()) {
		ifs.close();
		std::cerr << "Could not open file stream " << path.string() << std::endl;
		return false;
	}

	points_2d.resize(num + 1);

	std::string parsed;
	std::getline(ifs, parsed);
	for (auto &point : points_2d) {
		point.resize(8);
		for (int i = 0; i < 7; ++i) {
			Eigen::Vector2f pp;
			std::getline(ifs, parsed, ' ');
			pp(0) = stof(parsed);
			std::getline(ifs, parsed, ' ');
			pp(1) = stof(parsed);
			point[i] = pp;
		}
		Eigen::Vector2f pp_t;
		std::getline(ifs, parsed, ' ');
		pp_t(0) = stof(parsed);
		std::getline(ifs, parsed);
		pp_t(1) = stof(parsed);
		point[7] = pp_t;
		//std::cout << pose.matrix() << std::endl;
	}
	return true;
}
bool Read2dEdgeVector(
	const std::experimental::filesystem::path& path, std::vector<std::vector<Eigen::Vector2f>>& points_2d, int num) {
	std::ifstream ifs{ path.string(), std::ios::binary };
	if (!ifs.is_open() || ifs.fail()) {
		ifs.close();
		std::cerr << "Could not open file stream " << path.string() << std::endl;
		return false;
	}

	points_2d.resize(num + 1);

	std::string parsed;
	std::getline(ifs, parsed);
	for (auto &point : points_2d) {
		point.resize(28);
		for (int i = 0; i < 27; ++i) {
			Eigen::Vector2f pp;
			std::getline(ifs, parsed, ' ');
			pp(0) = stof(parsed);
			std::getline(ifs, parsed, ' ');
			pp(1) = stof(parsed);
			point[i] = pp;
		}
		Eigen::Vector2f pp_t;
		std::getline(ifs, parsed, ' ');
		pp_t(0) = stof(parsed);
		std::getline(ifs, parsed);
		pp_t(1) = stof(parsed);
		point[27] = pp_t;
		//std::cout << pp_t << std::endl;
	}
	return true;
}

bool Read3DpointVector(
	const std::experimental::filesystem::path& path, std::vector<Eigen::Vector3f>& points_3d, Eigen::Vector3f t_n, int num) {
	std::ifstream ifs{ path.string(), std::ios::binary };
	if (!ifs.is_open() || ifs.fail()) {
		ifs.close();
		std::cerr << "Could not open file stream " << path.string() << std::endl;
		return false;
	}

	points_3d.resize(num);

	std::string parsed;
	//std::getline(ifs, parsed);
	for (auto &point : points_3d) {
		Eigen::Vector3f pp_t;
		std::getline(ifs, parsed, ' ');
		pp_t(0) = stof(parsed) / 1000 - t_n(0);
		std::getline(ifs, parsed, ' ');
		pp_t(1) = stof(parsed) / 1000 - t_n(1);
		std::getline(ifs, parsed);
		pp_t(2) = stof(parsed) / 1000 - t_n(2);
		point = pp_t;
		//std::cout << pp_t << std::endl;
	}
	return true;
}

int main()
{
	///====针对qel装配平台物体的跟踪======试验采用===
#if 0

	std::vector<srt3d::Transform3fA> pose_list;
	std::experimental::filesystem::path pose_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pose\\pre_pose_with_ransac_new_2.txt" };
	ReadPosesRBOTDataset(pose_txt, &pose_list, 419); //6511

	//===读取从热图预测的2d点===
	/*std::vector<std::vector<Eigen::Vector2f>> points_2d_prediction;
	std::experimental::filesystem::path p2ds_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\pose\\points2d_for_save.txt" };
	Read2dPoints(p2ds_txt, points_2d_prediction, 419);*/
	//===读取关键点之间的预测向量===
	/*std::vector<std::vector<Eigen::Vector2f>> edgeVector_2d_prediction;
	std::experimental::filesystem::path EdgeVector2ds_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\pose\\edge_vector_save.txt" };
	Read2dEdgeVector(EdgeVector2ds_txt, edgeVector_2d_prediction, 419);*/

	//======读取关键点====
	/*std::vector<Eigen::Vector3f> points_3d_body;
	std::experimental::filesystem::path points3d_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\part_03.txt" };*/


	//Eigen::Vector3f T_n = { 1.31595, 0.241717, -0.210009 };
	//Read3DpointVector(points3d_txt, points_3d_body, T_n, 8);
	//1315.67 m  241.659 m  -244.461 m
	//part_01  1.3631, -0.139502, 0.0920479
	//part_02  1.31595, 0.241717, -0.210009
	//part_03  1.284, 0.243994, -0.455978
	//part_04  1.35063, -0.019077, -0.348939
	// part_05  1.31525, 0.255697, -0.622516
    //part_06   1.271, -0.358654, -0.602338
	vector<float> Tn_ = { 1.3631, -0.139502, 0.0920479 };
	
	/*MP6D*/
   /*读取测试txt*/
	std::ifstream infile;
	infile.open("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pose\\test_ids.txt", std::ios::in);
	if (!infile.is_open())
	{
		std::cout << "读取文件失败" << std::endl;
		return 0;
	}
	//第三种读取方法
	std::string buf;
	std::vector<std::string> rgbImg_path;
	std::vector<std::string> depthImg_path;
	std::vector<std::string> edgeImg_path;
	std::vector<std::string> maskImg_path;
	while (std::getline(infile, buf))
	{
		/*rgbImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\photo_cut\\val\\" + buf + ".png");
		edgeImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\pre_edge\\" + buf + ".png");
		maskImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\pre_mask\\" + buf + ".png");*/
		char buffer[10];
		sprintf(buffer, "%04d", std::stoi(buf));
		std::string name = std::string(buffer);

		//std::cout<< name <<std::endl;
		rgbImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\photo_cut\\val\\" + name + ".png");
		edgeImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pre_edge\\" + name + ".png");
		maskImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pre_mask\\" + name + ".png");
	}
	for (int i = 0; i < rgbImg_path.size();i++)
	{
		int index = i;
		std::experimental::filesystem::path model_directory_;
		std::shared_ptr<srt3d::Tracker> tracker_ptr_;
		std::shared_ptr<srt3d::RendererGeometry> renderer_geometry_ptr_;
		std::shared_ptr<srt3d::AzureKinectCamera> camera_ptr_;
		std::shared_ptr<srt3d::NormalViewer> viewer_ptr_;

		//model_directory_ = "F:\\BaiduNetdiskDownload\\MP6D\\models_cad\\models_cad";
		model_directory_ = "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\model";

		tracker_ptr_ = std::make_shared<srt3d::Tracker>("tracker");
		renderer_geometry_ptr_ = std::make_shared<srt3d::RendererGeometry>("renderer geometry");
		camera_ptr_ = std::make_shared<srt3d::AzureKinectCamera>("azure_kinect");

		//srt3d::Intrinsics intrinsics{ 567.53720406, 569.36175922,312.66570357, 257.1729701, 640, 480 };
		srt3d::Intrinsics intrinsics{ 1.83094488e+03 / 3.825, 1.83114709e+03 / 3.825,1.19980612e+03 / 3.825, 1.02603620e+03 / 3.825,
			640, 535 };

		camera_ptr_->intrinsics_ = intrinsics;
		cv::Mat1f distortion_(1, 5);

		distortion_ << 0, 0, 0, 0, 0;

		camera_ptr_->distortion_coeff_ = distortion_;

		cv::Mat input_image = cv::imread(rgbImg_path[i]);

		camera_ptr_->temp_image = input_image;

		cv::Mat input_image_edge = cv::imread(edgeImg_path[i]);
		camera_ptr_->preEdge_image = input_image_edge;
		cv::Mat input_image_mask = cv::imread(maskImg_path[i]);
		camera_ptr_->preMask_image = input_image_mask;

		viewer_ptr_ = std::make_shared<srt3d::NormalViewer>("view", camera_ptr_, renderer_geometry_ptr_);
		//viewer_ptr_->StartSavingImages("E:\\PoseTrackingCode\\ContourPose-main\\data\\train\\obj_01\\photo_cut\\visResult\\opt");
		tracker_ptr_->AddViewer(viewer_ptr_);

		vector<string> obj_names;
		obj_names.push_back("part_01"); //obj_02

		vector<vector<double>> poses_;

		vector<double> pose1_{ -135.542, -64.3672, 964.293, -48.9534, 24.3094, -53.3359 };

		poses_.push_back(pose1_);

		srt3d::Transform3fA body1_world2body_pose;
		body1_world2body_pose = pose_list[i];


		Eigen::Matrix4f result_pose_ = body1_world2body_pose.matrix();
		float x_ = result_pose_(0, 3) * 1000;
		float y_ = result_pose_(1, 3) * 1000;
		float z_ = result_pose_(2, 3) * 1000;

		float beta_result_ = 0;
		float alph_result_ = 0;
		float gamma_result_ = 0;

		beta_result_ = asin(result_pose_(0, 2));
		alph_result_ = atan2(-result_pose_(1, 2) / cos(beta_result_), result_pose_(2, 2) / cos(beta_result_));
		gamma_result_ = atan2(-result_pose_(0, 1) / cos(beta_result_), result_pose_(0, 0) / cos(beta_result_));

		beta_result_ = beta_result_ * 180 / CV_PI;
		alph_result_ = alph_result_ * 180 / CV_PI;
		gamma_result_ = gamma_result_ * 180 / CV_PI;
		//std::cout << "show_pose: " << x_ << " " << y_ << " " << z_ << " " << alph_result_ << " " << beta_result_ << " " << gamma_result_ << std::endl;
		std::cout << body1_world2body_pose.matrix() << std::endl;

		srt3d::Transform3fA tn_pose;
		TransVector2matrix(Tn_[0], Tn_[1], Tn_[2], 0, 0, 0, tn_pose);
        //输入的为geometry到world的转换
		//T_n为geometry到body的转换
        //而我们只需要输入world到body的转换
		//因此，我们先用geometry到world的矩阵求逆，得到world到geometry，然后左乘geometry到body
		//就可以得到world到body的矩阵。
		body1_world2body_pose =  body1_world2body_pose * tn_pose.inverse();
		body1_world2body_pose = body1_world2body_pose.inverse();
		

		//转换为六维

		for (int i = 0; i < obj_names.size(); i++)
		{
			string obj_name = obj_names[i];
			string obj_name_all = obj_name + ".obj";
			std::experimental::filesystem::path obj_name_ = obj_name_all;
			const std::experimental::filesystem::path body_geometry_path{ model_directory_ / obj_name_ };
			srt3d::Transform3fA body_geometry2body_pose{ Eigen::Translation3f(0.0f, 0.0f, 0.0f) };
			srt3d::Transform3fA body_world2body_pose;

			vector<double> pose = poses_[i];
			TransVector2matrix(pose[0] / 1000, pose[1] / 1000, pose[2] / 1000, pose[3], pose[4], pose[5], body_world2body_pose);
			//0.8f 
			auto body_ptr{ std::make_shared<srt3d::Body>(obj_name, body_geometry_path,0.001f, true, false, 0.8f, srt3d::Transform3fA::Identity(), i + 1) };
			
			body_ptr->set_world2body_pose(body1_world2body_pose);
			renderer_geometry_ptr_->AddBody(body_ptr);

			
#if 1
			//0.8f
			auto body_model_ptr{ std::make_shared<srt3d::Model>(obj_name, body_ptr, model_directory_, obj_name + "_model.bin", 0.8f, 4, 100, false, 2000) };
			auto body_region_modality_ptr{ std::make_shared<srt3d::RegionModality>(obj_name + "_region_modality", body_ptr, body_model_ptr, camera_ptr_) };

			//调整为8可用来计算前背景直方图的相似度
			body_region_modality_ptr->set_n_histogram_bins(32);
			//name很重要
			//==进行遮挡处理
			auto occlusion_renderer_ptr{ std::make_shared<srt3d::OcclusionRenderer>(obj_name + "_occlusion_renderer", renderer_geometry_ptr_, camera_ptr_) };
			body_region_modality_ptr->UseOcclusionHandling(occlusion_renderer_ptr);

			//body_region_modality_ptr->set_visualize_lines_correspondence(true);
			//body_region_modality_ptr->set_visualize_points_result(true);
			//body_region_modality_ptr->set_visualize_points_pose_update(true);

			//body_region_modality_ptr->set_visualize_points_occlusion_mask_correspondence(true);
			tracker_ptr_->AddRegionModality(body_region_modality_ptr);
#endif

#if 1
			//若使用基于边缘的方法时
			//1.5f
			auto body_edge_model_ptr{ std::make_shared<srt3d::EdgeModel>(obj_name, body_ptr, model_directory_, obj_name + "_edge_model.bin", 0.8f, 4, 200, false, 2000) };
			auto body_edge_modality_ptr{ std::make_shared<srt3d::EdgeModality>(obj_name +
			"_edge_modality", body_ptr, body_edge_model_ptr, camera_ptr_) };

			auto occlusion_renderer_edge_ptr{ std::make_shared<srt3d::OcclusionRenderer>(obj_name + "_occlusion_renderer_edge", renderer_geometry_ptr_, camera_ptr_) };
			body_edge_modality_ptr->UseOcclusionHandling(occlusion_renderer_edge_ptr);

			//body_edge_modality_ptr->pre_heatmap_2dpoints = points_2d_prediction[index];
			//body_edge_modality_ptr->pre_edge_vector_2d = edgeVector_2d_prediction[index];
			//body_edge_modality_ptr->points3d_body = points_3d_body;

		   // body_edge_modality_ptr->set_visualize_lines_correspondence(true);
			//body_edge_modality_ptr->set_visualize_points_result(true);
			//body_edge_modality_ptr->set_visualize_points_pose_update(true);

			//body_edge_modality_ptr->set_visualize_points_pose_update(true);
			tracker_ptr_->AddEdgeModality(body_edge_modality_ptr);
#endif	

#if 0
			//若使用基于直线的方法时
			//1.5f
			auto body_line_model_ptr{ std::make_shared<srt3d::LineModel>(obj_name, body_ptr, model_directory_, obj_name + "_edge_line_region.bin", 0.8f) };
			auto body_line_modality_ptr{ std::make_shared<srt3d::LineModality>(obj_name +
			"_line_modality", body_ptr, body_line_model_ptr, camera_ptr_) };
			body_line_modality_ptr->set_visualize_lines_correspondence(true);
			body_line_modality_ptr->set_visualize_points_result(true);
			body_line_modality_ptr->set_visualize_points_pose_update(true);

			tracker_ptr_->AddLineModality(body_line_modality_ptr);
#endif	
		}
		tracker_ptr_->SetUpTracker();
		//测试深度生成
		//可以将StartTracker在这里重写，从而保证可以进行通信传输
		tracker_ptr_->StartTracker(true);
		for (auto & region_ptr_ : tracker_ptr_->region_modality_ptrs_)
		{
			srt3d::Transform3fA tracking_pose_result = region_ptr_->body_ptr()->body2world_pose();
			//std::cout << tracking_pose_result.matrix() << std::endl;
			Eigen::Matrix4f result_pose1 = tracking_pose_result.matrix();
			float x_show = result_pose1(0, 3) * 1000;
			float y_show = result_pose1(1, 3) * 1000;
			float z_show = result_pose1(2, 3) * 1000;

			float beta_result_show = 0;
			float alph_result_show = 0;
			float gamma_result_show = 0;

			beta_result_show = asin(result_pose1(0, 2));
			alph_result_show = atan2(-result_pose1(1, 2) / cos(beta_result_show), result_pose1(2, 2) / cos(beta_result_show));
			gamma_result_show = atan2(-result_pose1(0, 1) / cos(beta_result_show), result_pose1(0, 0) / cos(beta_result_show));

			beta_result_show = beta_result_show * 180 / CV_PI;
			alph_result_show = alph_result_show * 180 / CV_PI;
			gamma_result_show = gamma_result_show * 180 / CV_PI;
			//std::cout << "show_pose: " << x_show << " " << y_show << " " << z_show << " " << alph_result_show << " " << beta_result_show << " " << gamma_result_show << std::endl;

           /////再转换回来。
			tracking_pose_result = tracking_pose_result * tn_pose;
			std::cout << tracking_pose_result.matrix() << std::endl;

			/*std::ofstream optimizedposefile_1("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pose\\pre_pose_with_ransac_new_2_opti.txt", std::ios::app);
			optimizedposefile_1 << tracking_pose_result(0, 0) << "\t" << tracking_pose_result(0, 1) << "\t" << tracking_pose_result(0, 2) << "\t" <<
				tracking_pose_result(1, 0) << "\t" << tracking_pose_result(1, 1) << "\t" << tracking_pose_result(1, 2) << "\t" <<
				tracking_pose_result(2, 0) << "\t" << tracking_pose_result(2, 1) << "\t" << tracking_pose_result(2, 2) << "\t" <<
				tracking_pose_result(0, 3) << "\t" << tracking_pose_result(1, 3) << "\t" << tracking_pose_result(2, 3) << "\t" << "\n";*/

		}
		cv::Mat viewImg = viewer_ptr_->viewImg;
		//cv::imwrite("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_02\\photo_cut\\vis_pvnet\\" + std::to_string(i) + ".png", viewImg);

		/*for (auto & region_ptr_ : tracker_ptr_->edge_modality_ptrs_)
		{
			srt3d::Transform3fA tracking_pose_result = region_ptr_->body_ptr()->body2world_pose();
			Eigen::Matrix4f result_pose1 = tracking_pose_result.matrix();
			float x_show = result_pose1(0, 3) * 1000;
			float y_show = result_pose1(1, 3) * 1000;
			float z_show = result_pose1(2, 3) * 1000;

			float beta_result_show = 0;
			float alph_result_show = 0;
			float gamma_result_show = 0;

			beta_result_show = asin(result_pose1(0, 2));
			alph_result_show = atan2(-result_pose1(1, 2) / cos(beta_result_show), result_pose1(2, 2) / cos(beta_result_show));
			gamma_result_show = atan2(-result_pose1(0, 1) / cos(beta_result_show), result_pose1(0, 0) / cos(beta_result_show));

			beta_result_show = beta_result_show * 180 / CV_PI;
			alph_result_show = alph_result_show * 180 / CV_PI;
			gamma_result_show = gamma_result_show * 180 / CV_PI;
			std::cout << "show_pose: " << x_show << " " << y_show << " " << z_show << " " << alph_result_show << " " << beta_result_show << " " << gamma_result_show << std::endl;

		}*/
	}	

#endif

	///====针对Mono6D物体的跟踪======
#if 0

	std::vector<srt3d::Transform3fA> pose_list;
	std::experimental::filesystem::path pose_txt = { "F:\\Data_pose\\Mono6D\\Stopper\\pose\\pre_Pose_Stopper_pvnet.txt" };
	//Bracket 400
	//connector 392
	// HingeBase 391
	//L-Holder 342
	//PoleClamp 410
	//SideClamp  516
	//Stopper   571
	//T-Holder  442
	ReadPosesRBOTDataset(pose_txt, &pose_list, 571);

	//Bracket 0.0255, 0.0255, 0.017
	//Connector  0.00775, -0.0005718, 0.012
	//HingeBase  0, -0.025, 0.02
	// L-Holder  0, 0.0065, 0.016
	//PoleClamp  0.00325, 0.0075, 0.01425
	//SideClamp  0.0045, 0, 0.0175
	//Stopper    0.0175, 0.0115, 0.017
	//T-Holder   0, -0.0024177, 0.01
	vector<float> Tn_ = { 0.0175, 0.0115, 0.017 };

	/*MP6D*/
   /*读取测试txt*/
	std::ifstream infile;
	infile.open("F:\\Data_pose\\Mono6D\\Stopper\\pose\\test_ids.txt", std::ios::in);
	if (!infile.is_open())
	{
		std::cout << "读取文件失败" << std::endl;
		return 0;
	}
	//第三种读取方法
	std::string buf;
	std::vector<std::string> rgbImg_path;
	std::vector<std::string> depthImg_path;
	std::vector<std::string> edgeImg_path;
	std::vector<std::string> maskImg_path;
	while (std::getline(infile, buf))
	{
		/*rgbImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\photo_cut\\val\\" + buf + ".png");
		edgeImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\pre_edge\\" + buf + ".png");
		maskImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\pre_mask\\" + buf + ".png");*/
		char buffer[10];
		sprintf(buffer, "%04d", std::stoi(buf));
		std::string name = std::string(buffer);

		//std::cout<< name <<std::endl;
		rgbImg_path.push_back("E:\\MetalDataset\\Mono-6D\\Re_modify\\Stopper\\photo_cut\\val\\" + name + ".png");
		edgeImg_path.push_back("F:\\Data_pose\\Mono6D\\Stopper\\pre_edge\\" + name + ".png");
		maskImg_path.push_back("F:\\Data_pose\\Mono6D\\Stopper\\pre_mask\\" + name + ".png");
	}
	for (int i = 0; i < rgbImg_path.size(); i++)
	{
		int index = i;
		std::experimental::filesystem::path model_directory_;
		std::shared_ptr<srt3d::Tracker> tracker_ptr_;
		std::shared_ptr<srt3d::RendererGeometry> renderer_geometry_ptr_;
		std::shared_ptr<srt3d::AzureKinectCamera> camera_ptr_;
		std::shared_ptr<srt3d::NormalViewer> viewer_ptr_;

		//model_directory_ = "F:\\BaiduNetdiskDownload\\MP6D\\models_cad\\models_cad";
		model_directory_ = "E:\\MetalDataset\\Mono-6D\\Re_modify\\Stopper\\model";

		tracker_ptr_ = std::make_shared<srt3d::Tracker>("tracker");
		renderer_geometry_ptr_ = std::make_shared<srt3d::RendererGeometry>("renderer geometry");
		camera_ptr_ = std::make_shared<srt3d::AzureKinectCamera>("azure_kinect");

		//srt3d::Intrinsics intrinsics{ 567.53720406, 569.36175922,312.66570357, 257.1729701, 640, 480 };
		srt3d::Intrinsics intrinsics{ 2209.878296, 2210.376676,349.751312, 254.828051,
			640, 480 };

		camera_ptr_->intrinsics_ = intrinsics;
		cv::Mat1f distortion_(1, 5);

		distortion_ << 0.0, 0.0, 0.0,0.0,0.0;

		camera_ptr_->distortion_coeff_ = distortion_;

		cv::Mat input_image = cv::imread(rgbImg_path[i]);

		camera_ptr_->temp_image = input_image;

		cv::Mat input_image_edge = cv::imread(edgeImg_path[i]);
		camera_ptr_->preEdge_image = input_image_edge;
		cv::Mat input_image_mask = cv::imread(maskImg_path[i]);
		camera_ptr_->preMask_image = input_image_mask;

		viewer_ptr_ = std::make_shared<srt3d::NormalViewer>("view", camera_ptr_, renderer_geometry_ptr_);
		//viewer_ptr_->StartSavingImages("E:\\PoseTrackingCode\\ContourPose-main\\data\\train\\obj_01\\photo_cut\\visResult\\opt");
		tracker_ptr_->AddViewer(viewer_ptr_);

		vector<string> obj_names;
		obj_names.push_back("Stopper"); //obj_02

		vector<vector<double>> poses_;

		vector<double> pose1_{ -135.542, -64.3672, 964.293, -48.9534, 24.3094, -53.3359 };

		poses_.push_back(pose1_);

		srt3d::Transform3fA body1_world2body_pose;
		body1_world2body_pose = pose_list[i];


		Eigen::Matrix4f result_pose_ = body1_world2body_pose.matrix();
		float x_ = result_pose_(0, 3) * 1000;
		float y_ = result_pose_(1, 3) * 1000;
		float z_ = result_pose_(2, 3) * 1000;

		float beta_result_ = 0;
		float alph_result_ = 0;
		float gamma_result_ = 0;

		beta_result_ = asin(result_pose_(0, 2));
		alph_result_ = atan2(-result_pose_(1, 2) / cos(beta_result_), result_pose_(2, 2) / cos(beta_result_));
		gamma_result_ = atan2(-result_pose_(0, 1) / cos(beta_result_), result_pose_(0, 0) / cos(beta_result_));

		beta_result_ = beta_result_ * 180 / CV_PI;
		alph_result_ = alph_result_ * 180 / CV_PI;
		gamma_result_ = gamma_result_ * 180 / CV_PI;
		//std::cout << "show_pose: " << x_ << " " << y_ << " " << z_ << " " << alph_result_ << " " << beta_result_ << " " << gamma_result_ << std::endl;
		std::cout << body1_world2body_pose.matrix() << std::endl;

		srt3d::Transform3fA tn_pose;
		TransVector2matrix(Tn_[0], Tn_[1], Tn_[2], 0, 0, 0, tn_pose);
		//输入的为geometry到world的转换
		//T_n为geometry到body的转换
		//而我们只需要输入world到body的转换
		//因此，我们先用geometry到world的矩阵求逆，得到world到geometry，然后左乘geometry到body
		//就可以得到world到body的矩阵。
		body1_world2body_pose = body1_world2body_pose * tn_pose.inverse();
		body1_world2body_pose = body1_world2body_pose.inverse();


		//转换为六维

		for (int i = 0; i < obj_names.size(); i++)
		{
			string obj_name = obj_names[i];
			string obj_name_all = obj_name + ".obj";
			std::experimental::filesystem::path obj_name_ = obj_name_all;
			const std::experimental::filesystem::path body_geometry_path{ model_directory_ / obj_name_ };
			srt3d::Transform3fA body_geometry2body_pose{ Eigen::Translation3f(0.0f, 0.0f, 0.0f) };
			srt3d::Transform3fA body_world2body_pose;

			vector<double> pose = poses_[i];
			TransVector2matrix(pose[0] / 1000, pose[1] / 1000, pose[2] / 1000, pose[3], pose[4], pose[5], body_world2body_pose);
			//0.8f 
			auto body_ptr{ std::make_shared<srt3d::Body>(obj_name, body_geometry_path,0.001f, true, false, 0.5f, srt3d::Transform3fA::Identity(), i + 1) };   //0.8f

			body_ptr->set_world2body_pose(body1_world2body_pose);
			renderer_geometry_ptr_->AddBody(body_ptr);


#if 1
			//0.8f
			auto body_model_ptr{ std::make_shared<srt3d::Model>(obj_name, body_ptr, model_directory_, obj_name + "_model.bin", 0.8f, 4, 100, false, 2000) };
			auto body_region_modality_ptr{ std::make_shared<srt3d::RegionModality>(obj_name + "_region_modality", body_ptr, body_model_ptr, camera_ptr_) };

			//调整为8可用来计算前背景直方图的相似度
			body_region_modality_ptr->set_n_histogram_bins(32);
			//name很重要
			//==进行遮挡处理
			auto occlusion_renderer_ptr{ std::make_shared<srt3d::OcclusionRenderer>(obj_name + "_occlusion_renderer", renderer_geometry_ptr_, camera_ptr_) };
			body_region_modality_ptr->UseOcclusionHandling(occlusion_renderer_ptr);

			//body_region_modality_ptr->set_visualize_lines_correspondence(true);
			//body_region_modality_ptr->set_visualize_points_result(true);
			//body_region_modality_ptr->set_visualize_points_pose_update(true);

			//body_region_modality_ptr->set_visualize_points_occlusion_mask_correspondence(true);
			tracker_ptr_->AddRegionModality(body_region_modality_ptr);
#endif

#if 1
			//若使用基于边缘的方法时
			//1.5f
			auto body_edge_model_ptr{ std::make_shared<srt3d::EdgeModel>(obj_name, body_ptr, model_directory_, obj_name + "_edge_model.bin", 0.8f, 4, 200, false, 2000) };
			auto body_edge_modality_ptr{ std::make_shared<srt3d::EdgeModality>(obj_name +
			"_edge_modality", body_ptr, body_edge_model_ptr, camera_ptr_) };

			auto occlusion_renderer_edge_ptr{ std::make_shared<srt3d::OcclusionRenderer>(obj_name + "_occlusion_renderer_edge", renderer_geometry_ptr_, camera_ptr_) };
			body_edge_modality_ptr->UseOcclusionHandling(occlusion_renderer_edge_ptr);

			//body_edge_modality_ptr->pre_heatmap_2dpoints = points_2d_prediction[index];
			//body_edge_modality_ptr->pre_edge_vector_2d = edgeVector_2d_prediction[index];
			//body_edge_modality_ptr->points3d_body = points_3d_body;

			//body_edge_modality_ptr->set_visualize_lines_correspondence(true);
			//body_edge_modality_ptr->set_visualize_points_result(true);
			//body_edge_modality_ptr->set_visualize_points_pose_update(true);

			//body_edge_modality_ptr->set_visualize_points_pose_update(true);
			tracker_ptr_->AddEdgeModality(body_edge_modality_ptr);
#endif	

#if 0
			//若使用基于直线的方法时
			//1.5f
			auto body_line_model_ptr{ std::make_shared<srt3d::LineModel>(obj_name, body_ptr, model_directory_, obj_name + "_edge_line_region.bin", 0.8f) };
			auto body_line_modality_ptr{ std::make_shared<srt3d::LineModality>(obj_name +
			"_line_modality", body_ptr, body_line_model_ptr, camera_ptr_) };
			body_line_modality_ptr->set_visualize_lines_correspondence(true);
			body_line_modality_ptr->set_visualize_points_result(true);
			body_line_modality_ptr->set_visualize_points_pose_update(true);

			tracker_ptr_->AddLineModality(body_line_modality_ptr);
#endif	
		}
		tracker_ptr_->SetUpTracker();
		//测试深度生成
		//可以将StartTracker在这里重写，从而保证可以进行通信传输
		tracker_ptr_->StartTracker(false);
		for (auto & region_ptr_ : tracker_ptr_->region_modality_ptrs_)
		{
			srt3d::Transform3fA tracking_pose_result = region_ptr_->body_ptr()->body2world_pose();
			//std::cout << tracking_pose_result.matrix() << std::endl;
			Eigen::Matrix4f result_pose1 = tracking_pose_result.matrix();
			float x_show = result_pose1(0, 3) * 1000;
			float y_show = result_pose1(1, 3) * 1000;
			float z_show = result_pose1(2, 3) * 1000;

			float beta_result_show = 0;
			float alph_result_show = 0;
			float gamma_result_show = 0;

			beta_result_show = asin(result_pose1(0, 2));
			alph_result_show = atan2(-result_pose1(1, 2) / cos(beta_result_show), result_pose1(2, 2) / cos(beta_result_show));
			gamma_result_show = atan2(-result_pose1(0, 1) / cos(beta_result_show), result_pose1(0, 0) / cos(beta_result_show));

			beta_result_show = beta_result_show * 180 / CV_PI;
			alph_result_show = alph_result_show * 180 / CV_PI;
			gamma_result_show = gamma_result_show * 180 / CV_PI;
			//std::cout << "show_pose: " << x_show << " " << y_show << " " << z_show << " " << alph_result_show << " " << beta_result_show << " " << gamma_result_show << std::endl;

		   /////再转换回来。
			tracking_pose_result = tracking_pose_result * tn_pose;
			//std::cout << tracking_pose_result.matrix() << std::endl;

			/*std::ofstream optimizedposefile_1("F:\\Data_pose\\Mono6D\\T-Holder\\pose\\pre_pose_with_optimization_adaptive.txt", std::ios::app);
			optimizedposefile_1 << tracking_pose_result(0, 0) << "\t" << tracking_pose_result(0, 1) << "\t" << tracking_pose_result(0, 2) << "\t" <<
				tracking_pose_result(1, 0) << "\t" << tracking_pose_result(1, 1) << "\t" << tracking_pose_result(1, 2) << "\t" <<
				tracking_pose_result(2, 0) << "\t" << tracking_pose_result(2, 1) << "\t" << tracking_pose_result(2, 2) << "\t" <<
				tracking_pose_result(0, 3) << "\t" << tracking_pose_result(1, 3) << "\t" << tracking_pose_result(2, 3) << "\t" << "\n";*/

		}
		cv::Mat viewImg = viewer_ptr_->viewImg;
		//cv::imwrite("F:\\Data_pose\\Mono6D\\Bracket\\vis_gt\\" + std::to_string(i) + ".png", viewImg);

	}

#endif

	///====针对qel装配平台物体的跟踪///=======在原图上估计======
#if 0

	std::vector<srt3d::Transform3fA> pose_list;
	std::experimental::filesystem::path pose_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pose\\pre_Pose_part_01_ours.txt" };
	ReadPosesRBOTDataset(pose_txt, &pose_list, 419); //6511

	//===读取从热图预测的2d点===
	/*std::vector<std::vector<Eigen::Vector2f>> points_2d_prediction;
	std::experimental::filesystem::path p2ds_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\pose\\points2d_for_save.txt" };
	Read2dPoints(p2ds_txt, points_2d_prediction, 419);*/
	//===读取关键点之间的预测向量===
	/*std::vector<std::vector<Eigen::Vector2f>> edgeVector_2d_prediction;
	std::experimental::filesystem::path EdgeVector2ds_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\pose\\edge_vector_save.txt" };
	Read2dEdgeVector(EdgeVector2ds_txt, edgeVector_2d_prediction, 419);*/

	//======读取关键点====
	/*std::vector<Eigen::Vector3f> points_3d_body;
	std::experimental::filesystem::path points3d_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\part_03.txt" };*/


	//Eigen::Vector3f T_n = { 1.31595, 0.241717, -0.210009 };
	//Read3DpointVector(points3d_txt, points_3d_body, T_n, 8);
	//1315.67 m  241.659 m  -244.461 m
	//part_01  1.3631, -0.139502, 0.0920479
	//part_02  1.31595, 0.241717, -0.210009
	//part_03  1.284, 0.243994, -0.455978
	//part_04  1.35063, -0.019077, -0.348939
	// part_05  1.31525, 0.255697, -0.622516
	//part_06   1.271, -0.358654, -0.602338
	vector<float> Tn_ = { 1.3631, -0.139502, 0.0920479 };

	/*MP6D*/
   /*读取测试txt*/
	std::ifstream infile;
	infile.open("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pose\\test_ids.txt", std::ios::in);
	if (!infile.is_open())
	{
		std::cout << "读取文件失败" << std::endl;
		return 0;
	}
	//第三种读取方法
	std::string buf;
	std::vector<std::string> rgbImg_path;
	std::vector<std::string> depthImg_path;
	std::vector<std::string> edgeImg_path;
	std::vector<std::string> maskImg_path;
	while (std::getline(infile, buf))
	{
		/*rgbImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\photo_cut\\val\\" + buf + ".png");
		edgeImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\pre_edge\\" + buf + ".png");
		maskImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\pre_mask\\" + buf + ".png");*/
		char buffer[10];
		sprintf(buffer, "%04d", std::stoi(buf));
		std::string name = std::string(buffer);

		//std::cout<< name <<std::endl;
		rgbImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\images_oriSize\\" + name + ".png");
		edgeImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pre_edge\\" + name + ".png");
		maskImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pre_mask\\" + name + ".png");
	}
	for (int i = 0; i < rgbImg_path.size(); i++)
	{
		int index = i;
		std::experimental::filesystem::path model_directory_;
		std::shared_ptr<srt3d::Tracker> tracker_ptr_;
		std::shared_ptr<srt3d::RendererGeometry> renderer_geometry_ptr_;
		std::shared_ptr<srt3d::AzureKinectCamera> camera_ptr_;
		std::shared_ptr<srt3d::NormalViewer> viewer_ptr_;

		//model_directory_ = "F:\\BaiduNetdiskDownload\\MP6D\\models_cad\\models_cad";
		model_directory_ = "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\model";

		tracker_ptr_ = std::make_shared<srt3d::Tracker>("tracker");
		renderer_geometry_ptr_ = std::make_shared<srt3d::RendererGeometry>("renderer geometry");
		camera_ptr_ = std::make_shared<srt3d::AzureKinectCamera>("azure_kinect");

		//srt3d::Intrinsics intrinsics{ 567.53720406, 569.36175922,312.66570357, 257.1729701, 640, 480 };
		srt3d::Intrinsics intrinsics{ 1.83094488e+03, 1.83114709e+03,1.19980612e+03, 1.02603620e+03,
			2448, 2048 };

		camera_ptr_->intrinsics_ = intrinsics;
		cv::Mat1f distortion_(1, 5);

		distortion_ << 0, 0, 0, 0, 0;

		camera_ptr_->distortion_coeff_ = distortion_;

		cv::Mat input_image = cv::imread(rgbImg_path[i]);

		camera_ptr_->temp_image = input_image;

		cv::Mat input_image_edge = cv::imread(edgeImg_path[i]);
		cv::resize(input_image_edge, input_image_edge,cv::Size(2448, 2048));

		camera_ptr_->preEdge_image = input_image_edge;
		cv::Mat input_image_mask = cv::imread(maskImg_path[i]);

		cv::resize(input_image_mask, input_image_mask, cv::Size(2448, 2048));
		
		camera_ptr_->preMask_image = input_image_mask;

		viewer_ptr_ = std::make_shared<srt3d::NormalViewer>("view", camera_ptr_, renderer_geometry_ptr_);
		//viewer_ptr_->StartSavingImages("E:\\PoseTrackingCode\\ContourPose-main\\data\\train\\obj_01\\photo_cut\\visResult\\opt");
		tracker_ptr_->AddViewer(viewer_ptr_);

		vector<string> obj_names;
		obj_names.push_back("part_01"); //obj_02

		vector<vector<double>> poses_;

		vector<double> pose1_{ -135.542, -64.3672, 964.293, -48.9534, 24.3094, -53.3359 };

		poses_.push_back(pose1_);

		srt3d::Transform3fA body1_world2body_pose;
		body1_world2body_pose = pose_list[i];


		Eigen::Matrix4f result_pose_ = body1_world2body_pose.matrix();
		float x_ = result_pose_(0, 3) * 1000;
		float y_ = result_pose_(1, 3) * 1000;
		float z_ = result_pose_(2, 3) * 1000;

		float beta_result_ = 0;
		float alph_result_ = 0;
		float gamma_result_ = 0;

		beta_result_ = asin(result_pose_(0, 2));
		alph_result_ = atan2(-result_pose_(1, 2) / cos(beta_result_), result_pose_(2, 2) / cos(beta_result_));
		gamma_result_ = atan2(-result_pose_(0, 1) / cos(beta_result_), result_pose_(0, 0) / cos(beta_result_));

		beta_result_ = beta_result_ * 180 / CV_PI;
		alph_result_ = alph_result_ * 180 / CV_PI;
		gamma_result_ = gamma_result_ * 180 / CV_PI;
		//std::cout << "show_pose: " << x_ << " " << y_ << " " << z_ << " " << alph_result_ << " " << beta_result_ << " " << gamma_result_ << std::endl;
		std::cout << body1_world2body_pose.matrix() << std::endl;

		srt3d::Transform3fA tn_pose;
		TransVector2matrix(Tn_[0], Tn_[1], Tn_[2], 0, 0, 0, tn_pose);
		//输入的为geometry到world的转换
		//T_n为geometry到body的转换
		//而我们只需要输入world到body的转换
		//因此，我们先用geometry到world的矩阵求逆，得到world到geometry，然后左乘geometry到body
		//就可以得到world到body的矩阵。
		body1_world2body_pose = body1_world2body_pose * tn_pose.inverse();
		body1_world2body_pose = body1_world2body_pose.inverse();


		//转换为六维

		for (int i = 0; i < obj_names.size(); i++)
		{
			string obj_name = obj_names[i];
			string obj_name_all = obj_name + ".obj";
			std::experimental::filesystem::path obj_name_ = obj_name_all;
			const std::experimental::filesystem::path body_geometry_path{ model_directory_ / obj_name_ };
			srt3d::Transform3fA body_geometry2body_pose{ Eigen::Translation3f(0.0f, 0.0f, 0.0f) };
			srt3d::Transform3fA body_world2body_pose;

			vector<double> pose = poses_[i];
			TransVector2matrix(pose[0] / 1000, pose[1] / 1000, pose[2] / 1000, pose[3], pose[4], pose[5], body_world2body_pose);
			//0.8f 
			auto body_ptr{ std::make_shared<srt3d::Body>(obj_name, body_geometry_path,0.001f, true, false, 0.8f, srt3d::Transform3fA::Identity(), i + 1) };

			body_ptr->set_world2body_pose(body1_world2body_pose);
			renderer_geometry_ptr_->AddBody(body_ptr);


#if 1
			//0.8f
			auto body_model_ptr{ std::make_shared<srt3d::Model>(obj_name, body_ptr, model_directory_, obj_name + "_model.bin", 0.8f, 4, 100, false, 2000) };
			auto body_region_modality_ptr{ std::make_shared<srt3d::RegionModality>(obj_name + "_region_modality", body_ptr, body_model_ptr, camera_ptr_) };

			//调整为8可用来计算前背景直方图的相似度
			body_region_modality_ptr->set_n_histogram_bins(32);
			//name很重要
			//==进行遮挡处理
			auto occlusion_renderer_ptr{ std::make_shared<srt3d::OcclusionRenderer>(obj_name + "_occlusion_renderer", renderer_geometry_ptr_, camera_ptr_) };
			body_region_modality_ptr->UseOcclusionHandling(occlusion_renderer_ptr);

			//body_region_modality_ptr->set_visualize_lines_correspondence(true);
			//body_region_modality_ptr->set_visualize_points_result(true);
			//body_region_modality_ptr->set_visualize_points_pose_update(true);

			//body_region_modality_ptr->set_visualize_points_occlusion_mask_correspondence(true);
			tracker_ptr_->AddRegionModality(body_region_modality_ptr);
#endif

#if 1
			//若使用基于边缘的方法时
			//1.5f
			auto body_edge_model_ptr{ std::make_shared<srt3d::EdgeModel>(obj_name, body_ptr, model_directory_, obj_name + "_edge_model.bin", 0.8f, 4, 200, false, 2000) };
			auto body_edge_modality_ptr{ std::make_shared<srt3d::EdgeModality>(obj_name +
			"_edge_modality", body_ptr, body_edge_model_ptr, camera_ptr_) };

			auto occlusion_renderer_edge_ptr{ std::make_shared<srt3d::OcclusionRenderer>(obj_name + "_occlusion_renderer_edge", renderer_geometry_ptr_, camera_ptr_) };
			body_edge_modality_ptr->UseOcclusionHandling(occlusion_renderer_edge_ptr);

			//body_edge_modality_ptr->pre_heatmap_2dpoints = points_2d_prediction[index];
			//body_edge_modality_ptr->pre_edge_vector_2d = edgeVector_2d_prediction[index];
			//body_edge_modality_ptr->points3d_body = points_3d_body;

			//body_edge_modality_ptr->set_visualize_lines_correspondence(true);
			//body_edge_modality_ptr->set_visualize_points_result(true);
			//body_edge_modality_ptr->set_visualize_points_pose_update(true);

			//body_edge_modality_ptr->set_visualize_points_pose_update(true);
			tracker_ptr_->AddEdgeModality(body_edge_modality_ptr);
#endif	

#if 0
			//若使用基于直线的方法时
			//1.5f
			auto body_line_model_ptr{ std::make_shared<srt3d::LineModel>(obj_name, body_ptr, model_directory_, obj_name + "_edge_line_region.bin", 0.8f) };
			auto body_line_modality_ptr{ std::make_shared<srt3d::LineModality>(obj_name +
			"_line_modality", body_ptr, body_line_model_ptr, camera_ptr_) };
			body_line_modality_ptr->set_visualize_lines_correspondence(true);
			body_line_modality_ptr->set_visualize_points_result(true);
			body_line_modality_ptr->set_visualize_points_pose_update(true);

			tracker_ptr_->AddLineModality(body_line_modality_ptr);
#endif	
		}
		tracker_ptr_->SetUpTracker();
		//测试深度生成
		//可以将StartTracker在这里重写，从而保证可以进行通信传输
		tracker_ptr_->StartTracker(true);
		for (auto & region_ptr_ : tracker_ptr_->region_modality_ptrs_)
		{
			srt3d::Transform3fA tracking_pose_result = region_ptr_->body_ptr()->body2world_pose();
			//std::cout << tracking_pose_result.matrix() << std::endl;
			Eigen::Matrix4f result_pose1 = tracking_pose_result.matrix();
			float x_show = result_pose1(0, 3) * 1000;
			float y_show = result_pose1(1, 3) * 1000;
			float z_show = result_pose1(2, 3) * 1000;

			float beta_result_show = 0;
			float alph_result_show = 0;
			float gamma_result_show = 0;

			beta_result_show = asin(result_pose1(0, 2));
			alph_result_show = atan2(-result_pose1(1, 2) / cos(beta_result_show), result_pose1(2, 2) / cos(beta_result_show));
			gamma_result_show = atan2(-result_pose1(0, 1) / cos(beta_result_show), result_pose1(0, 0) / cos(beta_result_show));

			beta_result_show = beta_result_show * 180 / CV_PI;
			alph_result_show = alph_result_show * 180 / CV_PI;
			gamma_result_show = gamma_result_show * 180 / CV_PI;
			//std::cout << "show_pose: " << x_show << " " << y_show << " " << z_show << " " << alph_result_show << " " << beta_result_show << " " << gamma_result_show << std::endl;

		   /////再转换回来。
			tracking_pose_result = tracking_pose_result * tn_pose;
			std::cout << tracking_pose_result.matrix() << std::endl;

			std::ofstream optimizedposefile_1("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pose\\pre_pose_with_ori_img.txt", std::ios::app);
			optimizedposefile_1 << tracking_pose_result(0, 0) << "\t" << tracking_pose_result(0, 1) << "\t" << tracking_pose_result(0, 2) << "\t" <<
				tracking_pose_result(1, 0) << "\t" << tracking_pose_result(1, 1) << "\t" << tracking_pose_result(1, 2) << "\t" <<
				tracking_pose_result(2, 0) << "\t" << tracking_pose_result(2, 1) << "\t" << tracking_pose_result(2, 2) << "\t" <<
				tracking_pose_result(0, 3) << "\t" << tracking_pose_result(1, 3) << "\t" << tracking_pose_result(2, 3) << "\t" << "\n";

		}
		cv::Mat viewImg = viewer_ptr_->viewImg;
		//cv::imwrite("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\photo_cut\\vis_op\\" + std::to_string(i) + ".png", viewImg);

		/*for (auto & region_ptr_ : tracker_ptr_->edge_modality_ptrs_)
		{
			srt3d::Transform3fA tracking_pose_result = region_ptr_->body_ptr()->body2world_pose();
			Eigen::Matrix4f result_pose1 = tracking_pose_result.matrix();
			float x_show = result_pose1(0, 3) * 1000;
			float y_show = result_pose1(1, 3) * 1000;
			float z_show = result_pose1(2, 3) * 1000;

			float beta_result_show = 0;
			float alph_result_show = 0;
			float gamma_result_show = 0;

			beta_result_show = asin(result_pose1(0, 2));
			alph_result_show = atan2(-result_pose1(1, 2) / cos(beta_result_show), result_pose1(2, 2) / cos(beta_result_show));
			gamma_result_show = atan2(-result_pose1(0, 1) / cos(beta_result_show), result_pose1(0, 0) / cos(beta_result_show));

			beta_result_show = beta_result_show * 180 / CV_PI;
			alph_result_show = alph_result_show * 180 / CV_PI;
			gamma_result_show = gamma_result_show * 180 / CV_PI;
			std::cout << "show_pose: " << x_show << " " << y_show << " " << z_show << " " << alph_result_show << " " << beta_result_show << " " << gamma_result_show << std::endl;

		}*/
	}

#endif


	///====针对装配平台物体的跟踪，单个图像的输入======
#if 1

	std::vector<srt3d::Transform3fA> pose_list;
	std::experimental::filesystem::path pose_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_06\\pose\\pre_pose_with_ransac_new_2.txt" };
	ReadPosesRBOTDataset(pose_txt, &pose_list, 419);

	//===读取从热图预测的2d点===
	/*std::vector<std::vector<Eigen::Vector2f>> points_2d_prediction;
	std::experimental::filesystem::path p2ds_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\pose\\points2d_for_save.txt" };
	Read2dPoints(p2ds_txt, points_2d_prediction, 419);*/
	//===读取关键点之间的预测向量===
	/*std::vector<std::vector<Eigen::Vector2f>> edgeVector_2d_prediction;
	std::experimental::filesystem::path EdgeVector2ds_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\pose\\edge_vector_save.txt" };
	Read2dEdgeVector(EdgeVector2ds_txt, edgeVector_2d_prediction, 419);*/

	//======读取关键点====
	/*std::vector<Eigen::Vector3f> points_3d_body;
	std::experimental::filesystem::path points3d_txt = { "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_03\\part_03.txt" };*/

	//Eigen::Vector3f T_n = { 1.31595, 0.241717, -0.210009 };
	//Read3DpointVector(points3d_txt, points_3d_body, T_n, 8);
	
	//1315.67 m  241.659 m  -244.461 m
	//part_01  1.3631, -0.139502, 0.0920479
	//part_02  1.31595, 0.241717, -0.210009
	//part_03  1.284, 0.243994, -0.455978
	//part_04  1.35063, -0.019077, -0.348939
	// part_05  1.31525, 0.255697, -0.622516
	//part_06   1.271, -0.358654, -0.602338
	vector<float> Tn_ = { 1.271, -0.358654, -0.602338 };

	/*MP6D*/
   /*读取测试txt*/
	std::ifstream infile;
	infile.open("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_06\\pose\\test_ids.txt", std::ios::in);
	if (!infile.is_open())
	{
		std::cout << "读取文件失败" << std::endl;
		return 0;
	}
	//第三种读取方法
	std::string buf;
	std::vector<std::string> rgbImg_path;
	std::vector<std::string> depthImg_path;
	std::vector<std::string> edgeImg_path;
	std::vector<std::string> maskImg_path;
	std::vector<int> img_id_name;
	int iiii = 0;
	while (std::getline(infile, buf))
	{
		/*rgbImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\photo_cut\\val\\" + buf + ".png");
		edgeImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\pre_edge\\" + buf + ".png");
		maskImg_path.push_back("E:\\PoseTrackingCode\\ContourPose-main\\data\\test\\scene12\\pre_mask\\" + buf + ".png");*/
		
		char buffer[10];
		sprintf(buffer, "%04d", std::stoi(buf));
		std::string name = std::string(buffer);
		if (name == "0519")
		{
			img_id_name.push_back(iiii);
		}
		iiii++;
		//std::cout<< name <<std::endl;
		rgbImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_06\\photo_cut\\val\\" + name + ".png");
		edgeImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_06\\pre_edge\\" + name + ".png");
		maskImg_path.push_back("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_06\\pre_mask\\" + name + ".png");
	}
	for (int i = 0; i < rgbImg_path.size(); i++)
	{
		if ( i != img_id_name[0])
			continue;

		int index = i;
		std::experimental::filesystem::path model_directory_;
		std::shared_ptr<srt3d::Tracker> tracker_ptr_;
		std::shared_ptr<srt3d::RendererGeometry> renderer_geometry_ptr_;
		std::shared_ptr<srt3d::AzureKinectCamera> camera_ptr_;
		std::shared_ptr<srt3d::NormalViewer> viewer_ptr_;

		//model_directory_ = "F:\\BaiduNetdiskDownload\\MP6D\\models_cad\\models_cad";
		model_directory_ = "F:\\Data_pose\\00QEL_Assembly_Datasets\\part_06\\model";

		tracker_ptr_ = std::make_shared<srt3d::Tracker>("tracker");
		renderer_geometry_ptr_ = std::make_shared<srt3d::RendererGeometry>("renderer geometry");
		camera_ptr_ = std::make_shared<srt3d::AzureKinectCamera>("azure_kinect");

		//srt3d::Intrinsics intrinsics{ 567.53720406, 569.36175922,312.66570357, 257.1729701, 640, 480 };
		srt3d::Intrinsics intrinsics{ 1.83094488e+03 / 3.825, 1.83114709e+03 / 3.825,1.19980612e+03 / 3.825, 1.02603620e+03 / 3.825,
			640, 535 };

		camera_ptr_->intrinsics_ = intrinsics;
		cv::Mat1f distortion_(1, 5);

		distortion_ << 0, 0, 0, 0, 0;

		camera_ptr_->distortion_coeff_ = distortion_;

		cv::Mat input_image = cv::imread(rgbImg_path[i]);

		camera_ptr_->temp_image = input_image;

		cv::Mat input_image_edge = cv::imread(edgeImg_path[i]);
		camera_ptr_->preEdge_image = input_image_edge;
		cv::Mat input_image_mask = cv::imread(maskImg_path[i]);
		camera_ptr_->preMask_image = input_image_mask;

		viewer_ptr_ = std::make_shared<srt3d::NormalViewer>("view", camera_ptr_, renderer_geometry_ptr_);
		//viewer_ptr_->StartSavingImages("E:\\PoseTrackingCode\\ContourPose-main\\data\\train\\obj_01\\photo_cut\\visResult\\opt");
		tracker_ptr_->AddViewer(viewer_ptr_);

		vector<string> obj_names;
		obj_names.push_back("part_06"); //obj_02

		vector<vector<double>> poses_;

		vector<double> pose1_{ -135.542, -64.3672, 964.293, -48.9534, 24.3094, -53.3359 };

		poses_.push_back(pose1_);

		srt3d::Transform3fA body1_world2body_pose;
		body1_world2body_pose = pose_list[i];


		Eigen::Matrix4f result_pose_ = body1_world2body_pose.matrix();
		float x_ = result_pose_(0, 3) * 1000;
		float y_ = result_pose_(1, 3) * 1000;
		float z_ = result_pose_(2, 3) * 1000;

		float beta_result_ = 0;
		float alph_result_ = 0;
		float gamma_result_ = 0;

		beta_result_ = asin(result_pose_(0, 2));
		alph_result_ = atan2(-result_pose_(1, 2) / cos(beta_result_), result_pose_(2, 2) / cos(beta_result_));
		gamma_result_ = atan2(-result_pose_(0, 1) / cos(beta_result_), result_pose_(0, 0) / cos(beta_result_));

		beta_result_ = beta_result_ * 180 / CV_PI;
		alph_result_ = alph_result_ * 180 / CV_PI;
		gamma_result_ = gamma_result_ * 180 / CV_PI;
		//std::cout << "show_pose: " << x_ << " " << y_ << " " << z_ << " " << alph_result_ << " " << beta_result_ << " " << gamma_result_ << std::endl;
		std::cout << body1_world2body_pose.matrix() << std::endl;

		srt3d::Transform3fA tn_pose;
		TransVector2matrix(Tn_[0], Tn_[1], Tn_[2], 0, 0, 0, tn_pose);
		//输入的为geometry到world的转换
		//T_n为geometry到body的转换
		//而我们只需要输入world到body的转换
		//因此，我们先用geometry到world的矩阵求逆，得到world到geometry，然后左乘geometry到body
		//就可以得到world到body的矩阵。
		body1_world2body_pose = body1_world2body_pose * tn_pose.inverse();
		body1_world2body_pose = body1_world2body_pose.inverse();


		//转换为六维

		for (int i = 0; i < obj_names.size(); i++)
		{
			string obj_name = obj_names[i];
			string obj_name_all = obj_name + ".obj";
			std::experimental::filesystem::path obj_name_ = obj_name_all;
			const std::experimental::filesystem::path body_geometry_path{ model_directory_ / obj_name_ };
			srt3d::Transform3fA body_geometry2body_pose{ Eigen::Translation3f(0.0f, 0.0f, 0.0f) };
			srt3d::Transform3fA body_world2body_pose;

			vector<double> pose = poses_[i];
			TransVector2matrix(pose[0] / 1000, pose[1] / 1000, pose[2] / 1000, pose[3], pose[4], pose[5], body_world2body_pose);
			//0.8f 
			auto body_ptr{ std::make_shared<srt3d::Body>(obj_name, body_geometry_path,0.001f, true, false, 0.8f, srt3d::Transform3fA::Identity(), i + 1) };

			body_ptr->set_world2body_pose(body1_world2body_pose);
			renderer_geometry_ptr_->AddBody(body_ptr);


#if 1
			//0.8f
			auto body_model_ptr{ std::make_shared<srt3d::Model>(obj_name, body_ptr, model_directory_, obj_name + "_model.bin", 0.8f, 4, 100, false, 2000) };
			auto body_region_modality_ptr{ std::make_shared<srt3d::RegionModality>(obj_name + "_region_modality", body_ptr, body_model_ptr, camera_ptr_) };

			//调整为8可用来计算前背景直方图的相似度
			body_region_modality_ptr->set_n_histogram_bins(32);
			//name很重要
			//==进行遮挡处理
			auto occlusion_renderer_ptr{ std::make_shared<srt3d::OcclusionRenderer>(obj_name + "_occlusion_renderer", renderer_geometry_ptr_, camera_ptr_) };
			body_region_modality_ptr->UseOcclusionHandling(occlusion_renderer_ptr);

			body_region_modality_ptr->set_visualize_lines_correspondence(true);
			//body_region_modality_ptr->set_visualize_points_result(true);
			//body_region_modality_ptr->set_visualize_points_pose_update(true);

			//body_region_modality_ptr->set_visualize_points_occlusion_mask_correspondence(true);
			tracker_ptr_->AddRegionModality(body_region_modality_ptr);
#endif

#if 1
			//若使用基于边缘的方法时
			//1.5f
			auto body_edge_model_ptr{ std::make_shared<srt3d::EdgeModel>(obj_name, body_ptr, model_directory_, obj_name + "_edge_model.bin", 0.8f, 4, 200, false, 2000) };
			auto body_edge_modality_ptr{ std::make_shared<srt3d::EdgeModality>(obj_name +
			"_edge_modality", body_ptr, body_edge_model_ptr, camera_ptr_) };

			auto occlusion_renderer_edge_ptr{ std::make_shared<srt3d::OcclusionRenderer>(obj_name + "_occlusion_renderer_edge", renderer_geometry_ptr_, camera_ptr_) };
			body_edge_modality_ptr->UseOcclusionHandling(occlusion_renderer_edge_ptr);


		    body_edge_modality_ptr->set_visualize_lines_correspondence(true);
			//body_edge_modality_ptr->set_visualize_points_result(true);
			//body_edge_modality_ptr->set_visualize_points_pose_update(true);

			//body_edge_modality_ptr->set_visualize_points_pose_update(true);
			tracker_ptr_->AddEdgeModality(body_edge_modality_ptr);
#endif	

#if 0
			//若使用基于直线的方法时
			//1.5f
			auto body_line_model_ptr{ std::make_shared<srt3d::LineModel>(obj_name, body_ptr, model_directory_, obj_name + "_edge_line_region.bin", 0.8f) };
			auto body_line_modality_ptr{ std::make_shared<srt3d::LineModality>(obj_name +
			"_line_modality", body_ptr, body_line_model_ptr, camera_ptr_) };
			body_line_modality_ptr->set_visualize_lines_correspondence(true);
			body_line_modality_ptr->set_visualize_points_result(true);
			body_line_modality_ptr->set_visualize_points_pose_update(true);

			tracker_ptr_->AddLineModality(body_line_modality_ptr);
#endif	
		}
		tracker_ptr_->SetUpTracker();
		tracker_ptr_->StartTracker(true);
		for (auto & region_ptr_ : tracker_ptr_->region_modality_ptrs_)
		{
			srt3d::Transform3fA tracking_pose_result = region_ptr_->body_ptr()->body2world_pose();
			//std::cout << tracking_pose_result.matrix() << std::endl;
			Eigen::Matrix4f result_pose1 = tracking_pose_result.matrix();
			float x_show = result_pose1(0, 3) * 1000;
			float y_show = result_pose1(1, 3) * 1000;
			float z_show = result_pose1(2, 3) * 1000;

			float beta_result_show = 0;
			float alph_result_show = 0;
			float gamma_result_show = 0;

			beta_result_show = asin(result_pose1(0, 2));
			alph_result_show = atan2(-result_pose1(1, 2) / cos(beta_result_show), result_pose1(2, 2) / cos(beta_result_show));
			gamma_result_show = atan2(-result_pose1(0, 1) / cos(beta_result_show), result_pose1(0, 0) / cos(beta_result_show));

			beta_result_show = beta_result_show * 180 / CV_PI;
			alph_result_show = alph_result_show * 180 / CV_PI;
			gamma_result_show = gamma_result_show * 180 / CV_PI;
			//std::cout << "show_pose: " << x_show << " " << y_show << " " << z_show << " " << alph_result_show << " " << beta_result_show << " " << gamma_result_show << std::endl;
			tracking_pose_result = tracking_pose_result * tn_pose;
			std::cout << tracking_pose_result.matrix() << std::endl;

			/*std::ofstream optimizedposefile_1("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_01\\pose\\pre_pose_with_ransac_new_2_opti.txt", std::ios::app);
			optimizedposefile_1 << tracking_pose_result(0, 0) << "\t" << tracking_pose_result(0, 1) << "\t" << tracking_pose_result(0, 2) << "\t" <<
				tracking_pose_result(1, 0) << "\t" << tracking_pose_result(1, 1) << "\t" << tracking_pose_result(1, 2) << "\t" <<
				tracking_pose_result(2, 0) << "\t" << tracking_pose_result(2, 1) << "\t" << tracking_pose_result(2, 2) << "\t" <<
				tracking_pose_result(0, 3) << "\t" << tracking_pose_result(1, 3) << "\t" << tracking_pose_result(2, 3) << "\t" << "\n";*/

		}
		cv::Mat viewImg = viewer_ptr_->viewImg;
		//cv::imwrite("F:\\Data_pose\\00QEL_Assembly_Datasets\\part_02\\photo_cut\\vis_pvnet\\" + std::to_string(i) + ".png", viewImg);
	}

#endif
    return 0;
}

