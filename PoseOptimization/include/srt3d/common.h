// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef OBJECT_TRACKING_INCLUDE_SRT3D_COMMON_H_
#define OBJECT_TRACKING_INCLUDE_SRT3D_COMMON_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include"glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <exception>  
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>
namespace srt3d {

// Commonly used types
using Transform3fA = Eigen::Transform<float, 3, Eigen::Affine>;

// Commonly used constants
constexpr float kPi = 3.1415926535897f;

// Commonly used structs
using Intrinsics = struct Intrinsics {
  float fu, fv;
  float ppu, ppv;
  int width, height;
};

// Commonly used mathematic functions
template <typename T>
inline int sgn(T value) {
  if (value < T(0))
    return -1;
  else if (value > T(0))
    return 1;
  else
    return 0;
}

template <typename T>
inline float sgnf(T value) {
  if (value < T(0))
    return -1.0f;
  else if (value > T(0))
    return 1.0f;
  else
    return 0.0f;
}

template <typename T>
bool AddPtrIfNameNotExists(const T &ptr, std::vector<T> *dest_ptrs) {
	if (!ptr) return true;
	if (!std::none_of(begin(*dest_ptrs), end(*dest_ptrs),
		[&ptr](const T &p) { return p->name() == ptr->name(); }))
		return false;
	dest_ptrs->push_back(ptr);
	return true;
}
template <typename T>
bool DeletePtrIfNameExists(const std::string &name, std::vector<T> *ptrs) {
	auto it{ std::remove_if(begin(*ptrs), end(*ptrs),
		[&name](const T &p) { return p->name() == name; }) };
	if (it == end(*ptrs)) return false;
	ptrs->erase(it, end(*ptrs));
	return true;
}

inline int pow_int(int x, int p) {
  if (p == 0) return 1;
  if (p == 1) return x;
  return pow_int(x, p - 1) * x;
}

inline Eigen::Matrix3f Vector2Skewsymmetric(const Eigen::Vector3f &vector) {
  Eigen::Matrix3f skew_symmetric;
  skew_symmetric << 0.0f, -vector(2), vector(1), vector(2), 0.0f, -vector(0),
      -vector(1), vector(0), 0.0f;
  return skew_symmetric;
}

template <typename T>
inline T LastValidValue(const std::vector<T> &values, int idx) {
	if (idx < values.size())
		return values[idx];
	else
		return values.back();
}

// Commonly used functions to read and write value to file
void ReadValueFromFile(std::ifstream &ifs, bool *value);
void ReadValueFromFile(std::ifstream &ifs, int *value);
void ReadValueFromFile(std::ifstream &ifs, float *value);
void ReadValueFromFile(std::ifstream &ifs, std::string *value);
void ReadValueFromFile(std::ifstream &ifs, Transform3fA *value);
void ReadValueFromFile(std::ifstream &ifs, Intrinsics *value);
void ReadValueFromFile(std::ifstream &ifs, std::experimental::filesystem::path *value);

void WriteValueToFile(std::ofstream &ofs, const std::string &name, bool value);
void WriteValueToFile(std::ofstream &ofs, const std::string &name, int value);
void WriteValueToFile(std::ofstream &ofs, const std::string &name, float value);
void WriteValueToFile(std::ofstream &ofs, const std::string &name,
                      const std::string &value);
void WriteValueToFile(std::ofstream &ofs, const std::string &name,
                      const Transform3fA &value);
void WriteValueToFile(std::ofstream &ofs, const std::string &name,
                      const Intrinsics &value);
void WriteValueToFile(std::ofstream &ofs, const std::string &name,
                      const std::experimental::filesystem::path &value);

// Commonly used functions to plot points to image
void DrawPointInImage(const Eigen::Vector3f &point_f_camera,
                      const cv::Vec3b &color, const Intrinsics &intrinsics,
                      cv::Mat *image);


inline cv::Matx33f getRFromGLM(const cv::Matx44f& m)
{
	const float* v = m.val;

	return  cv::Matx33f(v[0], v[4], v[8], v[1], v[5], v[9], v[2], v[6], v[10]);
}
template<typename _DestT, typename _T>
inline _DestT& cvt(const _T &m)
{
	return *(_DestT*)&m;
}
inline cv::Matx33f dir2OutofplaneRotation(const cv::Vec3f& dir)
{
	auto eyePos = dir;
	//glm::vec3(eyex, eyey, eyez), glm::vec3(centerx, centery, centerz), glm::vec3(upx, upy, upz)
	//eyePos[0], eyePos[1], eyePos[2], 0, 0, 0, 0, 1, 0
	auto m = glm::lookAt(glm::vec3(eyePos[0], eyePos[1], eyePos[2]), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	cv::Matx44f m_ = cvt<cv::Matx44f>(m);
	return getRFromGLM(m_);
}
inline cv::Matx33f dir2InofplaneRotation(float gamma)
{
	auto eyePos = cv::Vec3f(0,0,1);
	//glm::vec3(eyex, eyey, eyez), glm::vec3(centerx, centery, centerz), glm::vec3(upx, upy, upz)
	//eyePos[0], eyePos[1], eyePos[2], 0, 0, 0, 0, 1, 0
	float up_y = cos(gamma);
	float up_x = sin(gamma);
	auto m = glm::lookAt(glm::vec3(eyePos[0], eyePos[1], eyePos[2]), glm::vec3(0, 0, 0), glm::vec3(up_x, up_y, 0));
	cv::Matx44f m_ = cvt<cv::Matx44f>(m);
	return getRFromGLM(m_);
}

inline cv::Vec3f theta2Dir(float theta_x, float theta_y)
{
	CV_Assert(fabs(theta_x) < CV_PI / 2 || fabs(theta_y) < CV_PI / 2);

	if (fabs(theta_x) < 1e-6f)
	{
		return cv::Vec3f(0.f, sin(theta_y), cos(theta_y));
	}
	else if (fabs(theta_y) < 1e-6f)
	{
		return cv::Vec3f(sin(theta_x), 0.f, cos(theta_x));
	}

	float a = 1.f / tan(theta_x);
	float b = 1.f / tan(theta_y);
	float z = sqrt(1.f / (a * a + b * b + 1.f));

	return cv::normalize(cv::Vec3f(z / a, z / b, z));
	/*可以证明，原始计算方法有问题*/
	/*float a = cos(theta_y) * sin(theta_x);
	float b = sin(theta_y);
	float z = cos(theta_y) * cos(theta_x);
	return cv::normalize(cv::Vec3f(a, b, z));*/
}

inline cv::Matx33f theta2OutofplaneRotation(float theta_x, float theta_y)
{
	return dir2OutofplaneRotation(theta2Dir(theta_x, theta_y));
}


inline cv::Vec3f viewDirFromR(const cv::Matx33f& R)
{
	return cv::normalize(cv::Vec3f(R(2, 0), R(2, 1), R(2, 2)));
}

inline cv::Vec2f dir2Theta(const cv::Vec3f& dir)
{
	float theta_x = atan2(dir[0], dir[2]);
	float theta_y = atan2(dir[1], dir[2]);
	return cv::Vec2f(theta_x, theta_y);
}

inline double dist(cv::Point& p1, cv::Point& p2)
{
	return sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
}
inline std::vector <cv::Point> getSampledPoints(std::vector<cv::Point>& v, int sr)
{
	std::vector<cv::Point> result;
	//采样方式为均匀采样，间隔
	for (int i = 0; i < v.size(); i += sr)
		result.push_back(v[i]);
	return result;
}
inline std::pair<cv::Point, cv::Point> getMinMax(std::vector<cv::Point>& cpts1, std::vector<cv::Point>& cpts2)
{
	int mx = 9999999, my = 99999999;
	int Mx = -999999999, My = -999999999;

	int s = std::max(cpts1.size(), cpts2.size());
	for (int i = 0; i < s; i++)
	{
		if (i < cpts1.size())
		{
			if (mx > cpts1[i].x) mx = cpts1[i].x;
			if (my > cpts1[i].y) my = cpts1[i].y;

			if (Mx < cpts1[i].x) Mx = cpts1[i].x;
			if (My < cpts1[i].y) My = cpts1[i].y;
		}

		if (i < cpts2.size())
		{
			if (mx > cpts2[i].x) mx = cpts2[i].x;
			if (my > cpts2[i].y) my = cpts2[i].y;

			if (Mx < cpts2[i].x) Mx = cpts2[i].x;
			if (My < cpts2[i].y) My = cpts2[i].y;
		}
	}
	return std::make_pair(cv::Point(mx, my), cv::Point(Mx, My));
}
inline double angle(cv::Point& p1, cv::Point& p2)
{
	int ydif = p2.y - p1.y;
	int xdif = p2.x - p1.x;

	if (xdif == 0)
	{
		if (ydif < 0) return (-1.0 * CV_PI / 2.0);
		else return (CV_PI / 2.0);
	}

	float slope = ydif / xdif;
	double theta = atan(slope);

	if (theta > 0 && xdif > 0) theta = theta;
	if (theta > 0 && xdif < 0) theta = theta + CV_PI;
	else if (theta < 0 && xdif > 0) theta = theta + CV_PI * 3.0 / 4.0;
	else theta = theta + CV_PI / 2.0;

	return theta;
}

inline double angle_2(cv::Point& p1, cv::Point& p2)
{
	int ydif = p2.y - p1.y;
	int xdif = p2.x - p1.x;
	double theta = atan2(ydif, xdif);

	return theta;
}

inline std::vector<std::vector<int>> getHistogramFromContourPts(std::vector<cv::Point>& contourPts)
{
	// get the average distance
	double avgDistance = 0;
	int numPairs = 0;

	for (int i = 0; i < contourPts.size(); i++)
	{
		cv::Point p1 = contourPts[i];

		for (int j = i + 1; j < contourPts.size(); j++)
		{
			cv::Point p2 = contourPts[j];

			double distance = dist(p1, p2);
			avgDistance += distance;
			numPairs++;
		}
	}
	//计算点的平均距离
	avgDistance = avgDistance / numPairs;

	double maxLogDistance = -9999999.9;
	double minLogDistance = 999999.9;
	for (int i = 0; i < contourPts.size(); i++)
	{
		cv::Point p1 = contourPts[i];

		for (int j = i + 1; j < contourPts.size(); j++)
		{
			cv::Point p2 = contourPts[j];

			double distance = dist(p1, p2);
			distance /= avgDistance;
			//计算log距离
			distance = log(distance);
			//计算最近与最远距离
			if (distance > maxLogDistance) maxLogDistance = distance;
			if (distance < minLogDistance) minLogDistance = std::max(0.0, distance);
		}
	}
	//计算半径
	double radialBound = maxLogDistance + (maxLogDistance - minLogDistance) * 0.01;
	//间隔大小
	double intervalSize = radialBound / 5.0;
	//角度大小
	double angleSize = CV_PI / 6.0;

	//	int **histogram;
	//	histogram = (int **) malloc(sizeof(int *) * contourPts.size());
	//	for (int i=0; i<contourPts.size(); i++)
	//	{
	//		histogram[i] = (int *) malloc(sizeof(int) * 60);
	//		for (int j=0; j<60; j++)
	//		{
	//			histogram[i][j] = 0;
	//		}
	//	}

	//每个边缘点用60个数量级来刻画
	std::vector< std::vector<int> > histogram;
	for (int i = 0; i < contourPts.size(); ++i)
	{
		std::vector<int> tmp;
		//应该是60个扇区
		for (int j = 0; j < 60; j++)
		{
			tmp.push_back(0);
		}
		histogram.push_back(tmp);
	}

	for (int i = 0; i < contourPts.size(); i++)
	{
		cv::Point p1 = contourPts[i];

		for (int j = 0; j < contourPts.size(); j++)
		{
			//非当前遍历点
			if (i == j) continue;

			cv::Point p2 = contourPts[j];
			//两点的角度
			double ang = angle(p1, p2);
			int angleBin = (int)floor(ang / angleSize);
			//两点距离
			double distance = dist(p1, p2);
			distance /= avgDistance;

			distance = std::max(0.0, log(distance));
			int distanceBin = (int)floor(distance / intervalSize);

			//if (distanceBin * 12 + angleBin >= 60) cout << distanceBin * 12 + angleBin << endl;
			if (distanceBin * 12 + angleBin < 0)
				break;
			//cout << distanceBin * 12 + angleBin << endl;
			//每个边缘点，所建立的基于扇区的统计直方图
			//每个点建立了60个扇区
			histogram[i][distanceBin * 12 + angleBin]++;
		}
	}

	return histogram;
}
inline void getChiStatistic(std::vector<std::vector<double> >& stats,
	std::vector< std::vector<int> >& histogram1, int size1,
	std::vector< std::vector<int> >& histogram2, int size2)
{
	int size = std::max(size1, size2);

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (i >= size1 || j >= size2)
			{
				stats[i][j] = 1.2;
				continue;
			}

			double summation = 0;
			for (int k = 0; k < 60; k++)
			{
				double diff = histogram1[i][k] - histogram2[j][k];
				double sum = histogram1[i][k] + histogram2[j][k];
				if (sum == 0)
				{
					summation = summation;
				}
				else
				{
					summation = summation + diff * diff / sum;
				}
				//summation = summation + diff*diff / sum;
				//summation = (double)cv::norm(diff);
			}
			summation = summation / 2;
			stats[i][j] = summation;
		}
	}
	return;
}
inline float getRDiff(const cv::Matx33f& R1, const cv::Matx33f& R2)
{
	cv::Matx33f tmp = R1.t() * R2;
	float cmin = __min(__min(tmp(0, 0), tmp(1, 1)), tmp(2, 2));
	return acos(cmin);
}

}  // namespace srt3d

#endif  // OBJECT_TRACKING_INCLUDE_SRT3D_COMMON_H_
