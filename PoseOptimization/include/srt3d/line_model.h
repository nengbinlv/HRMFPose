// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef OBJECT_TRACKING_INCLUDE_SRT3D_LineModel_H_
#define OBJECT_TRACKING_INCLUDE_SRT3D_LineModel_H_

#include <omp.h>
#include <srt3d/body.h>
#include <srt3d/common.h>
#include <srt3d/normal_renderer.h>
#include <srt3d/teture_render.h>
#include <srt3d/renderer_geometry.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include <opencv2/ximgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/line_descriptor/descriptor.hpp>

namespace srt3d {

// Class that stores a sparse viewpoint LineModel of a body that consists of views
// with multiple contour points. It includes all functionality to generate,
// save, and load the LineModel
class LineModel {
 private:
  // Some constants
  static constexpr int kVersionID = 2;
  static constexpr int kContourNormalApproxRadius = 3; //3
  static constexpr int kMinContourLength = 30;         //15
  static constexpr int kMinEdgeLength = 50;         //15
  static constexpr int kImageSizeSafetyBoundary = 20;

  // Struct with operator that compares two Vector3f and checks if v1 < v2
  struct CompareSmallerVector3f {
    bool operator()(const Eigen::Vector3f &v1,
                    const Eigen::Vector3f &v2) const {
      return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]) ||
             (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] < v2[2]);
    }
  };

 public:
  using PointData = struct PointData {
    Eigen::Vector3f center_f_body;
    Eigen::Vector3f normal_f_body;
    float foreground_distance = 0.0f;
    float background_distance = 0.0f;
	//Eigen::Vector3f vimg_desc_body;
	//ÓÐÎÊÌâ
	/*Eigen::Vector3f vimg_desc_body1;
	Eigen::Vector3f vimg_desc_body2;
	Eigen::Vector3f vimg_desc_body3;
	Eigen::Vector3f vimg_desc_body4;
	Eigen::Vector3f vimg_desc_body5;
	Eigen::Vector3f vimg_desc_body6;
	Eigen::Vector3f vimg_desc_body7;
	Eigen::Vector3f vimg_desc_body8;
	Eigen::Vector3f vimg_desc_body9;
	Eigen::Vector3f vimg_desc_body10;
	Eigen::Vector3f vimg_desc_body11;
	Eigen::Vector3f vimg_desc_body12;*/
  };

  using TemplateView = struct TemplateView {
    std::vector<PointData> data_points;
    Eigen::Vector3f orientation;  // points from camera to body center
  };

  // Constructors and setup methods
  LineModel(const std::string &name, std::shared_ptr<Body> body_ptr,
        const std::experimental::filesystem::path &directory, const std::string &filename,
        float sphere_radius = 0.8, int n_divides = 4, int n_points = 300,//200
        bool use_random_seed = true, int image_size = 2000);
  bool SetUp();

  // Setters
  void set_name(const std::string &name);
  void set_body_ptr(std::shared_ptr<Body> body_ptr);
  void set_directory(const std::experimental::filesystem::path &directory);
  void set_filename(const std::string &filename);
  void set_sphere_radius(float sphere_radius);
  void set_n_divides(int n_divides);
  void set_n_points(int n_points);
  void set_use_random_seed(bool use_random_seed);
  void set_image_size(int image_size);

  // Main methods
  bool GetClosestTemplateView(const Transform3fA &body2camera_pose,
                              const TemplateView **closest_template_view) const;

  // Getters
  const std::string &name() const;
  std::shared_ptr<Body> body_ptr() const;
  const std::experimental::filesystem::path &directory() const;
  const std::string &filename() const;
  float sphere_radius() const;
  int n_divides() const;
  int n_points() const;
  bool use_random_seed() const;
  int image_size() const;
  bool set_up() const;

 private:
  // Helper methods for LineModel set up
  bool GenerateLineModel();
  bool LoadLineModel();
  bool SaveLineModel() const;

  // Helper methods for point data
  bool GeneratePointData(const NormalRenderer &renderer,
                         const Transform3fA &camera2body_pose,
                         std::vector<PointData> *data_points) const;
  bool GenerateValidContours(const cv::Mat &silhouette_image,
	                         const cv::Mat &normal_image,
	                         const cv::Mat &depth_iamge,
                             std::vector<std::vector<cv::Point2i>> *contours,
                             int *total_contour_length_in_pixel) const;
  static cv::Point2i SampleContourPointCoordinate(
      const std::vector<std::vector<cv::Point2i>> &contours,
      int total_contour_length_in_pixel, std::mt19937 &generator);
  static bool CalculateContourSegment(
      const std::vector<std::vector<cv::Point2i>> &contours,
      cv::Point2i &center, std::vector<cv::Point2i> *contour_segment);
  static Eigen::Vector2f ApproximateNormalVector(
      const std::vector<cv::Point2i> &contour_segment);
  void CalculateLineDistances(
      const cv::Mat &silhouette_image,
      const std::vector<std::vector<cv::Point2i>> &contours,
      const cv::Point2i &center, const Eigen::Vector2f &normal,
      float pixel_to_meter, float *foreground_distance,
      float *background_distance) const;
  static void FindClosestContourPoint(
      const std::vector<std::vector<cv::Point2i>> &contours, float u, float v,
      int *u_contour, int *v_contour);

  // Halper methods for view data
  bool SetUpRenderer(
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      std::shared_ptr<NormalRenderer> *renderer) const;
  void GenerateGeodesicPoses(
      std::vector<Transform3fA> *camera2body_poses) const;
  void GenerateGeodesicPoints(
      std::set<Eigen::Vector3f, CompareSmallerVector3f> *geodesic_points) const;
  static void SubdivideTriangle(
      const Eigen::Vector3f &v1, const Eigen::Vector3f &v2,
      const Eigen::Vector3f &v3, int n_divides,
      std::set<Eigen::Vector3f, CompareSmallerVector3f> *geodesic_points);

  // LineModel data
  std::vector<TemplateView> template_views_;

  // Data
  std::string name_{};
  std::shared_ptr<Body> body_ptr_ = nullptr;
  std::experimental::filesystem::path directory_{};
  std::string filename_{};
  float sphere_radius_{};
  int n_divides_{};
  int n_points_{};
  bool use_random_seed_{};
  int image_size_{};
  bool set_up_ = false;

public:
	
  static  void checkLineExtremes(cv::Vec4f& extremes, cv::Size imageSize);

  bool LineModel::extractEdgeDescriptor(cv::Mat& desc, const cv::Mat& im, cv::Point pt,
	  double edge_dir, unsigned int window_size) const;
  void LineModel::calcImageGradientDirection(cv::Mat& dst, const cv::Mat& src, const std::vector<cv::Point>& pts) const;
  std::vector<double> LineModel::calcImageGradientDirection(const cv::Mat& src, const std::vector<cv::Point>& pts) const;
      
public:
	struct sort_lines_by_response
	{
		inline bool operator()(const cv::line_descriptor::KeyLine& a, const cv::line_descriptor::KeyLine& b) {
			return (a.response > b.response);
		}
	};
};

}  // namespace srt3d

#endif  // OBJECT_TRACKING_INCLUDE_SRT3D_LineModel_H_
