// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#include <srt3d/line_model.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <exception>  
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <assert.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <set>

//cv::Mat image_edge;

namespace srt3d {

LineModel::LineModel(const std::string &name, std::shared_ptr<Body> body_ptr,
             const std::experimental::filesystem::path &directory,
             const std::string &filename, float sphere_radius, int n_divides,
             int n_points, bool use_random_seed, int image_size)
    : name_{name},
      body_ptr_{std::move(body_ptr)},
      directory_{directory},
      filename_{filename},
      sphere_radius_{sphere_radius},
      n_divides_{n_divides},
      n_points_{n_points},
      use_random_seed_{use_random_seed},
      image_size_{image_size} {}

bool LineModel::SetUp() {
  set_up_ = false;
  if (!LoadLineModel()) {
    if (!GenerateLineModel()) return false;
    if (!SaveLineModel()) return false;
  }
  set_up_ = true;
  return true;
}

void LineModel::set_name(const std::string &name) { name_ = name; }

void LineModel::set_body_ptr(std::shared_ptr<Body> body_ptr) {
  body_ptr_ = std::move(body_ptr);
  set_up_ = false;
}

void LineModel::set_directory(const std::experimental::filesystem::path &directory) {
  directory_ = directory;
  set_up_ = false;
}

void LineModel::set_filename(const std::string &filename) {
  filename_ = filename;
  set_up_ = false;
}

void LineModel::set_sphere_radius(float sphere_radius) {
  sphere_radius_ = sphere_radius;
  set_up_ = false;
}

void LineModel::set_n_divides(int n_divides) {
  n_divides_ = n_divides;
  set_up_ = false;
}

void LineModel::set_n_points(int n_points) {
  n_points_ = n_points;
  set_up_ = false;
}

void LineModel::set_use_random_seed(bool use_random_seed) {
  use_random_seed_ = use_random_seed;
  set_up_ = false;
}

void LineModel::set_image_size(int image_size) {
  image_size_ = image_size;
  set_up_ = false;
}

//通过计算相机与模型的视角向量，计算当前位姿下最近的视角
//得到的视角结构体中包含了轮廓点数据，以及相机到模型的向量
bool LineModel::GetClosestTemplateView(
    const Transform3fA &body2camera_pose,
    const TemplateView **closest_template_view) const {
  if (!set_up_) {
    std::cerr << "Set up LineModel " << name_ << " first" << std::endl;
    return false;
  }
  //这里用来计算与之前建立的模板的最近的视角，首先计算两个坐标系的平移，然后进行归一化
  //再进行旋转操作，即可得到两个坐标系的方向向量，即相机与模型的坐标系的方向向量
  //应用的非常巧妙
  Eigen::Vector3f orientation{
      body2camera_pose.rotation().inverse() *
      body2camera_pose.translation().matrix().normalized()};

  float closest_dot = -1.0f;
  for (auto &template_view : template_views_) {
	  //通过dot点乘的方式计算
	  try {
		  float dot = orientation.dot(template_view.orientation);
		  if (dot > closest_dot) {
			  *closest_template_view = &template_view;
			  closest_dot = dot;
		  }
	  }
	  catch (exception& e)
	  {
		  cout << e.what() << endl;
	  }
  }
  return true;
}

const std::string &LineModel::name() const { return name_; }

std::shared_ptr<Body> LineModel::body_ptr() const { return body_ptr_; }

const std::experimental::filesystem::path &LineModel::directory() const { return directory_; }

const std::string &LineModel::filename() const { return filename_; }

float LineModel::sphere_radius() const { return sphere_radius_; }

int LineModel::n_divides() const { return n_divides_; }

int LineModel::n_points() const { return n_points_; }

bool LineModel::use_random_seed() const { return use_random_seed_; }

int LineModel::image_size() const { return image_size_; }

bool LineModel::set_up() const { return set_up_; }

//生成bin文件
bool LineModel::GenerateLineModel() {
  // Generate camera poses
  std::vector<Transform3fA> camera2body_poses;
  //创建多个位姿
  GenerateGeodesicPoses(&camera2body_poses);

  // Create RendererGeometries in main thread to comply with GLFW thread safety
  // requirements
  std::vector<std::shared_ptr<RendererGeometry>> renderer_geometry_ptrs(
      omp_get_max_threads());
  for (auto &renderer_geometry_ptr : renderer_geometry_ptrs) {
    renderer_geometry_ptr = std::make_shared<RendererGeometry>("rg");
    renderer_geometry_ptr->SetUp();
  }

  // Generate template views
  std::cout << "Start generating LineModel " << name_ << std::endl;
  template_views_.resize(camera2body_poses.size());
  bool cancel = false;
  std::atomic<int> count = 1;
#pragma omp parallel
  {
    //std::shared_ptr<NormalRenderer> renderer_ptr;
	std::shared_ptr<NormalRenderer> renderer_ptr;
    if (!SetUpRenderer(renderer_geometry_ptrs[omp_get_thread_num()],
                       &renderer_ptr))
      cancel = true;
#pragma omp for
    for (int i = 0; i < int(template_views_.size()); ++i) {
      if (cancel) continue;
      std::stringstream msg;
      msg << "Generate " << body_ptr_->name() << " template view " << count++
          << " of " << template_views_.size() << std::endl;
      std::cout << msg.str();

      // Render images
      renderer_ptr->set_camera2world_pose(camera2body_poses[i]);
      renderer_ptr->StartRendering();
      renderer_ptr->FetchNormalImage();
      renderer_ptr->FetchDepthImage();

      // Generate data
	  //从相机到模型中心的向量
      template_views_[i].orientation =
          camera2body_poses[i].matrix().col(2).segment(0, 3);
      template_views_[i].data_points.resize(n_points_);
      if (!GeneratePointData(*renderer_ptr, camera2body_poses[i],
                             &template_views_[i].data_points))
        cancel = true;
    }
  }
  if (cancel) return false;
  std::cout << "Finish generating LineModel " << name_ << std::endl;
  return true;
}

bool LineModel::LoadLineModel() {
  std::experimental::filesystem::path data_path{directory_ / filename_};
  std::ifstream data_ifs{data_path, std::ios::in | std::ios::binary};
  if (!data_ifs.is_open() || data_ifs.fail()) {
    data_ifs.close();
    std::cout << "Could not open LineModel file " << data_path << std::endl;
    return false;
  }

  // Check if LineModel file has correct parameters
  int version_id_file;
  float sphere_radius_file;
  int n_divides_file;
  int n_points_file;
  bool use_random_seed_file;
  int image_size_file;
  data_ifs.read((char *)(&version_id_file), sizeof(version_id_file));
  data_ifs.read((char *)(&sphere_radius_file), sizeof(sphere_radius_file));
  data_ifs.read((char *)(&n_divides_file), sizeof(n_divides_file));
  data_ifs.read((char *)(&n_points_file), sizeof(n_points_file));
  data_ifs.read((char *)(&use_random_seed_file), sizeof(use_random_seed_file));
  data_ifs.read((char *)(&image_size_file), sizeof(image_size_file));
  if (version_id_file != kVersionID || sphere_radius_file != sphere_radius_ ||
      n_divides_file != n_divides_ || n_points_file < n_points_ ||
      use_random_seed_file != use_random_seed_ ||
      image_size_file != image_size_) {
    std::cout << "LineModel file " << data_path
              << " was generated using different LineModel parameters" << std::endl;
    return false;
  }

  // Check if LineModel file has correct geometry data
  std::string geometry_path_string;
  std::string::size_type geometry_path_length;
  float geometry_unit_in_meter;
  bool geometry_counterclockwise;
  bool geometry_enable_culling;
  float maximum_body_diameter;
  Transform3fA geometry2body_pose;
  data_ifs.read((char *)(&geometry_path_length), sizeof(geometry_path_length));
  geometry_path_string.resize(geometry_path_length);
  data_ifs.read((char *)(geometry_path_string.data()), geometry_path_length);
  data_ifs.read((char *)(&geometry_unit_in_meter),
                sizeof(geometry_unit_in_meter));
  data_ifs.read((char *)(&geometry_counterclockwise),
                sizeof(geometry_counterclockwise));
  data_ifs.read((char *)(&geometry_enable_culling),
                sizeof(geometry_enable_culling));
  data_ifs.read((char *)(&maximum_body_diameter),
                sizeof(maximum_body_diameter));
  data_ifs.read((char *)(geometry2body_pose.data()),
                sizeof(geometry2body_pose));
  if (geometry_path_string != body_ptr_->geometry_path() ||
      geometry_unit_in_meter != body_ptr_->geometry_unit_in_meter() ||
      geometry_counterclockwise != body_ptr_->geometry_counterclockwise() ||
      geometry_enable_culling != body_ptr_->geometry_enable_culling() ||
      maximum_body_diameter != body_ptr_->maximum_body_diameter() ||
      geometry2body_pose.matrix() != body_ptr_->geometry2body_pose().matrix()) {
    std::cout << "LineModel file " << data_path
              << " was generated using different body parameters" << std::endl;
    return false;
  }

  // Load template view data
  size_t n_template_views;
  data_ifs.read((char *)(&n_template_views), sizeof(n_template_views));
  template_views_.clear();
  template_views_.reserve(n_template_views);
  for (size_t i = 0; i < n_template_views; i++) {
    TemplateView tv;
    tv.data_points.resize(n_points_);
    data_ifs.read((char *)(tv.data_points.data()),
                  n_points_ * sizeof(PointData));
    data_ifs.read((char *)(tv.orientation.data()), sizeof(tv.orientation));
    template_views_.push_back(std::move(tv));
  }
  data_ifs.close();
  return true;
}

bool LineModel::SaveLineModel() const {
  std::experimental::filesystem::path data_path{directory_ / filename_};
  std::ofstream data_ofs{data_path, std::ios::out | std::ios::binary};

  // Save template parameters
  size_t n_template_views = template_views_.size();
  data_ofs.write((const char *)(&kVersionID), sizeof(kVersionID));
  data_ofs.write((const char *)(&sphere_radius_), sizeof(sphere_radius_));
  data_ofs.write((const char *)(&n_divides_), sizeof(n_divides_));
  data_ofs.write((const char *)(&n_points_), sizeof(n_points_));
  data_ofs.write((const char *)(&use_random_seed_), sizeof(use_random_seed_));
  data_ofs.write((const char *)(&image_size_), sizeof(image_size_));

  // Save geometry data
  std::string geometry_path_string = body_ptr_->geometry_path().string();
  std::string::size_type geometry_path_length = geometry_path_string.length();
  float geometry_unit_in_meter = body_ptr_->geometry_unit_in_meter();
  bool geometry_counterclockwise = body_ptr_->geometry_counterclockwise();
  bool geometry_enable_culling = body_ptr_->geometry_enable_culling();
  float maximum_body_diameter = body_ptr_->maximum_body_diameter();
  Transform3fA geometry2body_pose = body_ptr_->geometry2body_pose();
  data_ofs.write((const char *)(&geometry_path_length),
                 sizeof(geometry_path_length));
  data_ofs.write((const char *)(geometry_path_string.data()),
                 geometry_path_length);
  data_ofs.write((const char *)(&geometry_unit_in_meter),
                 sizeof(geometry_unit_in_meter));
  data_ofs.write((const char *)(&geometry_counterclockwise),
                 sizeof(geometry_counterclockwise));
  data_ofs.write((const char *)(&geometry_enable_culling),
                 sizeof(geometry_enable_culling));
  data_ofs.write((const char *)(&maximum_body_diameter),
                 sizeof(maximum_body_diameter));
  data_ofs.write((const char *)(geometry2body_pose.data()),
                 sizeof(geometry2body_pose));

  // Save main data
  data_ofs.write((const char *)(&n_template_views), sizeof(n_template_views));
  for (const auto &tv : template_views_) {
    data_ofs.write((const char *)(tv.data_points.data()),
                   n_points_ * sizeof(PointData));
    data_ofs.write((const char *)(tv.orientation.data()),
                   sizeof(tv.orientation));
  }
  data_ofs.flush();
  data_ofs.close();
  return true;
}

//bool LineModel::GeneratePointData(const teture_render &renderer,
//                              const Transform3fA &camera2body_pose,
//                              std::vector<PointData> *data_points) const {
//  // Compute silhouette
//  std::vector<cv::Mat> normal_image_channels(4);
//  cv::split(renderer.normal_image(), normal_image_channels);
//  cv::Mat silhouette_image{ normal_image_channels[3] };
//
//  //cv::namedWindow("normal",0);
//  //cv::imshow("normal", renderer.normal_image());
//  //cv::waitKey(0);
//  //通过normal图像提取边缘
//  cv::Mat normal_mat = renderer.normal_image();
//  if (normal_mat.channels()>1)
//  {
//	  cv::cvtColor(normal_mat, normal_mat, CV_BGRA2GRAY);
//  }
//  cv::Mat normal_canny;
//  int low_thre = 50;
//  int height_thre = 100;
//  cv::Canny(normal_mat, normal_canny, low_thre, height_thre);
//  //cv::imwrite("normal_canny.jpg", normal_canny);
//  //通过depth提取边缘
//  cv::Mat depth_mat = renderer.depth_image();
//  if (depth_mat.channels() > 1)
//  {
//	  cv::cvtColor(depth_mat, depth_mat, CV_BGRA2GRAY);
//  }
//  //===16U转8U
//  depth_mat = depth_mat / 257;
//  depth_mat.convertTo(depth_mat, CV_8U);
//  cv::Mat depth_canny;
//  cv::Canny(depth_mat, depth_canny, low_thre, height_thre);
//  //cv::imwrite("depth_canny.jpg", depth_canny);
//  cv::Mat normal_depth_canny;
//  normal_depth_canny = normal_canny + depth_canny;
//  //cv::imwrite("normal_depth_canny.jpg", normal_depth_canny);
//  //cv::waitKey(1);
//  //计算联通的边缘，并建立边缘segment
//
//
//  std::vector<std::vector<cv::Point2i>> contours_normal_depth;
//  cv::findContours(silhouette_image, contours_normal_depth, cv::RetrievalModes::RETR_LIST,   //RETR_TREE
//	  cv::ContourApproximationModes::CHAIN_APPROX_NONE);
//
//
//  	int total_contour_length_in_pixel;
//  	std::vector<std::vector<cv::Point2i>> contours;
//	GenerateValidContours(silhouette_image, &contours, &total_contour_length_in_pixel);
//
//
//  // Filter contours that are too short
//  //正则表达式
//  contours_normal_depth.erase(std::remove_if(begin(contours_normal_depth), end(contours_normal_depth),
//	  [](const std::vector<cv::Point2i> &contour) {return contour.size() < kMinEdgeLength;}),end(contours_normal_depth));
//
//  cv::Mat contours_img(normal_depth_canny.size(), CV_8U, cv::Scalar(0));
//  cv::drawContours(contours_img, contours_normal_depth, -1, cv::Scalar(255), 1);
//
//  /*cv::namedWindow("contours_img", 0);
//  cv::imshow("contours_img", contours_img);
//  cv::imwrite("contours_img.jpg", contours_img);*/
//
//  int total_contour_length = 0;
//  for (auto &contour : contours_normal_depth) {
//	  total_contour_length += int(contour.size());
//  }
//  //std::cout << "total_contour_length:"<< total_contour_length << std::endl;
//  // Set up generator
//  std::mt19937 generator{7};
//  if (use_random_seed_)
//    generator.seed(
//        unsigned(std::chrono::system_clock::now().time_since_epoch().count()));
//
//  //cv::Mat draw_lines = cv::Mat::zeros(normal_depth_canny.size(), CV_8UC3);
//  cv::Mat draw_lines = contours_img.clone();
//  // Calculate data for contour points
//  for (auto data_point{begin(*data_points)}; data_point != end(*data_points);) {
//    // Randomly sample point on contour and calculate 3D center
//    cv::Point2i center{SampleContourPointCoordinate(
//		contours, total_contour_length_in_pixel, generator)};
//
//	//计算得到相机坐标系的三维坐标
//    Eigen::Vector3f center_f_camera{renderer.GetPointVector(center)};
//	//模型坐标系下的三维坐标
//    data_point->center_f_body = camera2body_pose * center_f_camera;
//
//    // Calculate contour segment and approximate normal vector
//    std::vector<cv::Point2i> contour_segment;
//	//通过边缘点center建立边缘段，为了计算normal
//    if (!CalculateContourSegment(contours, center, &contour_segment)) continue;
//    Eigen::Vector2f normal{ApproximateNormalVector(contour_segment)};
//
//	/*for (int i = 0;i < contour_segment.size();i++)
//	{
//		cv::circle(draw_lines, contour_segment[i], 1, (255, 255, 255), 1, 8);
//	}*/
//	//相机坐标系下的法向量
//    Eigen::Vector3f normal_f_camera{normal.x(), normal.y(), 0.0f};
//	//图像中的法向量转换为模型坐标系的法向量
//    data_point->normal_f_body = camera2body_pose.rotation() * normal_f_camera;
//	//u v的计算有问题
//	//cout << "normal:" << normal.x() << "  " << normal.y() << endl;
//	
//	//int u = 0, v=0 ;
//	//int searchline_length_ = 15;
//	//int searchline_length_half = (float(searchline_length_) - 1.0f) / 2.0f;
//	//for (int i = 0; i < searchline_length_; ++i) {
//	//	//绘制搜索线每个点
//	//	if (std::fabs(normal.x()) > std::fabs(normal.y())) {
//	//		u = int(center.x + float(sgn(normal.x()))* (float(i) - searchline_length_half));
//	//		if (std::fabs(normal.x()) == 0)
//	//			v = int(center.y + float(sgn(normal.y())) * (float(i) - searchline_length_half));
//	//		else
//	//		    v = int(center.y + (float(u) - center.x) * (normal.y() / normal.x()));
//	//	}
//	//	else {		
//	//		v = int(center.y + float(sgn(normal.y())) * (float(i) - searchline_length_half));
//	//		//如果x方向为0，则u不增加
//	//		if (std::fabs(normal.x()) == 0)
//	//			u = int(center.x);
//	//		else
//	//		    u = int(center.x + (float(v) - center.y) *	(normal.x() / normal.y()));
//	//	}	
//	//	//cout << "x,y:"<<u <<"  "<< v << endl;
//	//	draw_lines.at<uchar>(v, u) = 255;
//	//}
//
//	////cv::waitKey(0);
//    // Calculate foreground and background distance
//	//
//    float pixel_to_meter = center_f_camera(2) / renderer.intrinsics().fu;
//    CalculateLineDistances(silhouette_image, contours, center, normal,
//                           pixel_to_meter, &data_point->foreground_distance,
//                           &data_point->background_distance);
//    data_point++;
// }
//  /*cv::namedWindow("draw_lines", 0);
//  cv::imshow("draw_lines", draw_lines);
//  cv::imwrite("draw_lines.jpg", draw_lines);*/
//  //cv::waitKey(0);
//   return true;
//}

bool LineModel::GeneratePointData(const NormalRenderer &renderer,
	const Transform3fA &camera2body_pose,
	std::vector<PointData> *data_points) const {
	// Compute silhouette
	std::vector<cv::Mat> normal_image_channels(4);
	cv::split(renderer.normal_image(), normal_image_channels);
	cv::Mat &silhouette_image{ normal_image_channels[3] };
	cv::Mat normal_mat = renderer.normal_image();
	cv::Mat &normal_image{ normal_mat };
	cv::Mat depth_mat = renderer.depth_image();
	cv::Mat &depth_image{ depth_mat };
	// Generate contour
	int total_contour_length_in_pixel;
	std::vector<std::vector<cv::Point2i>> contours;
	if (!GenerateValidContours(silhouette_image, normal_image, depth_image, &contours,
		&total_contour_length_in_pixel))
		return false;

	// Set up generator
	std::mt19937 generator{ 7 };
	if (use_random_seed_)
		generator.seed(
			unsigned(std::chrono::system_clock::now().time_since_epoch().count()));

	// Calculate data for contour points
	//std::ofstream outFile;
	//outFile.open("re4.xyz", std::ofstream::out | std::ofstream::trunc);
	//轮廓点采样  n_points_
	//计算总contours大小
	//看一下contours的情况
	//cv::Mat contours_img = cv::Mat::zeros(silhouette_image.size(),CV_8UC3);
	//for (int i = 0;i < contours.size();i++)
	//{
	//	for (int j =0;j<contours[i].size();j++)
	//	{
	//		//cv::circle(contours_img, contours[i][j], 1, cv::Scalar(0, 255, 0), -1);
	//		contours_img.at<cv::Vec3b>(contours[i][j].y, contours[i][j].x) = cv::Vec3b(0, 255, 0);
	//	    cv::namedWindow("contour",0);
	//		cv::imshow("contour", contours_img);
	//		cv::imwrite("test_contour.jpg", contours_img);
	//		cv::waitKey(0);
	//	}	
	//}	
	//cv::imwrite("contours_img.jpg", contours_img);

	vector<cv::Point2i> center_vetor;
	/*int all_num = 0;
	for (auto contour : contours)
	{
		
		all_num = all_num + contour.size();
	}*/
	
	// int step = all_num / n_points_;
	/*for (auto c : contours)
	{
		double area = cv::contourArea(c,true);
		if (area < 0)
			std::reverse(c.begin(),c.end());
		for (int i = step/2;i < int(c.size());i+=step)
		{
			center_vetor.push_back(c[i]);
		}
	}*/
	std::vector<cv::Point2i> vector_points_all;
	cv::Mat img_show = renderer.normal_image();
	for (auto contour : contours)
	{
		/*for (int i = 0;i < contour.size();i = i + step)
		{
			center_vetor.push_back(contour[i]);
		}*/
		for (auto point : contour)
		{
			//运行很慢
			//vector<cv::Point2i> ::iterator t;
			//t = std::find(vector_points_all.begin(), vector_points_all.end(), point);
			//if (t != vector_points_all.end())
			//{
			//	//找到了
			//	//cout<<"重复"<<endl;
			//	continue;
			//}
			vector_points_all.push_back(point);
		}	
	}
	//去重-稍微快一些
	//好像结果不对
	//vector<cv::Point2i>::iterator it, it1;
	//for (it = ++vector_points_all.begin(); it != vector_points_all.end();)
	//{
	//	//若当前位置之前存在重复元素，删除当前元素,erase返回当前元素的下一个元素指针
	//	it1 = std::find(vector_points_all.begin(), it, *it); 
	//	if (it1 != it)
	//		it = vector_points_all.erase(it);
	//	else
	//		it++;
	//}
#if 1
	float distance = static_cast<float>(vector_points_all.size() / (n_points_)+1);
	std::vector<cv::Point2i> Candidates;
	//Candidates.resize(n_points_);
	float distance_sq = distance * distance / 2;
	int i = 0;
	bool first_select = true;
	while (true)
	{
		cv::Point2i c = vector_points_all[i];

		// Add if sufficient distance away from any previously chosen feature
		bool keep = true;
		for (int j = 0; (j < (int)Candidates.size()) && keep; ++j)
		{
			cv::Point2i f = Candidates[j];
			keep = (c.x - f.x) * (c.x - f.x) + (c.y - f.y) * (c.y - f.y) >= distance_sq;
		}
		if (keep)
		{
			Candidates.push_back(c);
		}
		//cout << Candidates.size() << endl;

		i = i + 1;
		//++i
		if (i == (int)vector_points_all.size()) {
			bool num_ok = Candidates.size() >= n_points_;
			//cout<< num_ok <<endl;
			if (first_select) {
				if (num_ok) {
					Candidates.clear(); // we don't want too many first time
					i = 0;
					distance += 1.0f;
					distance_sq = distance * distance / 2;
					continue;
				}
				else {
					first_select = false;
				}
			}
			// Start back at beginning, and relax required distance
			i = 0;
			distance -= 1.0f;

			distance_sq = distance * distance / 2;
			if (num_ok || distance < 3) {
				//cout<<"break"<<endl;
				break;
			}
			else
			{
				Candidates.clear();
			}
		}
	}


	////cout << Candidates.size() << endl;
	//for (int i = 0;i < Candidates.size();i++)
	//{
	//	cv::Mat img_show = renderer.normal_image();
	//	cv::circle(img_show, Candidates[i], 3, cv::Scalar(0, 0, 255), -1);
	//	cv::namedWindow("Candidates_point", 0);
	//	cv::imshow("Candidates_point", img_show);
	//	cv::waitKey(0);
	//}
#endif

	int all_num_ = Candidates.size();
	int step = all_num_ / n_points_;
	//cout<< all_num_ <<endl;

	for (int i =0;i< Candidates.size();i = i + step)
	{
		center_vetor.push_back(Candidates[i]);
	}

	//得到的轮廓点数与 给定不符
	if (center_vetor.size() >= n_points_)
	{
		//需要删除的点数
		int n = center_vetor.size() - n_points_;
		if (n != 0)
		{
			//共需要剔除n个点
			int index_step = center_vetor.size() / n - 1;
			int s = 0;
			while (n != 0)
			{
				center_vetor.erase(center_vetor.begin() + s);
				s = s + index_step;
				n--;
			}
		}
	}
	else
		cout << "点数不够"<< endl;

	//cout << center_vetor .size()<< endl;
	
	for (int index = 0;index < center_vetor.size();index++)
	{
		cv::circle(img_show, center_vetor[index],1,cv::Scalar(0,0,255),-1);
	}
	/*cv::namedWindow("img_show", 0);
	cv::imshow("img_show", img_show);
	cv::waitKey(0);*/

	int kk = 0;

	//cv::Mat edge_dir;
	//std::vector<cv::Point> edge_pts;
	//edge_pts = center_vetor;
	//calcImageGradientDirection(edge_dir, image_edge, edge_pts);
	//计算结果为弧度
	//cv::imshow("edge_dir", edge_dir);
	//cv::imwrite("edge_dir.jpg", edge_dir);
	//cv::waitKey(0);

	for (auto data_point{ begin(*data_points) }; data_point != end(*data_points);) {
		// Randomly sample point on contour and calculate 3D center
		/*cv::Point2i center{ SampleContourPointCoordinate(
			contours, total_contour_length_in_pixel, generator) };*/		
		if (kk >= n_points_)
		{
			break;
		}
		cv::Point2i center = center_vetor[kk];
		//cout << kk << endl;
		kk++;
		
		Eigen::Vector3f center_f_camera{ renderer.GetPointVector(center) };
		if (center_f_camera(2) == 0 )
		{
			//cout<<"xxx"<<endl;
			continue;
		}
		data_point->center_f_body = camera2body_pose * center_f_camera;
		//保存为xyz文件
		/*std::ostringstream oss1;
		oss1 << center_f_camera.x();
		std::string strx(oss1.str());
		std::ostringstream oss2;
		oss2 << center_f_camera.y();
		std::string stry(oss2.str());
		std::ostringstream oss3;
		oss3 << center_f_camera.z();
		std::string strz(oss3.str());		
		outFile << strx + " " + stry + " " + strz + " " << std::endl;*/		
		// Calculate contour segment and approximate normal vector
		std::vector<cv::Point2i> contour_segment;
		if (!CalculateContourSegment(contours, center, &contour_segment)) 
			continue;

		Eigen::Vector2f normal{ ApproximateNormalVector(contour_segment) };
		Eigen::Vector3f normal_f_camera{ normal.x(), normal.y(), 0.0f };
		//图像中的法向量转换为模型坐标系的法向量
		data_point->normal_f_body = camera2body_pose.rotation() * normal_f_camera;
		
		//在center该点处展开一个小矩形
		//目前这个函数存在问题
		/********************************************************************/
		/*Eigen::VectorXf vimg_desc;
		double dir = atan2(normal.y(), normal.x());
		extractEdgeDescriptor(vimg_desc, edge_dir, cv::Point(center), dir, 1);
		Eigen::VectorXf vimg_desc_test{ vimg_desc };
		data_point->vimg_desc_body = vimg_desc_test;*/
#if 0
		cv::HOGDescriptor detector = cv::HOGDescriptor(cv::Size(5, 5), cv::Size(4, 4), cv::Size(1, 1), cv::Size(4, 4), 9);
		vector<float> descriptions;

		double dir = atan2(normal.y(), normal.x());
		cv::Mat roi_img;
		extractEdgeDescriptor(roi_img, image_edge, cv::Point(center), dir, 2);
		/*cv::imshow("roi_img", roi_img);
		cv::waitKey(0);*/
		detector.compute(roi_img, descriptions);

		data_point->vimg_desc_body1 = Eigen::Vector3f(descriptions[0], descriptions[1], descriptions[2]);
		data_point->vimg_desc_body2 = Eigen::Vector3f(descriptions[3], descriptions[4], descriptions[5]);
		data_point->vimg_desc_body3 = Eigen::Vector3f(descriptions[6], descriptions[7], descriptions[8]);
		data_point->vimg_desc_body4 = Eigen::Vector3f(descriptions[9], descriptions[10], descriptions[11]);
		data_point->vimg_desc_body5 = Eigen::Vector3f(descriptions[12], descriptions[13], descriptions[14]);
		data_point->vimg_desc_body6 = Eigen::Vector3f(descriptions[15], descriptions[16], descriptions[17]);
		data_point->vimg_desc_body7 = Eigen::Vector3f(descriptions[18], descriptions[19], descriptions[20]);
		data_point->vimg_desc_body8 = Eigen::Vector3f(descriptions[21], descriptions[22], descriptions[23]);
		data_point->vimg_desc_body9 = Eigen::Vector3f(descriptions[24], descriptions[25], descriptions[26]);
		data_point->vimg_desc_body10 = Eigen::Vector3f(descriptions[27], descriptions[26], descriptions[29]);
		data_point->vimg_desc_body11 = Eigen::Vector3f(descriptions[30], descriptions[31], descriptions[32]);
		data_point->vimg_desc_body12 = Eigen::Vector3f(descriptions[33], descriptions[34], descriptions[35]);
		//Eigen::VectorXf ds = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(descriptions.data(), descriptions.size());
		//cout<< ds <<endl;
		//data_point->vimg_desc_body = ds;
#endif
		//cout<< data_point->vimg_desc_body <<endl;
		// Calculate foreground and background distance
		//cout<< data_point->vimg_desc_body <<endl;
		float pixel_to_meter = center_f_camera(2) / renderer.intrinsics().fu;
		CalculateLineDistances(silhouette_image, contours, center, normal,
			pixel_to_meter, &data_point->foreground_distance,
			&data_point->background_distance);
		data_point++;

		//cv::circle(img_show, center, 2, cv::Scalar(0,255,0),-1);
	}

	/*cv::namedWindow("img_show",0);
	cv::imshow("img_show", img_show);
	cv::waitKey(0);*/
	//outFile.close();
	return true;
}
void LineModel::calcImageGradientDirection(cv::Mat& dst, const cv::Mat& src, const std::vector<cv::Point>& pts) const
{
	dst = cv::Mat(src.rows, src.cols, CV_32F, cv::Scalar(0));
	std::vector<double> egrads = calcImageGradientDirection(src, pts);
	for (int i = 0; i < egrads.size(); i++)
	{
		dst.at<float>(pts[i].y, pts[i].x) = egrads[i];
	}
}

std::vector<double> LineModel::calcImageGradientDirection(const cv::Mat& src, const std::vector<cv::Point>& pts) const
{
	std::vector<double> grad_dirs(pts.size());
	for (int i = 0; i < pts.size(); i++)
	{
		cv::Point pt = pts[i];
		if ((pt.x - 1) < 0 || (pt.x + 1) >= src.cols || (pt.y - 1) < 0 || (pt.y + 1) >= src.rows)
		{
			grad_dirs[i] = 0;
		}
		else
		{
			//利用卷积核计算梯度，可以得到方向
			double grad_x = -3 * src.at<uchar>(pt.y - 1, pt.x - 1) + 3 * src.at<uchar>(pt.y - 1, pt.x + 1)
				- 10 * src.at<uchar>(pt.y, pt.x - 1) + 10 * src.at<uchar>(pt.y, pt.x + 1)
				- 3 * src.at<uchar>(pt.y + 1, pt.x - 1) + 3 * src.at<uchar>(pt.y + 1, pt.x + 1);
			double grad_y = -3 * src.at<uchar>(pt.y - 1, pt.x - 1) - 10 * src.at<uchar>(pt.y - 1, pt.x) - 3 * src.at<uchar>(pt.y - 1, pt.x + 1)
				+ 3 * src.at<uchar>(pt.y + 1, pt.x - 1) + 10 * src.at<uchar>(pt.y + 1, pt.x) + 3 * src.at<uchar>(pt.y + 1, pt.x + 1);
			grad_dirs[i] = atan2(grad_y, grad_x);
		}
	}
	return grad_dirs;
}


bool LineModel::extractEdgeDescriptor(cv::Mat& desc, const cv::Mat& im, cv::Point pt,
	double edge_dir, unsigned int window_size) const
{
	double dx = cos(edge_dir);
	double dy = sin(edge_dir);

	double dx_k = cos(CV_PI / 2 + edge_dir);
	double dy_k = sin(CV_PI / 2 + edge_dir);

	desc.create(2 * window_size + 1, 2 * window_size+1,CV_8UC1);
	desc.setTo(0);
	//desc.resize(2 * window_size + 1);
	
	for (int j = 0; j < window_size + 1; j++)
	{
		double x_center1 = round(pt.x + j * dx_k);
		double y_center1 = round(pt.y + j * dy_k);

		if (x_center1 < 0 || x_center1 >= im.cols || y_center1 < 0 || y_center1 >= im.rows)
			return false;

		for (int i = 0; i < window_size + 1; i++)
		{
			int x1 = round(x_center1 + i * dx);
			int y1 = round(y_center1 + i * dy);
			if (x1 < 0 || x1 >= im.cols || y1 < 0 || y1 >= im.rows)
				return false;
			desc.at<uchar>(window_size +  i,  (window_size - j)) = im.at<uchar>(y1, x1);

			//cv::circle(im,cv::Point(x1,y1), 1, 255);
			if (i > 0)
			{
				int x2 = round(x_center1 - i * dx);
				int y2 = round(y_center1 - i * dy);
				if (x2 < 0 || x2 >= im.cols || y2 < 0 || y2 >= im.rows)
					return false;

				desc.at<uchar>(window_size - i,  (window_size - j)) = im.at<uchar>(y2, x2);
				//cv::circle(im, cv::Point(x2, y2), 1, 255);
			}
			/*cv::imshow("im", im);
			cv::imshow("desc", desc);

			cv::imwrite("desc.jpg", desc);
			cv::waitKey(0);*/

		}

		if (j > 0)
		{
			double x_center2 = round(pt.x - j * dx_k);
			double y_center2 = round(pt.y - j * dy_k);

			if (x_center2 < 0 || x_center2 >= im.cols || y_center2 < 0 || y_center2 >= im.rows)
				return false;

			for (int i = 0; i < window_size + 1; i++)
			{
				int x1 = round(x_center2 + i * dx);
				int y1 = round(y_center2 + i * dy);
				if (x1 < 0 || x1 >= im.cols || y1 < 0 || y1 >= im.rows)
					return false;
				desc.at<uchar>((window_size + i), (window_size + j)) = im.at<uchar>(y1, x1);
				//cv::circle(im, cv::Point(x1, y1), 1, 255);
				if (i > 0)
				{
					int x2 = round(x_center2 - i * dx);
					int y2 = round(y_center2 - i * dy);
					if (x2 < 0 || x2 >= im.cols || y2 < 0 || y2 >= im.rows)
						return false;

					desc.at<uchar>((window_size - i), (window_size + j)) = im.at<uchar>(y2, x2);
					//cv::circle(im, cv::Point(x2, y2), 1, 255);
				}
			}
			/*cv::imshow("im", im);
			cv::imshow("desc", desc);
			cv::imwrite("desc.jpg", desc);
			cv::waitKey(0);*/
		}
	}
	/*cv::imshow("im", im);
	cv::imshow("desc", desc);
	cv::waitKey(0);*/
	return true;
}

bool LineModel::GenerateValidContours(
    const cv::Mat &silhouette_image,
	const cv::Mat &normal_image,
	const cv::Mat &depth_iamge,
    std::vector<std::vector<cv::Point2i>> *contours,
    int *total_contour_length_in_pixel) const {
  // test if outer border is empty
  for (int i = 0; i < image_size_; ++i) {
    if (silhouette_image.at<uchar>(0, i) ||
        silhouette_image.at<uchar>(image_size_ - 1, i) ||
        silhouette_image.at<uchar>(i, 0) ||
        silhouette_image.at<uchar>(i, image_size_ - 1)) {
      std::cout << "BodyData does not fit into image" << std::endl
                << "Check body2camera_pose and maximum_body_diameter"
                << std::endl;
	  cv::namedWindow("Silhouette Image", 0);
      cv::imshow("Silhouette Image", silhouette_image);
      cv::waitKey(0);
      return false;
    }
  }
    //通过normal图像提取边缘
  cv::Mat normal_image_temp;
  if (normal_image.channels() > 1)
  {
	  cv::cvtColor(normal_image, normal_image_temp, CV_BGRA2GRAY);
  }
  else
	  normal_image_temp = normal_image.clone(); //.clone()

  //cv::cvtColor(normal_image, normal_image_temp, CV_BGRA2BGR);
#if 1
    cv::Mat normal_canny;
    int low_thre = 50;
    int height_thre = 100;
    cv::Canny(normal_image_temp, normal_canny, low_thre, height_thre);

    //通过depth提取边缘
	cv::Mat depth_iamge_temp;
    if (depth_iamge.channels() > 1)
    {
  	  cv::cvtColor(depth_iamge, depth_iamge_temp, CV_BGR2GRAY);
    }
	else
	{
		depth_iamge_temp = depth_iamge.clone();   //.clone()
	}	
    //===16U转8U
	depth_iamge_temp = depth_iamge_temp / 255;
	depth_iamge_temp.convertTo(depth_iamge_temp, CV_8U);

#endif
#if 0
	int length_threshold = 10;
	float distance_threshold = 1.41421356f;
	double canny_th1 = 50.0;
	double canny_th2 = 50.0;
	int canny_aperture_size = 3;
	bool do_merge = true;
	cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(length_threshold,
		distance_threshold, canny_th1, canny_th2, canny_aperture_size,
		do_merge);
	vector<cv::Vec4f> lines;
	lines.clear();
	fld->detect(normal_image_temp, lines);
	int class_counter = -1;
	vector<cv::line_descriptor::KeyLine> keylines;
	for (int k = 0; k < (int)lines.size(); k++)
	{
		cv::line_descriptor::KeyLine kl;
		cv::Vec4f extremes = lines[k];
		/* check data validity */
		checkLineExtremes(extremes, normal_image_temp.size());
		/* fill KeyLine's fields */
		kl.startPointX = extremes[0];
		kl.startPointY = extremes[1];
		kl.endPointX = extremes[2];
		kl.endPointY = extremes[3];
		kl.sPointInOctaveX = extremes[0];
		kl.sPointInOctaveY = extremes[1];
		kl.ePointInOctaveX = extremes[2];
		kl.ePointInOctaveY = extremes[3];
		kl.lineLength = (float)sqrt(pow(extremes[0] - extremes[2], 2) + pow(extremes[1] - extremes[3], 2));

		/* compute number of pixels covered by line */
		cv::LineIterator li(normal_image_temp, cv::Point2f(extremes[0], extremes[1]), cv::Point2f(extremes[2], extremes[3]));
		kl.numOfPixels = li.count;

		kl.angle = atan2((kl.endPointY - kl.startPointY), (kl.endPointX - kl.startPointX));
		kl.class_id = ++class_counter;
		kl.octave = 0;
		kl.size = (kl.endPointX - kl.startPointX) * (kl.endPointY - kl.startPointY);
		kl.response = kl.lineLength / max(normal_image_temp.cols, normal_image_temp.rows);
		kl.pt = cv::Point2f((kl.endPointX + kl.startPointX) / 2, (kl.endPointY + kl.startPointY) / 2);
		keylines.push_back(kl);
	}
	int lsdNFeatures = 10;
	//cout << "filter lines" << endl;
	if (keylines.size()>lsdNFeatures)
	{
		sort(keylines.begin(), keylines.end(), sort_lines_by_response());
		keylines.resize(lsdNFeatures);
		for (int i = 0; i<lsdNFeatures; i++)
			keylines[i].class_id = i;
	}

	cv::Mat line_image = cv::Mat::zeros(normal_image_temp.size(), CV_8UC1);
	for (int i = 0;i < lines.size();i++)
	{
		cv::line_descriptor::KeyLine line = keylines[i];
		cv::line(line_image, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(255), 1);
	}
#endif
	//EdgeDrawing 速度要快一些
#if 1
	cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
	ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;   //LSD SOBEL
	ed->params.GradientThresholdValue = 50;
	ed->params.AnchorThresholdValue = 3;
	vector<cv::Vec6d> ellipses;
	vector<cv::Vec4f> lines;
	vector<cv::Vec4f> lines_depth;
	lines.clear();
	lines_depth.clear();
	//you should call this before detectLines() and detectEllipses()
	//cv::imwrite("normal_image_temp.jpg", normal_image_temp);
	ed->detectEdges(normal_image_temp);
	// Detect lines
	ed->detectLines(lines);
	int class_counter = -1;
	vector<cv::line_descriptor::KeyLine> keylines;
	//cout<< lines.size() <<endl;
	for (int k = 0; k < (int)lines.size(); k++)
	{
		cv::line_descriptor::KeyLine kl;
		cv::Vec4f extremes = lines[k];
		/* check data validity */
		checkLineExtremes(extremes, normal_image_temp.size());
		/* fill KeyLine's fields */
		kl.startPointX = extremes[0];
		kl.startPointY = extremes[1];
		kl.endPointX = extremes[2];
		kl.endPointY = extremes[3];
		kl.sPointInOctaveX = extremes[0];
		kl.sPointInOctaveY = extremes[1];
		kl.ePointInOctaveX = extremes[2];
		kl.ePointInOctaveY = extremes[3];
		kl.lineLength = (float)sqrt(pow(extremes[0] - extremes[2], 2) + pow(extremes[1] - extremes[3], 2));

		/* compute number of pixels covered by line */
		cv::LineIterator li(normal_image_temp, cv::Point2f(extremes[0], extremes[1]), cv::Point2f(extremes[2], extremes[3]));
		kl.numOfPixels = li.count;

		kl.angle = atan2((kl.endPointY - kl.startPointY), (kl.endPointX - kl.startPointX));
		kl.class_id = ++class_counter;
		kl.octave = 0;
		kl.size = (kl.endPointX - kl.startPointX) * (kl.endPointY - kl.startPointY);
		kl.response = kl.lineLength / max(normal_image_temp.cols, normal_image_temp.rows);
		kl.pt = cv::Point2f((kl.endPointX + kl.startPointX) / 2, (kl.endPointY + kl.startPointY) / 2);
		keylines.push_back(kl);
	}
	int lsdNFeatures = 20;
	//cout << "filter lines" << endl;
	/*if (keylines.size()>lsdNFeatures)
	{
		sort(keylines.begin(), keylines.end(), sort_lines_by_response());
		keylines.resize(lsdNFeatures);
		for (int i = 0; i<lsdNFeatures; i++)
			keylines[i].class_id = i;
	}*/
	ed->detectEdges(depth_iamge_temp);
	// Detect lines
	ed->detectLines(lines_depth);
	int class_counter_depth = -1;
	vector<cv::line_descriptor::KeyLine> keylines_depth;
	for (int k = 0; k < (int)lines_depth.size(); k++)
	{
		cv::line_descriptor::KeyLine kl;
		cv::Vec4f extremes = lines_depth[k];
		/* check data validity */
		checkLineExtremes(extremes, depth_iamge_temp.size());
		/* fill KeyLine's fields */
		kl.startPointX = extremes[0];
		kl.startPointY = extremes[1];
		kl.endPointX = extremes[2];
		kl.endPointY = extremes[3];
		kl.sPointInOctaveX = extremes[0];
		kl.sPointInOctaveY = extremes[1];
		kl.ePointInOctaveX = extremes[2];
		kl.ePointInOctaveY = extremes[3];
		kl.lineLength = (float)sqrt(pow(extremes[0] - extremes[2], 2) + pow(extremes[1] - extremes[3], 2));

		/* compute number of pixels covered by line */
		cv::LineIterator li(depth_iamge_temp, cv::Point2f(extremes[0], extremes[1]), cv::Point2f(extremes[2], extremes[3]));
		kl.numOfPixels = li.count;

		kl.angle = atan2((kl.endPointY - kl.startPointY), (kl.endPointX - kl.startPointX));
		kl.class_id = ++class_counter;
		kl.octave = 0;
		kl.size = (kl.endPointX - kl.startPointX) * (kl.endPointY - kl.startPointY);
		kl.response = kl.lineLength / max(depth_iamge_temp.cols, depth_iamge_temp.rows);
		kl.pt = cv::Point2f((kl.endPointX + kl.startPointX) / 2, (kl.endPointY + kl.startPointY) / 2);
		keylines_depth.push_back(kl);
	}

	cv::Mat edge_image_ed = cv::Mat::zeros(normal_image_temp.size(), CV_8UC1);
	for (int i = 0;i < keylines.size();i++)
	{
		cv::line_descriptor::KeyLine line = keylines[i];
		cv::line(edge_image_ed, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(255), 1);
	}

	cv::Mat edge_image_ed_depth = cv::Mat::zeros(normal_image_temp.size(), CV_8UC1);
	for (int i = 0; i < keylines_depth.size(); i++)
	{
		cv::line_descriptor::KeyLine line = keylines_depth[i];
		cv::line(edge_image_ed_depth, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(255), 1);
	}

	cv::Mat struct_element;
	struct_element = cv::getStructuringElement(0, cv::Size(3, 3));
	cv::Mat depth_dilate;
	cv::dilate(edge_image_ed_depth, depth_dilate, struct_element);
	cv::dilate(depth_dilate, depth_dilate, struct_element);
	cv::dilate(depth_dilate, depth_dilate, struct_element);

	cv::Mat normal_sub_depthDilate;
	//normal_sub_depthDilate = laplace_Mat_normal - depth_dilate;
	normal_sub_depthDilate = edge_image_ed - depth_dilate;

	cv::Mat laplace_depth_normal;
	laplace_depth_normal = edge_image_ed_depth + normal_sub_depthDilate;
#endif
	/*cv::namedWindow("edge_image_ed", 0);
	cv::Mat edge_image_ed_show = normal_depth_canny.clone();
	cv::resize(edge_image_ed_show, edge_image_ed_show,cv::Size(800,800));*/
	/*cv::namedWindow("edge_image_ed", 0);
	cv::imshow("edge_image_ed", laplace_depth_normal);
	bitwise_not(laplace_depth_normal, laplace_depth_normal);
	cv::imwrite("edge_image_ed_show.jpg", laplace_depth_normal);
	cv::waitKey(0);*/

  // Compute contours
	//silhouette_image normal_depth_canny laplace_depth_normal
	
    cv::findContours(laplace_depth_normal, *contours, cv::RetrievalModes::RETR_LIST,
                   cv::ContourApproximationModes::CHAIN_APPROX_NONE);  //
  // Filter contours that are too short
  contours->erase(std::remove_if(begin(*contours), end(*contours),
                                 [](const std::vector<cv::Point2i> &contour) {
                                   return contour.size() < kMinContourLength;
                                 }),
                  end(*contours));

  /*cv::Mat contours_img(silhouette_image.size(), CV_8U, cv::Scalar(0));
  for (int i = 0; i < contours->size(); i++)
  {
	  cv::drawContours(contours_img, *contours, i, cv::Scalar(255), 1);
	  
	  cv::imwrite("contours_img.jpg", contours_img);
	  
  }
  cv::namedWindow("contours_img", 0);
  cv::imshow("contours_img", contours_img);
  cv::waitKey(0);*/
  //检测边缘或者直线等几何特征
  // Test if contours are closed
  /*for (auto &contour : *contours) {
    if (abs(contour.front().x - contour.back().x) > 1 ||
        abs(contour.front().y - contour.back().y) > 1) {
      std::cerr << "Contours are not closed. " << std::endl;
      return false;
    }
  }*/

  //将外边缘不在物体上的点用在物体上的代替
  //计算得到外轮廓
  /*std::vector<std::vector<cv::Point2i>> contours_out;
  cv::findContours(silhouette_image, contours_out, cv::RetrievalModes::RETR_LIST,
	  cv::ContourApproximationModes::CHAIN_APPROX_NONE);*/
  ////若距离太近进行剔除
  //std::vector<std::vector<cv::Point2i>> *contours_filter_final;
  //for (auto &contour : *contours)
  //{
	 // std::vector<cv::Point2i> contour_points;
	 // for (auto &point : contour)
	 // {
		//  int flag_close_out = 0;
		//  for (auto contour : contours_out)
		//  {
		//	  for (auto contour_out : contour)
		//	  {
		//		  float distance = hypotf(float(contour_out.x) - float(point.x), float(contour_out.y) - float(point.y));
		//		  if (distance < 5)
		//		  {
		//			  flag_close_out = 1;
		//		  }
		//	  }
		//  }
		//  if (flag_close_out == 0)
		//  {
		//	  contour_points.push_back(point);
		//  }
	 // }
	 // if (contour_points.size() > 0)
	 // {
		//  contours_filter_final->push_back(contour_points);
	 // }
  //}
  //*contours = *contours_filter_final;

  //for (auto &contour : *contours) {
	 // for (auto &point : contour) {
		//  if (!silhouette_image.at<uchar>(point))
		//  {
		//	  //计算与外轮廓最近的点
		//	  float min_distance = std::numeric_limits<float>::max();
		//	  for (auto contour_out: contours_out)
		//	  {
		//		  for (auto point_out : contour_out)
		//		  {
		//			  float distance = hypotf(float(point.x) - float(point_out.x), float(point.y) - float(point_out.y));
		//			  if (distance < min_distance) {
		//				  point.x = point_out.x;
		//				  point.y = point_out.y;
		//				  min_distance = distance;
		//			  }
		//		  }
		//	  }		 
		//  }		  
	 // }
  //}

  // Calculate total pixel length of contour
  *total_contour_length_in_pixel = 0;
  for (auto &contour : *contours) {
    *total_contour_length_in_pixel += int(contour.size());
  }

  // Check if pixel length is greater zero
  if (*total_contour_length_in_pixel == 0) {
    std::cerr << "No valid contour in image " << std::endl;
    return false;
  }
  return true;
}

cv::Point2i LineModel::SampleContourPointCoordinate(
    const std::vector<std::vector<cv::Point2i>> &contours,
    int total_contour_length_in_pixel, std::mt19937 &generator) {
  int idx = int(generator() % total_contour_length_in_pixel);
  for (auto &contour : contours) {
    if (idx < contour.size())
      return contour[idx];
    else
      idx -= int(contour.size());
  }
  return cv::Point2i();  // Never reached
}

bool LineModel::CalculateContourSegment(
    const std::vector<std::vector<cv::Point2i>> &contours, cv::Point2i &center,
    std::vector<cv::Point2i> *contour_segment) {
  for (auto &contour : contours) {
    for (int idx = 0; idx < contour.size(); ++idx) {
      if (contour.at(idx) == center) {
        int start_idx = idx - kContourNormalApproxRadius;
        int end_idx = idx + kContourNormalApproxRadius;
        if (start_idx < 0) {
          contour_segment->insert(end(*contour_segment),
                                  end(contour) + start_idx, end(contour));
          start_idx = 0;
        }
        if (end_idx >= int(contour.size())) {
          contour_segment->insert(end(*contour_segment),
                                  begin(contour) + start_idx, end(contour));
          start_idx = 0;
          end_idx = end_idx - int(contour.size());
        }
        contour_segment->insert(end(*contour_segment),
                                begin(contour) + start_idx,
                                begin(contour) + end_idx + 1);

        // Check quality of contour segment
        float segment_distance = std::hypotf(
            float(contour_segment->back().x - contour_segment->front().x),
            float(contour_segment->back().y - contour_segment->front().y));
        return segment_distance > float(kContourNormalApproxRadius);
      }
    }
  }
  std::cerr << "Could not find point on contour" << std::endl;
  return false;
}

Eigen::Vector2f LineModel::ApproximateNormalVector(
    const std::vector<cv::Point2i> &contour_segment) {
  return Eigen::Vector2f{
      -float(contour_segment.back().y - contour_segment.front().y),
      float(contour_segment.back().x - contour_segment.front().x)}
      .normalized();
}

void LineModel::CalculateLineDistances(
    const cv::Mat &silhouette_image,
    const std::vector<std::vector<cv::Point2i>> &contours,
    const cv::Point2i &center, const Eigen::Vector2f &normal,
    float pixel_to_meter, float *foreground_distance,
    float *background_distance) const {
  // Calculate starting positions and steps for both sides of the line
  float u_out = float(center.x) + 0.5f;
  float v_out = float(center.y) + 0.5f;
  float u_in = float(center.x) + 0.5f;
  float v_in = float(center.y) + 0.5f;
  float u_step, v_step;
  if (std::fabs(normal.y()) < std::fabs(normal.x())) {
    u_step = float(sgn(normal.x()));
    v_step = normal.y() / abs(normal.x());
  } else {
    u_step = normal.x() / abs(normal.y());
    v_step = float(sgn(normal.y()));
  }

  // Search for first inwards intersection with contour
  int u_in_endpoint, v_in_endpoint;
  while (true) {
    u_in -= u_step;
    v_in -= v_step;
    if (!silhouette_image.at<uchar>(int(v_in), int(u_in))) {
      FindClosestContourPoint(contours, u_in + u_step - 0.5f,
                              v_in + v_step - 0.5f, &u_in_endpoint,
                              &v_in_endpoint);
      *foreground_distance =
          pixel_to_meter * hypotf(float(u_in_endpoint - center.x),
                                  float(v_in_endpoint - center.y));
      break;
    }
  }

  // Search for first outwards intersection with contour
  int u_out_endpoint, v_out_endpoint;
  while (true) {
    u_out += u_step;
    v_out += v_step;
    if (int(u_out) < 0 || int(u_out) >= image_size_ || int(v_out) < 0 ||
        int(v_out) >= image_size_) {
      *background_distance = std::numeric_limits<float>::max();
      break;
    }
    if (silhouette_image.at<uchar>(int(v_out), int(u_out))) {
      FindClosestContourPoint(contours, u_out - 0.5f, v_out - 0.5f,
                              &u_out_endpoint, &v_out_endpoint);
      *background_distance =
          pixel_to_meter * hypotf(float(u_out_endpoint - center.x),
                                  float(v_out_endpoint - center.y));
      break;
    }
  }
}

void LineModel::FindClosestContourPoint(
    const std::vector<std::vector<cv::Point2i>> &contours, float u, float v,
    int *u_contour, int *v_contour) {
  float min_distance = std::numeric_limits<float>::max();
  for (auto &contour : contours) {
    for (auto &point : contour) {
      float distance = hypotf(float(point.x) - u, float(point.y) - v);
      if (distance < min_distance) {
        *u_contour = point.x;
        *v_contour = point.y;
        min_distance = distance;
      }
    }
  }
}

bool LineModel::SetUpRenderer(
    const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
    std::shared_ptr<NormalRenderer> *renderer_ptr) const {
  // Set up renderer geometry
  auto copied_body_ptr{std::make_shared<Body>(*body_ptr_)};
  copied_body_ptr->set_body2world_pose(Transform3fA::Identity());
  if (!renderer_geometry_ptr->AddBody(copied_body_ptr)) return false;

  // Calculate parameters
  float focal_length = float(image_size_ - kImageSizeSafetyBoundary) *
                       sphere_radius_ / body_ptr_->maximum_body_diameter();
  float principal_point = float(image_size_) / 2.0f;
  Intrinsics intrinsics{focal_length,    focal_length, principal_point,
                        principal_point, image_size_,  image_size_};
  float z_min = sphere_radius_ - body_ptr_->maximum_body_diameter() * 0.5f;
  float z_max = sphere_radius_ + body_ptr_->maximum_body_diameter() * 0.5f;

  // Set up renderer
  *renderer_ptr = std::make_shared<NormalRenderer>(
      "renderer", renderer_geometry_ptr, Transform3fA::Identity(), intrinsics,
      z_min, z_max);
  return (*renderer_ptr)->SetUp();
}

void LineModel::GenerateGeodesicPoses(
    std::vector<Transform3fA> *camera2body_poses) const {
  // Generate geodesic points
  std::set<Eigen::Vector3f, CompareSmallerVector3f> geodesic_points;
  //创建球点
  GenerateGeodesicPoints(&geodesic_points);

  // Generate geodesic poses from points
  Eigen::Vector3f downwards{0.0f, 1.0f, 0.0f};  // direction in body frame
  camera2body_poses->clear();
  for (const auto &geodesic_point : geodesic_points) {
    Transform3fA pose;
    pose = Eigen::Translation<float, 3>{geodesic_point * sphere_radius_};

	//geodesic_point每个点表示了相机与模型的关系，类似于调整相机的位置得到view矩阵一样；
	//但这里调整的是模型，相机在世界坐标系不动，模型进行变化
    Eigen::Matrix3f Rotation;
	//z轴
    Rotation.col(2) = -geodesic_point;
	//下向量叉乘z轴，x轴
    Rotation.col(0) = downwards.cross(-geodesic_point).normalized();
    if (Rotation.col(0).sum() == 0) {
      Rotation.col(0) = Eigen::Vector3f{1.0f, 0.0f, 0.0f};
    }
	//y轴：z轴叉乘x轴
    Rotation.col(1) = Rotation.col(2).cross(Rotation.col(0));
    //对只有平移的位姿矩阵进行旋转

	pose.rotate(Rotation);
    camera2body_poses->push_back(pose);
  }
}

void LineModel::GenerateGeodesicPoints(
    std::set<Eigen::Vector3f, CompareSmallerVector3f> *geodesic_points) const {
  // Define icosahedron
	//定义二十面体
  constexpr float x = 0.525731112119133606f;
  constexpr float z = 0.850650808352039932f;
  //先创建包围球体的12个点
  std::vector<Eigen::Vector3f> icosahedron_points{
      {-x, 0.0f, z}, {x, 0.0f, z},  {-x, 0.0f, -z}, {x, 0.0f, -z},
      {0.0f, z, x},  {0.0f, z, -x}, {0.0f, -z, x},  {0.0f, -z, -x},
      {z, x, 0.0f},  {-z, x, 0.0f}, {z, -x, 0.0f},  {-z, -x, 0.0f}};
  //依次连接三个点，形成三角面片
  std::vector<std::array<int, 3>> icosahedron_ids{
      {0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1},
      {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3},  {2, 7, 3},
      {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
      {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5},  {7, 2, 11}};

  // Create points
  //通过求三角面片的中点，再对三角面片进行细分
  geodesic_points->clear();
  for (const auto &icosahedron_id : icosahedron_ids) {
    SubdivideTriangle(icosahedron_points[icosahedron_id[0]],
                      icosahedron_points[icosahedron_id[1]],
                      icosahedron_points[icosahedron_id[2]], n_divides_,
                      geodesic_points);
  }
}

void LineModel::SubdivideTriangle(
    const Eigen::Vector3f &v1, const Eigen::Vector3f &v2,
    const Eigen::Vector3f &v3, int n_divides,
    std::set<Eigen::Vector3f, CompareSmallerVector3f> *geodesic_points) {
  if (n_divides == 0) {
    geodesic_points->insert(v1);
    geodesic_points->insert(v2);
    geodesic_points->insert(v3);
  } else {
    Eigen::Vector3f v12 = (v1 + v2).normalized();
    Eigen::Vector3f v13 = (v1 + v3).normalized();
    Eigen::Vector3f v23 = (v2 + v3).normalized();
    SubdivideTriangle(v1, v12, v13, n_divides - 1, geodesic_points);
    SubdivideTriangle(v2, v12, v23, n_divides - 1, geodesic_points);
    SubdivideTriangle(v3, v13, v23, n_divides - 1, geodesic_points);
    SubdivideTriangle(v12, v13, v23, n_divides - 1, geodesic_points);
  }
}

void LineModel::checkLineExtremes(cv::Vec4f& extremes, cv::Size imageSize)
{

	if (extremes[0] < 0)
		extremes[0] = 0;

	if (extremes[0] >= imageSize.width)
		extremes[0] = (float)imageSize.width - 1.0f;

	if (extremes[2] < 0)
		extremes[2] = 0;

	if (extremes[2] >= imageSize.width)
		extremes[2] = (float)imageSize.width - 1.0f;

	if (extremes[1] < 0)
		extremes[1] = 0;

	if (extremes[1] >= imageSize.height)
		extremes[1] = (float)imageSize.height - 1.0f;

	if (extremes[3] < 0)
		extremes[3] = 0;

	if (extremes[3] >= imageSize.height)
		extremes[3] = (float)imageSize.height - 1.0f;
}

}  // namespace srt3d
