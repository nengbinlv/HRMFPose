// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#include <srt3d/EdgeModality.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
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
#include <opencv2/shape/shape_distance.hpp>

#include <assert.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <stdlib.h>

#include <time.h>
#include <math.h>
#include <srt3d/lap.h>

namespace srt3d {

	EdgeModality::EdgeModality(const std::string &name,
		std::shared_ptr<Body> body_ptr,
		std::shared_ptr<EdgeModel> EdgeModel_ptr,
		std::shared_ptr<Camera> camera_ptr)
		: name_{ name },
		body_ptr_{ std::move(body_ptr) },
		EdgeModel_ptr_{ std::move(EdgeModel_ptr) },
		camera_ptr_{ std::move(camera_ptr) } {
		tikhonov_matrix_.setZero();
		tikhonov_matrix_.diagonal().head<3>().array() = tikhonov_parameter_rotation_;
		tikhonov_matrix_.diagonal().tail<3>().array() =
			tikhonov_parameter_translation_;
	}

	bool EdgeModality::SetUp() {
		set_up_ = false;

		// Check if all required objects are set up
		if (!EdgeModel_ptr_->set_up()) {
			std::cout << "EdgeModel " << EdgeModel_ptr_->name() << " was not set up"
				<< std::endl;
			return false;
		}
		if (!camera_ptr_->set_up()) {
			std::cout << "Camera " << camera_ptr_->name() << " was not set up"
				<< std::endl;
			return false;
		}
		if (use_occlusion_handling_ && !occlusion_renderer_ptr_->set_up()) {
			std::cout << "Occlusion renderer " << occlusion_renderer_ptr_->name()
				<< " was not set up" << std::endl;
			return false;
		}

		PrecalculateFunctionLookup();
		//PrecalculateDistributionVariables();
		PrecalculateHistogramBinVariables();
		PrecalculateBodyVariables();
		PrecalculateCameraVariables();
		SetImshowVariables();

		set_up_ = true;
		return true;
	}

	void EdgeModality::set_n_lines(int n_lines) { n_lines_ = n_lines; }

	void EdgeModality::set_function_amplitude(float function_amplitude) {
		function_amplitude_ = function_amplitude;
		set_up_ = false;
	}

	void EdgeModality::set_function_slope(float function_slope) {
		function_slope_ = function_slope;
		set_up_ = false;
	}

	void EdgeModality::set_learning_rate(float learning_rate) {
		learning_rate_ = learning_rate;
	}

	void EdgeModality::set_function_length(int function_length) {
		function_length_ = function_length;
		set_up_ = false;
	}

	void EdgeModality::set_distribution_length(int distribution_length) {
		distribution_length_ = distribution_length;
		set_up_ = false;
	}

	void EdgeModality::set_scales(const std::vector<int> &scales) {
		scales_ = scales;
	}

	void EdgeModality::set_n_newton_iterations(int n_newton_iterations) {
		n_newton_iterations_ = n_newton_iterations;
	}

	void EdgeModality::set_min_continuous_distance(
		float min_continuous_distance) {
		min_continuous_distance_ = min_continuous_distance;
	}

	bool EdgeModality::set_n_histogram_bins(int n_histogram_bins) {
		switch (n_histogram_bins) {
		case 2:
			histogram_bitshift_ = 7;
			break;
		case 4:
			histogram_bitshift_ = 6;
			break;
		case 8:
			histogram_bitshift_ = 5;
			break;
		case 16:
			histogram_bitshift_ = 4;
			break;
		case 32:
			histogram_bitshift_ = 3;
			break;
		case 64:
			histogram_bitshift_ = 2;
			break;
		default:
			std::cerr << "n_histogram_bins = " << n_histogram_bins << " not valid. "
				<< "Has to be of value 2, 4, 8, 16, 32, or 64" << std::endl;
			return false;
		}
		n_histogram_bins_ = n_histogram_bins;
		set_up_ = false;
		return true;
	}

	void EdgeModality::set_learning_rate_f(float learning_rate_f) {
		learning_rate_f_ = learning_rate_f;
	}

	void EdgeModality::set_learning_rate_b(float learning_rate_b) {
		learning_rate_b_ = learning_rate_b;
	}

	void EdgeModality::set_unconsidered_line_length(
		float unconsidered_line_length) {
		unconsidered_line_length_ = unconsidered_line_length;
	}

	void EdgeModality::set_considered_line_length(float considered_line_length) {
		considered_line_length_ = considered_line_length;
	}

	void EdgeModality::set_tikhonov_parameter_rotation(
		float tikhonov_parameter_rotation) {
		tikhonov_parameter_rotation_ = tikhonov_parameter_rotation;
		tikhonov_matrix_.diagonal().head<3>().array() = tikhonov_parameter_rotation_;
	}

	void EdgeModality::set_tikhonov_parameter_translation(
		float tikhonov_parameter_translation) {
		tikhonov_parameter_translation_ = tikhonov_parameter_translation;
		tikhonov_matrix_.diagonal().tail<3>().array() =
			tikhonov_parameter_translation_;
	}

	void EdgeModality::UseOcclusionHandling(
		std::shared_ptr<OcclusionRenderer> occlusion_renderer_ptr) {
		occlusion_renderer_ptr_ = std::move(occlusion_renderer_ptr);
		use_occlusion_handling_ = true;
		set_up_ = false;
	}

	void EdgeModality::DoNotUseOcclusionHandling() {
		occlusion_renderer_ptr_ = nullptr;
		use_occlusion_handling_ = false;
		set_up_ = false;
	}

	void EdgeModality::set_display_visualization(bool display_visualization) {
		display_visualization_ = display_visualization;
	}

	void EdgeModality::StartSavingVisualizations(
		const std::experimental::filesystem::path &save_directory) {
		save_visualizations_ = true;
		save_directory_ = save_directory;
	}

	void EdgeModality::StopSavingVisualizations() {
		save_visualizations_ = false;
	}

	void EdgeModality::set_visualize_lines_correspondence(
		bool visualize_lines_correspondence) {
		visualize_lines_correspondence_ = visualize_lines_correspondence;
		SetImshowVariables();
	}

	void EdgeModality::set_visualize_points_occlusion_mask_correspondence(
		bool visualize_points_occlusion_mask_correspondence) {
		visualize_points_occlusion_mask_correspondence_ =
			visualize_points_occlusion_mask_correspondence;
		SetImshowVariables();
	}

	void EdgeModality::set_visualize_points_pose_update(
		bool visualize_points_pose_update) {
		visualize_points_pose_update_ = visualize_points_pose_update;
		SetImshowVariables();
	}

	void EdgeModality::set_visualize_points_histogram_image_pose_update(
		bool visualize_points_histogram_image_pose_update) {
		visualize_points_histogram_image_pose_update_ =
			visualize_points_histogram_image_pose_update;
		SetImshowVariables();
	}

	void EdgeModality::set_visualize_points_result(bool visualize_points_result) {
		visualize_points_result_ = visualize_points_result;
		SetImshowVariables();
	}

	void EdgeModality::set_visualize_points_histogram_image_result(
		bool visualize_points_histogram_image_result) {
		visualize_points_histogram_image_result_ =
			visualize_points_histogram_image_result;
		SetImshowVariables();
	}

	bool EdgeModality::StartModality() {
		if (!IsSetup()) return false;

		// Initialize histograms
		//这一步计算模型与相机的转换关系，是需要的
		PrecalculatePoseVariables();
		//PrecalculateExtractEdge();

		//mog2->setHistory(10);        //
		//mog2->setVarThreshold(50);   //16
	    //mog2->setDetectShadows(false);
		
		//构建颜色直方图
		if (flag_use_histogram_edge)
		{
			AddLinePixelColorsToTempHistograms();
			//直方图归一化
			if (CalculateHistogram(1.0f, temp_histogram_f_, &histogram_f_) 
				/*&&CalculateHistogram(1.0f, temp_histogram_b_, &histogram_b_)*/) 
			{
				return true;
			}
			else {
				std::cout << "Histograms could not be initialised for modality " << name_
					<< std::endl;
				return false;
			}
		}
		
		//=========计算边缘的视觉外观-基于上一帧跟踪成功的结果==============
		if (flag_use_correlation_) {
			AddEdgePixelGradientForCorrelation();
		}
		return true;
		/*光流法中需要的参考点*/
	}

	bool EdgeModality::CalculateBeforeCameraUpdate() {
		if (!IsSetup()) return false;

		PrecalculatePoseVariables();
		if (flag_use_histogram_edge)
		{
			AddLinePixelColorsToTempHistograms();
			//std::fill(begin(histogram_f_), end(histogram_f_), 0.0f);
		    CalculateHistogram(learning_rate_f_, temp_histogram_f_, &histogram_f_);
			/*for (int i = 0;i < n_histogram_bins_cubed_;i++)
			{
				if ((histogram_f_)[i] != 0)
				{
					cout<< (histogram_f_)[i] <<endl;
				}				
			}*/
		    //CalculateHistogram(learning_rate_b_, temp_histogram_b_, &histogram_b_);
		}
		//AddLinePixelColorsToTempHistograms();
		//CalculateHistogram(learning_rate_f_, temp_histogram_f_, &histogram_f_);
		//CalculateHistogram(learning_rate_b_, temp_histogram_b_, &histogram_b_);
		/*跟踪成功*/

		flag_tracking_success_ = 1;
		//cout<< "edge_error_count_: "<<error_count_ <<endl;

		if (error_count_ <= 1.2)
		{
			flag_tracking_success_ = 1;
			if (flag_use_correlation_)
			{
				AddEdgePixelGradientForCorrelation();
			}		
		}
		else
		{
			flag_tracking_success_ = 0;
		}
		
		return true;
	}

	bool EdgeModality::CalculateCorrespondences(int corr_iteration) {
		//corr_iteration   0,1,2,3,4,5,6
		

		/*光流法找匹配点*/
		//const cv::Mat &image_tt{ camera_ptr_->image() };
		//cv::Mat image_LK = image_tt.clone();
		//cv::Mat mask = cv::Mat::zeros(Last_Frame.size(), Last_Frame.type());
		//cv::Mat frame_gray;
		//cv::cvtColor(image_LK, frame_gray, cv::COLOR_BGR2GRAY);
		//// calculate optical flow
		//vector<uchar> status;
		//vector<float> err;
		//cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
		//cv::Mat Last_Frame_gray;
		//cv::cvtColor(Last_Frame, Last_Frame_gray, cv::COLOR_BGR2GRAY);
		//vector<cv::Point2f> p1;
		//calcOpticalFlowPyrLK(Last_Frame_gray, frame_gray, points_lk_, p1, status, err, cv::Size(15, 15), 2, criteria);
		//vector<cv::Point2f> good_new;
		//
		//vector<cv::Scalar> colors;
		//cv::RNG rng;
		//for (int i = 0; i < 500; i++)
		//{
		//	int r = rng.uniform(0, 256);
		//	int g = rng.uniform(0, 256);
		//	int b = rng.uniform(0, 256);
		//	colors.push_back(cv::Scalar(r, g, b));
		//}
		//for (uint i = 0; i < points_lk_.size(); i++)
		//{
		//	// Select good points
		//	if (status[i] == 1) {
		//		good_new.push_back(p1[i]);
		//		// draw the tracks
		//		cv::circle(image_LK, points_lk_[i], 2, cv::Scalar(0, 0, 255), -1);
		//		cv::line(mask, p1[i], points_lk_[i], colors[i], 1);
		//		cv::circle(image_LK, p1[i], 2, cv::Scalar(0, 255, 0), -1);
		//	}
		//}
		//cv::Mat img;
		//cv::add(image_LK, mask, img);
		//cv::namedWindow("Frame",0);
		//cv::imshow("Frame", img);
		//cv::waitKey(0);

		if (!IsSetup()) return false;
		//cout<< corr_iteration <<endl;
		if (corr_iteration >= 1 && corr_iteration <= 4)
		{
			//cout << corr_iteration << endl;
			flag_num_iter_for_use = 1;
		}
		else
		{
			flag_num_iter_for_use = 0;
		}
		//flag_num_iter_for_use = 1;

		float length_ = LastValidValue(distribution_length_vector_, corr_iteration);
		//cout<< length_ <<endl;

		distribution_length_ = length_;
		PrecalculateDistributionVariables();

		float turkey_noram_real_ = LastValidValue(tukey_norm_constant_vector_, corr_iteration);
		tukey_norm_constant_ = turkey_noram_real_;

		PrecalculatePoseVariables();
		//计算线的长度
		PrecalculateScaleDependentVariables(corr_iteration);

		

		//预计算分布
		//为整个分布再建立一个分布

		function_lookup_edge_distrbution_all.resize(distribution_length_);
		
		float scale_function = LastValidValue(sigma_global_, corr_iteration);
	
		//cout<< scale_function <<endl;
		for (int i = 0; i < distribution_length_; ++i) {
			float x = float(i) - float(distribution_length_ - 1) / 2.0f;
			function_lookup_edge_distrbution_all[i] = 1 / (2.506628 * scale_function)  * std::exp(-(pow(x - mu_, 2) / (2 * scale_function * scale_function)));
		}

		if (use_occlusion_handling_) occlusion_renderer_ptr_->FetchOcclusionMask();

		// Search closest template view
		const EdgeModel::TemplateView *template_view;
		//cout << body2camera_pose_.matrix() << endl;
		EdgeModel_ptr_->GetClosestTemplateView(body2camera_pose_, &template_view);
				
		// Iterate over n_lines
		std::vector<float> segment_probabilities_f(line_length_in_segments_);
		std::vector<float> segment_probabilities_b(line_length_in_segments_);
		data_lines_.clear();
#if 1
		//提取边缘图像
		//cv::Mat sobel_dx;
		//cv::Mat sobel_dy;
		//cv::Mat gray_img;
		const cv::Mat &image{ camera_ptr_->image() };
		////cv::Mat img_show_match_point = image.clone();
		//cv::cvtColor(image, gray_img, CV_BGR2GRAY);
		//cv::Sobel(gray_img, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
		//cv::Sobel(gray_img, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_DEFAULT);

		//cv::Mat mag_temp = cv::Mat::zeros(image.size(), CV_32FC1);
		//cv::Mat ori_temp = cv::Mat::zeros(image.size(), CV_32FC1);
		////==为true时表示以角度表示====
		//cv::cartToPolar(sobel_dx, sobel_dy, mag_temp, ori_temp, false);
		//normalize(mag_temp, mag_temp, 0.0, 255.0, cv::NORM_MINMAX);

		std::vector<std::vector<cv::Vec3b>> current_search_lines_gradient;
		std::vector<std::vector<cv::Point>> index_points;
		std::vector<cv::Point2f> points_edge;
		std::vector<DataLine> dataLines;
		flag_lines_use_.clear();
		cv::Mat1f searchLines_Center(cv::Size(3, n_lines_));
		cv::Mat1f searchLines_Center_LastFrame(cv::Size(3, n_lines_));
		int iii = 0;
		for (auto data_point = begin(template_view->data_points);
			data_point != begin(template_view->data_points) + n_lines_;
			++data_point) {
			/*为建立knn做准备*/
			if (flag_use_correlation_)
			{
				memcpy(searchLines_Center.ptr(iii), &data_point->center_f_body, sizeof(float) * 3);
				memcpy(searchLines_Center_LastFrame.ptr(iii), &lastFrame_CenterPoints_[iii], sizeof(float) * 3);
				iii++;
			}
			
			/*利用前一帧得到的边缘外观计算最相似的匹配点*/
			//center_f_body为模型上的点
			/*Eigen::Vector3f center_f_camera{ body2camera_pose_ *
				data_point->center_f_body };*/

			//Eigen::Vector3f current_center_point = data_point->center_f_body;
			///*找配对的模型3D点*/
			///*速度太慢*/
			//float min_value = 99999;
			//float index = 0;
			//for (int i = 0;i < lastFrame_CenterPoints_.size();i++)
			//{
			//	float dis = (current_center_point - lastFrame_CenterPoints_[i]).norm();
			//	//cout<< dis <<endl;
			//	if (dis < min_value)
			//	{
			//		min_value = dis;
			//		index = i;
			//	}
			//}
			////cout<< min_value <<endl;
			//if (min_value > 0.003)
			//{
			//	//cout<<"找不到对应"<<endl;
			//}
			//Eigen::Vector3f center_f_body = lastFrame_CenterPoints_[index];
			//Eigen::Vector3f center_f_camera{ body2camera_pose_ *
			//	center_f_body };
			//将其转换到图像
			//Eigen::Vector2f center{
			//	center_f_camera(0) * fu_ / center_f_camera(2) + ppu_,
			//	center_f_camera(1) * fv_ / center_f_camera(2) + ppv_ };
			////Eigen::Vector2f normal = lastFrame_CenterNormal_[index];
			////将其转换到图像
			//Eigen::Vector2f normal{
			//	(body2camera_rotation_xy_ * data_point->normal_f_body).normalized() };

			//float u = center(0) - normal(0) * distribution_length_ / 2 - + 0.5f;
			//float v = center(1) - normal(1) * distribution_length_ / 2 + 0.5f;
			//std::vector<cv::Vec3b> temp_search_gradient;
			//std::vector<cv::Point> temp_search_index_point;
			//int n_iteration = int(distribution_length_ + 0.5f);
			//耗时2ms
			//for (int i = 0; i < n_iteration; ++i) {
			//	if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
			//		int(v) > image_height_minus_1_)
			//	{
			//		temp_search_gradient.push_back(cv::Vec3b(0,0,0));
			//		temp_search_index_point.push_back(cv::Point(int(u),int(v)));
			//		break;
			//	}
			//	temp_search_gradient.push_back(image.at<cv::Vec3b>(int(v), int(u)));
			//	temp_search_index_point.push_back(cv::Point(int(u), int(v)));

			//	//img_show_match_point.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(0,0,255);
			//	u += normal(0);
			//	v += normal(1);
			//}
			//current_search_lines_gradient.push_back(temp_search_gradient);
			//index_points.push_back(temp_search_index_point);
		//	/*********************************************/
			DataLine data_line;
			//建立搜索线
			CalculateBasicLineData(*data_point, &data_line);
			dataLines.push_back(data_line);
			//计算包围区域		
			if (!IsLineValid(data_line.center_u, data_line.center_v,
				data_line.continuous_distance))
			{
				flag_lines_use_.push_back(0);
				continue;				
			}
			points_edge.push_back(cv::Point2f(data_line.center_u, data_line.center_v));	
			flag_lines_use_.push_back(1);
		}

		/*cv::Mat bundle_img = cv::Mat::zeros(cv::Size(current_search_lines_gradient.size(), current_search_lines_gradient[0].size()), CV_8UC3);
		for (int i = 0; i < current_search_lines_gradient.size(); i++)
		{
			for (int j = 0; j < current_search_lines_gradient[i].size(); j++)
			{
				bundle_img.at<cv::Vec3b>(j, i) = current_search_lines_gradient[i][j];
			}
		}
		cv::namedWindow("current_img",0);
		cv::imshow("current_img", bundle_img);
		cv::imwrite("current_img.jpg", bundle_img);
		cv::waitKey(0);*/

		//最近邻搜索算法
		cv::flann::Index  knn;
		cv::Mat1i indices;
		cv::Mat1f dists;
		//建立knn索引
		if (flag_use_correlation_)
		{
			knn.build(searchLines_Center_LastFrame, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_L2);
			//如何加速
			//cv::flann::KDTreeIndexParams(1);
			knn.knnSearch(searchLines_Center, indices, dists, 1, cv::flann::SearchParams(32));
		}	
		/**NCC计算互相关*/
		//try 
		//{
		//	//cv::Mat img_match = bundle_img.clone();
		//	//cv::cvtColor(img_match, img_match, CV_GRAY2BGR);

		//	match_ratio_of_lastFrame_.clear();
		//	for (int i = 0; i < current_search_lines_gradient.size(); i++)
		//	{
		//		//cout<< i <<endl;
		//		cv::Mat last_img = cv::Mat::zeros(cv::Size(1, SearchLinesGradient_[i].size()), CV_8UC3);
		//		for (int j = 0; j < SearchLinesGradient_[i].size(); j++)
		//		{
		//			last_img.at<cv::Vec3b>(j, 0) = SearchLinesGradient_[i][j];
		//		}
		//		//cv::imwrite("last_img.jpg", last_img);

		//		cv::Mat current_img = cv::Mat::zeros(cv::Size(1, current_search_lines_gradient[i].size()), CV_8UC3);
		//		for (int j = 0; j < current_search_lines_gradient[i].size(); j++)
		//		{
		//			current_img.at<cv::Vec3b>(j, 0) = current_search_lines_gradient[i][j];

		//			//img_show_match_point.at<cv::Vec3b>(index_points[i][j]) = cv::Vec3b(0,0,255);
		//		}

		//		cv::Mat result;
		//		cv::matchTemplate(current_img, last_img, result, cv::TM_CCORR_NORMED);
		//		/*搜索线有效*/
		//		if (flag_lines_use_[i] == 1)
		//		{
		//			std::vector<float> temp_match_;
		//			for (int i = 0; i < result.rows; i++)
		//			{
		//				temp_match_.push_back(result.at<float>(i, 0));
		//				//cout << result.at<float>(i, 0) << endl;
		//			}
		//			match_ratio_of_lastFrame_.push_back(temp_match_);
		//		}
		//		//cout<<"========"<<endl;
		//		//double minVal, maxVal;
		//		//cv::Point minLoc, maxLoc;
		//		//cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
		//		//// 最大匹配值即为归一化 NCC
		//		//double ncc = maxVal;
		//		//cv::Vec3b color = cv::Vec3b(0, 255, 0) * ncc;

		//		//img_match.at<cv::Vec3b>(maxLoc.y + considered_line_length_/2, i) = color;
		//		//img_show_match_point.at<cv::Vec3b>(index_points[i][maxLoc.y + considered_line_length_ / 2]) = color;
		//	}
		//	/*cv::imwrite("img_match.jpg", img_match);
		//	cv::namedWindow("img_show_match_point",0);
		//	cv::imshow("img_show_match_point", img_show_match_point);
		//	cv::imwrite("img_show_match_point.jpg", img_show_match_point);*/
		//	
		//}
		//catch (cv::Exception& e)
		//{

		//	const char* err_msg = e.what();
		//	cout << err_msg << endl;
		//}	
		rect_roi_ = boundingRect(points_edge);
		/*计算搜索线长度*/
		/*float w = rect_roi_.width;
		float h = rect_roi_.height;
		int max_w_h = int(max(w,h));
		int s = int(0.18 * float(max_w_h) + 15.6);*/
		//cout<< s <<endl;

		PrecalculateExtractEdge();

		cv::Mat result;
		//corr_iteration
		float clamped_x = (corr_iteration < 1) ? 1 : (corr_iteration > 6) ? 6 : corr_iteration;
		//0.3   0.2
		float weight = 0.2 * std::exp(-1.5 * (clamped_x - 1));
		semantic_weight = weight;

		cv::addWeighted(mag_temp, 1 - semantic_weight, pre_edge_, semantic_weight, 0, result);

		cv::Mat mag;
		result.convertTo(result, CV_32FC1, 1.0 / 255);
		normalize(result, mag, 0.0, 1.0, cv::NORM_MINMAX);
		mag_ = mag;
#endif
		//show_gmm = camera_ptr_->image().clone();
		//mean_vector_.clear();
		//index_for_match_lastFrame_ = 0;
		//cout<< bundle_vector .size()<<endl;
		/*for (auto data_point = begin(template_view->data_points);
			data_point != begin(template_view->data_points) + n_lines_;
			++data_point) 	*/
		current_mat_vector.clear();
		last_mat_vector.clear();
		match_result_vector.clear();
//#pragma omp parallel for
		for(int i = 0;i < dataLines.size();i++)
		{
			DataLine data_line = dataLines[i];
			//DataLine data_line;
			//建立搜索线
			//CalculateBasicLineData(*data_point, &data_line);
			//计算得到normal_component_to_scale 和 delta_r
			/*if (!IsLineValid(data_line.center_u, data_line.center_v,
				data_line.continuous_distance))
			{
				continue;
			}*/
			if (flag_use_correlation_)
				index_for_match_lastFrame_ = indices.at<int>(i);
			if (flag_lines_use_[i] == 0)
			{
				continue;
			}
			
			/*上一帧的图像*/
			if (flag_use_correlation_)
			{
				float dis = dists.at<float>(i);
				if (dis * 1000 > 0.3)
				{
					flag_num_iter_for_use = 0;
				}
				else
				{
					flag_num_iter_for_use = 1;
				}
				cv::Mat last_img = cv::Mat::zeros(cv::Size(1, Lasr_Frame_Search_Line_img.rows), CV_8UC3);
				for (int j = 0; j < Lasr_Frame_Search_Line_img.rows; j++)
				{
					last_img.at<cv::Vec3b>(j, 0) = Lasr_Frame_Search_Line_img.at<cv::Vec3b>(j, index_for_match_lastFrame_);
				}
				Last_Frame_one_line = last_img;
				
				last_mat_vector.push_back(Last_Frame_one_line);
			}		

#if 1
			/*如何并行计算*/
			if (!CalculateSegmentProbabilities_edge(
				data_line.center_u, data_line.center_v, data_line.normal_u,
				data_line.normal_v, &segment_probabilities_f,
				&segment_probabilities_b, &data_line.normal_component_to_scale,
				&data_line.delta_r, &data_line.distribution, &data_line.mean, &data_line.standard_deviation, &data_line.variance,
				data_line.vimg_desc))
			{
				continue;
			}			
			data_lines_.push_back(std::move(data_line));
#endif			
			/*CalculateEdgeSearchLineDistribution(data_line.center_u, data_line.center_v, data_line.normal_u, data_line.normal_v,
				&data_line.distribution);

			CalculateDistributionMoments(data_line.distribution, &data_line.mean,
				&data_line.standard_deviation,
				&data_line.variance);*/	
		}

		/*测试搜索线提取的像素值*/
		if (0)  //flag_num_iter_for_use
		{
			cv::Mat img = cv::Mat::zeros(cv::Size(last_mat_vector.size(), last_mat_vector[0].rows), CV_8UC3);
			for (int i = 0; i < last_mat_vector.size(); i++)
			{
				cv::Mat mm = last_mat_vector[i];
				for (int j = 0; j < mm.rows; j++)
				{
					img.at<cv::Vec3b>(j, i) = mm.at<cv::Vec3b>(j, 0);
				}
			}

			cv::Mat img_c = cv::Mat::zeros(cv::Size(current_mat_vector.size(), current_mat_vector[0].rows), CV_8UC3);
			for (int i = 0; i < current_mat_vector.size(); i++)
			{
				cv::Mat mm = current_mat_vector[i];
				for (int j = 0; j < mm.rows; j++)
				{
					img_c.at<cv::Vec3b>(j, i) = mm.at<cv::Vec3b>(j, 0);
				}
				img_c.at<cv::Vec3b>(match_result_vector[i].y + considered_line_length_ncc_ / 2, i) = cv::Vec3b(0, 255, 0);
			}

			cv::imwrite("img.jpg", img);
			cv::imwrite("img_c.jpg", img_c);
			cv::imshow("img", img);
			cv::imshow("show_gmm", img_c);
			cv::waitKey(0);
		}
		
		//cout<< data_lines_ .size()<<endl;
		//计算高斯混合模型
		//preGMM();

		return true;
	}
	bool EdgeModality::preGMM(float center_u, float center_v, float normal_u, float normal_v,int index_line, std::vector<cv::Vec3b> *value) const
	{
		//只在尺度为2时计算
		if (scale_ == 1)
		{
			const cv::Mat &image{ camera_ptr_->image() };

			if (std::fabs(normal_v) < std::fabs(normal_u)) {
				// Calculate step and starting position
				float v_step = normal_v / normal_u;
				// Notice: u = int(center_u - (line_length / 2 - 0.5) + 0.5)
				int u = int(center_u - line_length_half_minus_1_);
				int u_end = u + line_length_minus_1_;
				float v_f = center_v + v_step * (float(u) - center_u) + 0.5f;
				float v_f_end = v_f + v_step * float(line_length_minus_1_);

				// Check if line is on image (margin of 1 for rounding errors of v_f_end)
				if (u < 0 || u_end > image_width_minus_1_ || int(v_f) < 0 ||
					int(v_f) > image_height_minus_1_ || int(v_f_end) < 1 ||
					int(v_f_end) > image_height_minus_2_) {
					return false;
				}

				// Iterate over all pixels of line and calculate probabilities
				if (normal_u > 0) {
					int col = 0;
					for (; u <= u_end; ++u, v_f += v_step) {
						value->push_back(image.at<cv::Vec3b>(int(v_f),u));
						col++;
					}
				}
				else {
					int col = 0;
					for (; u <= u_end; ++u, v_f += v_step) {
						
						value->push_back(image.at<cv::Vec3b>(int(v_f), u));
						col++;
					}
				}
			}
			else {
				// Calculate step and starting position
				float u_step = normal_u / normal_v;
				// Notice: v = int(center_v - (line_length / 2 - 0.5) + 0.5)
				int v = int(center_v - line_length_half_minus_1_);
				int v_end = v + line_length_minus_1_;
				float u_f = center_u + u_step * (float(v) - center_v) + 0.5f;
				float u_f_end = u_f + u_step * float(line_length_minus_1_);

				// Check if line is on image (margin of 1 for rounding errors of u_f_end)
				if (v < 0 || v_end > image_height_minus_1_ || int(u_f) < 0 ||
					int(u_f) > image_width_minus_1_ || int(u_f_end) < 1 ||
					int(u_f_end) > image_width_minus_2_) {
					return false;
				}
				if (normal_v > 0) {
					int col = 0;
					for (; v <= v_end; ++v, u_f += u_step) {
						value->push_back(image.at<cv::Vec3b>(int(v), int(u_f)));
						col++;
					}
				}
				else {
					int col = 0;
					for (; v <= v_end; ++v, u_f += u_step) {
						value->push_back(image.at<cv::Vec3b>(int(v), int(u_f)));
						col++;
					}
				}
			}
		}
		
	}
	bool EdgeModality::CalculateEdgeSearchLineDistribution(float center_u, float center_v, float normal_u, float normal_v, 
		std::vector<float> *distribution) {

		cv::Mat visualization_image{ camera_ptr_->image() };
		
		//cv::Mat visualization_image;
		//cv::cvtColor(Canny_Mat_.clone(), visualization_image,CV_GRAY2BGR);

		if (0)
		{
			if (std::fabs(normal_v) < std::fabs(normal_u)) {
				// Calculate step and starting position
				float v_step = normal_v / normal_u;
				// Notice: u = int(center_u - (line_length / 2 - 0.5) + 0.5)
				int u = int(center_u - line_length_half_minus_1_);
				int u_end = u + line_length_minus_1_;
				distribution->resize(line_length_minus_1_);

				float v_f = center_v + v_step * (float(u) - center_u) + 0.5f;
				float v_f_end = v_f + v_step * float(line_length_minus_1_);

				// Check if line is on image (margin of 1 for rounding errors of v_f_end)
				if (u < 0 || u_end > image_width_minus_1_ || int(v_f) < 0 ||
					int(v_f) > image_height_minus_1_ || int(v_f_end) < 1 ||
					int(v_f_end) > image_height_minus_2_) {
					return false;
				}
				auto distribution_it = begin(*distribution);

				//计算距离
				int segment_idx = 0;
				for (; u <= u_end; ++u, v_f += v_step, ++distribution_it) {
					*distribution_it = 1.0f;

					//对图像进行处理提取边缘或提取信息
					//先对图像进行边缘提取，其次计算搜索线上为边缘的点到中心的turkey距离	
					if (distribution_it == end(*distribution))
					{
						break;
					}

					if (Canny_Mat_.at<uchar>(int(v_f), u))
					{
						//计算turkey距离值
						float ex = normal_u * (center_u - u) + normal_v * (center_v - int(v_f));
						//int turkey_weiht = 10;
						/*float turkey = 0;
						if (fabs(ex) <= turkey_weiht)
						{
						turkey = (1 - (ex / turkey_weiht)*(ex / turkey_weiht))*(1 - (ex / turkey_weiht)*(ex / turkey_weiht))*(1 - (ex / turkey_weiht)*(ex / turkey_weiht));
						}
						else
						turkey = 0;*/
						*distribution_it *= ex;
						//visualization_image.at<cv::Vec3b>(int(v_f), u) = cv::Vec3b(255, 255, 255);
					}
					else
						*distribution_it *= 1000;
				}
			}
			else {
				// Calculate step and starting position
				float u_step = normal_u / normal_v;
				// Notice: v = int(center_v - (line_length / 2 - 0.5) + 0.5)
				int v = int(center_v - line_length_half_minus_1_);
				int v_end = v + line_length_minus_1_;
				distribution->resize(line_length_minus_1_);

				float u_f = center_u + u_step * (float(v) - center_v) + 0.5f;
				float u_f_end = u_f + u_step * float(line_length_minus_1_);

				// Check if line is on image (margin of 1 for rounding errors of u_f_end)
				if (v < 0 || v_end > image_height_minus_1_ || int(u_f) < 0 ||
					int(u_f) > image_width_minus_1_ || int(u_f_end) < 1 ||
					int(u_f_end) > image_width_minus_2_) {
					return false;
				}
				auto distribution_it = begin(*distribution);

				//计算距离
				for (; v <= v_end; ++v, u_f += u_step, ++distribution_it) {
					//对图像进行处理提取边缘或提取信息
					*distribution_it = 1.0f;

					//先对图像进行边缘提取，其次计算搜索线上为边缘的点到中心的turkey距离
					if (distribution_it == end(*distribution))
					{
						break;
					}

					if (Canny_Mat_.at<uchar>(v, int(u_f)))
					{
						//计算turkey距离值
						float ex = normal_u * (center_u - int(u_f)) + normal_v * (center_v - v);
						/*int turkey_weiht = 30;
						float turkey = 0;
						if (fabs(ex) <= turkey_weiht)
						{
						turkey = (1 - (ex / turkey_weiht)*(ex / turkey_weiht))*(1 - (ex / turkey_weiht)*(ex / turkey_weiht))*(1 - (ex / turkey_weiht)*(ex / turkey_weiht));
						}
						else
						turkey = 0;*/
						*distribution_it *= ex;
						//visualization_image.at<cv::Vec3b>(v, int(u_f)) = cv::Vec3b(255, 255, 255);
					}
					else
						*distribution_it *= 1000;

				}
				// define normal component and calculate delta_r
			}
		}
		// Select case if line is more horizontal or vertical
		if (1)
		{
			float scale_minus_1_half_ = (fscale_ - 1.0f) / 2.0f;
			int u, v;
			//搜索线的分布大小
			//distribution->resize(line_length_minus_1_);
			distribution->resize(distribution_length_ * scale_,1.0f);
			auto distribution_it = begin(*distribution);
			for (int i = 0; i < distribution_length_; ++i) {
				for (int j = 0; j < scale_; ++j) {
					//首先绘制的是那个点？
					if (std::fabs(normal_u) > std::fabs(normal_v)) {
						u = int(
							center_u +
							float(sgn(normal_u)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						v = int(center_v +
							(float(u) - center_u) *
							(normal_v / normal_u) +
							0.5f);

						if (v < 0 || v > image_height_minus_1_ || u < 0 || u > image_width_minus_1_ ) {
							return false;
						}

						*distribution_it = 1.0f;
						//先对图像进行边缘提取，其次计算搜索线上为边缘的点到中心的turkey距离
						if (distribution_it == end(*distribution))
						{
							break;
						}

						//visualization_image.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 255, 255);
#if 0 
						if (Canny_Mat_.at<uchar>(v, u))
						{
							float angle_real = line_angle.at<float>(v, u);
							float angle_virtual = atan2(normal_v, normal_u);
							float distance_angle = fabs(angle_real - angle_virtual) * 180 / CV_PI;
							int angle_thresh = 180;
							/*cout << "angle_real: "<<angle_real << endl;
							cout << "angle_virtual: " << angle_virtual << endl;
							cout << "distance_angle: " << distance_angle << endl;*/
							if (distance_angle <= angle_thresh || (distance_angle > 180 - angle_thresh && distance_angle < 180 + angle_thresh))
							{
								float ex = normal_u * (center_u - u) + normal_v * (center_v - v);
								*distribution_it *=   ex;
								//cout << "ex:"<<ex << endl;
								//cv::circle(visualization_image,cv::Point(u,v),5,cv::Scalar(0,255,0),-1);
								//visualization_image.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 255, 255);
							}						
						}						
						else
							*distribution_it *= 9999;						
#endif
						//距离
						float ex = normal_u * (center_u - u) + normal_v * (center_v - v);
						float turkey_weiht = 20;
						//计算turkey，使得[0-1]分布
						float turkey = (1 - (ex / turkey_weiht)*(ex / turkey_weiht))
							*(1 - (ex / turkey_weiht)*(ex / turkey_weiht));
						//再乘以图像梯度
						float mag = mag_.at<float>(v, u);
						//边缘提取的响应值
						if (mag > 0.1)
						{
							float angle_real = ori_.at<float>(v, u);
							float angle_virtual = atan2(normal_v, normal_u);
							float distance_angle = fabs(angle_real - angle_virtual) * 180 / CV_PI;
							int angle_thresh = 180;
							/*cout << "angle_real: " << angle_real << endl;
							cout << "angle_virtual: " << angle_virtual << endl;
							cout << "distance_angle: " << distance_angle << endl;*/
							if (distance_angle <= angle_thresh || (distance_angle > 180 - angle_thresh && distance_angle < 180 + angle_thresh))
							{
							     turkey = turkey * mag;
							}
							//turkey = turkey * mag;
							else
							turkey = 0;
							    *distribution_it *= turkey;
						}
						else
							*distribution_it *= 0;
						
					}
					else {
						v = int(
							center_v +
							float(sgn(normal_v)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						u = int(center_u +
							(float(v) - center_v) *
							(normal_u / normal_v) +
							0.5f);
						if (v < 0 || v > image_height_minus_1_ || u < 0 || u > image_width_minus_1_) {
							return false;
						}
						*distribution_it = 1.0f;
						//先对图像进行边缘提取，其次计算搜索线上为边缘的点到中心的turkey距离
						if (distribution_it == end(*distribution))
						{
							break;
						}

						//visualization_image.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 255, 255);
#if 0 
						if (Canny_Mat_.at<uchar>(v, u))
						{
							float angle_real = line_angle.at<float>(v, u);
							float angle_virtual = atan2(normal_v, normal_u);
							float distance_angle = fabs(angle_real - angle_virtual) * 180 / CV_PI;
							int angle_thresh = 180;
							/*cout << "angle_real: " << angle_real << endl;
							cout << "angle_virtual: " << angle_virtual << endl;
							cout << "distance_angle: " << distance_angle << endl;*/
							if (distance_angle <= angle_thresh || (distance_angle > 180 - angle_thresh && distance_angle < 180 + angle_thresh))
							{
								float ex = normal_u * (center_u - u) + normal_v * (center_v - v);
								*distribution_it *= ex;
								//cout << "ex:" << ex << endl;
								//cv::circle(visualization_image, cv::Point(u, v), 5, cv::Scalar(0, 255, 0), -1);
								//visualization_image.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 255, 255);
							}
						}
						else
							*distribution_it *= 9999;
#endif
						float ex = normal_u * (center_u - u) + normal_v * (center_v - v);
						float turkey_weiht = 20;
						//计算turkey，使得[0-1]分布
						float turkey = (1 - (ex / turkey_weiht)*(ex / turkey_weiht))
							*(1 - (ex / turkey_weiht)*(ex / turkey_weiht));
						//再乘以图像梯度
						float mag = mag_.at<float>(v, u);
						if (mag > 0.1)
						{
							float angle_real = ori_.at<float>(v, u);
							float angle_virtual = atan2(normal_v, normal_u);
							float distance_angle = fabs(angle_real - angle_virtual) * 180 / CV_PI;
							int angle_thresh = 180;
							/*cout << "angle_real: " << angle_real << endl;
							cout << "angle_virtual: " << angle_virtual << endl;
							cout << "distance_angle: " << distance_angle << endl;*/
							if (distance_angle <= angle_thresh || (distance_angle > 180 - angle_thresh && distance_angle < 180 + angle_thresh))
							{
							    turkey = turkey * mag;
							}
							//turkey = turkey * mag;
							else
							    turkey = 0;

							*distribution_it *= turkey;
						}
						else
							*distribution_it *= 0;
					}

					++distribution_it;
				}
			}
		}

		if (0)
		{
			cv::namedWindow("cc", 0);
			cv::imshow("visualization_image", visualization_image);
			cv::waitKey(0);
			//cv::imwrite("test.jpg", visualization_image);
		}
		
		return true;
	}

	bool EdgeModality::VisualizeCorrespondences(int save_idx) {
		if (!IsSetup()) return false;

		if (visualize_lines_correspondence_)
			VisualizeLines("lines_correspondence", save_idx);
		if (visualize_points_occlusion_mask_correspondence_ &&
			use_occlusion_handling_)
			VisualizePointsOcclusionMask("occlusion_mask_correspondence", save_idx);
		return true;
	}

	//计算位姿
	bool EdgeModality::CalculatePoseUpdate(int corr_iteration,
		int update_iteration) {
		if (!IsSetup()) return false;

		PrecalculatePoseVariables();
		Eigen::Matrix<float, 6, 1> gradient;
		Eigen::Matrix<float, 6, 6> hessian;
		gradient.setZero();
		hessian.setZero();
	
		gradient_edge.setZero();
		hessian_edge.setZero();
		//==================计算误差以及方向向量================================//	
#if 0
		int angle_left_num = 0;
		int angle_right_num = 0;
		int angle_up_num = 0;
		int angle_down_num = 0;
		float error_count = 0.0f;
		cv::Mat img = camera_ptr_->image().clone();
		vector <cv::Point> contourPts1;
		vector <cv::Point> contourPts2;
		for (auto &data_line : data_lines_) {

			data_line.center_f_camera = body2camera_pose_ * data_line.center_f_body;
			//cout<< body2camera_pose_.matrix() <<endl;
			//估计的轮廓？？？
			float x = data_line.center_f_camera(0);
			float y = data_line.center_f_camera(1);
			float z = data_line.center_f_camera(2);

			// Calculate delta_cs
			float fu_z = fu_ / z;
			float fv_z = fv_ / z;
			//
			float xfu_z = x * fu_z;
			float yfv_z = y * fv_z;
			//意义：dsi
			// normal_component_to_scale   ni/s
			//cout<<"u1:"<< xfu_z + ppu_ <<endl;
			//cout << "u2:" << data_line.center_u << endl;
			/*cv::circle(img, cv::Point(xfu_z + ppu_, yfv_z + ppv_), 1, cv::Scalar(0, 0, 255), -1);
			cv::circle(img, cv::Point(data_line.center_u, data_line.center_v), 1, cv::Scalar(0, 255, 0), -1);
			cv::imshow("img", img);
			cv::waitKey(0);*/
			//计算前后两次的变化量
			float delta_cs = (data_line.normal_u * (xfu_z + ppu_ - data_line.center_u) +
				data_line.normal_v * (yfv_z + ppv_ - data_line.center_v) -
				data_line.delta_r) *
				data_line.normal_component_to_scale;
			//计算误差===============================
			float max_prob = 0;
			float scale_minus_1_half_ = (fscale_ - 1.0f) / 2.0f;
			int u, v;
			int x_, y_;
			std::vector<std::pair<cv::Point, float>> prob_v;
			cout << "===start===" << endl;
			for (int i = 0; i < distribution_length_; ++i) {
				for (int j = 0; j < scale_; ++j) {
					if (std::fabs(data_line.normal_u) > std::fabs(data_line.normal_v)) {
						u = int(
							data_line.center_u +
							float(sgn(data_line.normal_u)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						v = int(data_line.center_v +
							(float(u) - data_line.center_u) *
							(data_line.normal_v / data_line.normal_u) +
							0.5f);
					}
					else {
						v = int(
							data_line.center_v +
							float(sgn(data_line.normal_v)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						u = int(data_line.center_u +
							(float(v) - data_line.center_v) *
							(data_line.normal_u / data_line.normal_v) +
							0.5f);
					}
					float prob = data_line.distribution[i];
					cout << prob << endl;
					prob_v.push_back(std::make_pair(cv::Point(x_, y_), prob));
					if (prob > max_prob)
					{
						max_prob = prob;
						x_ = u;
						y_ = v;
					}
				}
			}
			cout << "===end===" << endl;
			//sort(prob_v.begin(), prob_v.end(), [](std::pair<cv::Point, float> x, std::pair<cv::Point, float> y) {return x.second > y.second; });
			/*if (max_prob < 0.3)
			{
				continue;
			}*/
			//绘制前三个点

			/*cv::circle(img, prob_v[0].first, 1, cv::Scalar(0, 255, 255), -1);
			cv::circle(img, prob_v[1].first, 1, cv::Scalar(0, 255, 255), -1);
			cv::circle(img, prob_v[2].first, 1, cv::Scalar(0, 255, 255), -1);*/

			float dloglikelihood_ddelta_cs;
			if (update_iteration < n_newton_iterations_) {
				dloglikelihood_ddelta_cs =
					(data_line.mean - delta_cs) / data_line.variance;
			}
			else {
				int dist_idx_upper = int(delta_cs + distribution_length_plus_1_half_);
				int dist_idx_lower = dist_idx_upper - 1;
				if (dist_idx_lower < 0 || dist_idx_upper >= distribution_length_)
					continue;
				dloglikelihood_ddelta_cs =
					(std::log(data_line.distribution[dist_idx_upper]) -
						std::log(data_line.distribution[dist_idx_lower])) *
					learning_rate_ / (data_line.variance + 0.00001);
			}
			if (scale_ != 0)
			{
				//error_count += error * max_prob;
				error_count += fabs(dloglikelihood_ddelta_cs);
			}

			//计算方向向量
			Eigen::Vector2f ori_vetor(data_line.center_u - x_, data_line.center_v - y_);
			float angle_v2r = angle_2(cv::Point(x_, y_), cv::Point(data_line.center_u, data_line.center_v));
			angle_v2r = angle_v2r * 180 / CV_PI;
			
			//当误差较大的时候启动方向向量投票机制
			if (angle_v2r >= -20 && angle_v2r <= 20)
			{
				angle_left_num++;
			}
			if ((angle_v2r >= -180 && angle_v2r <= -160) || (angle_v2r >= 160 && angle_v2r <= 180))
			{
				angle_right_num++;
			}
			if (angle_v2r >= 70 && angle_v2r <= 110)
			{
				angle_up_num++;
			}
			if (angle_v2r >= -110 && angle_v2r <= -70)
			{
				angle_down_num++;
			}
			
			cv::circle(img, cv::Point(x_, y_), 1, cv::Scalar(0, 255, 255), -1);
			cv::circle(img, cv::Point(data_line.center_u, data_line.center_v), 1, cv::Scalar(0, 0, 255), -1);
			cv::line(img, cv::Point(x_, y_), cv::Point(data_line.center_u, data_line.center_v), cv::Scalar(255, 0, 0), 1);
			//计算两个点偏移方向

			//cv::namedWindow("img",0);
			//cv::imshow("img", img);
			//cv::waitKey(0);
			//===保存匹配数据===
			contourPts1.push_back(cv::Point(x_, y_));
			contourPts2.push_back(cv::Point(data_line.center_u, data_line.center_v));
		}
		error_count_ = error_count / data_lines_.size();
		
		cv::imshow("img",img);
		cv::waitKey(0);
		//cout<< "edge_error_count: "<< error_count_ <<endl;
		//cout << "angle_left_num: " << angle_left_num << endl;
		//cout << "angle_right_num: " << angle_right_num << endl;
		//cout << "angle_up_num: " << angle_up_num << endl;
		//cout << "angle_down_num: " << angle_down_num << endl;
		//***********计算形状相似度********************//
		//CalShape(contourPts1, contourPts2);
#endif
	//==========多峰都考虑的方案============//
#if 1
		float error_count = 0.0f;
		int index_dataline = 0;
		vector <cv::Point> contourPts1;
		vector <cv::Point> contourPts2;
		float match_num = 0;
		//cv::Mat img = camera_ptr_->image().clone();

		for (auto &data_line : data_lines_) {
			// Calculate point coordinates in camera frame
			data_line.center_f_camera = body2camera_pose_ * data_line.center_f_body;
			//cout<< body2camera_pose_.matrix() <<endl;
			//估计的轮廓？？？
			float x = data_line.center_f_camera(0);
			float y = data_line.center_f_camera(1);
			float z = data_line.center_f_camera(2);

			// Calculate delta_cs
			float fu_z = fu_ / z;
			float fv_z = fv_ / z;
			//
			float xfu_z = x * fu_z;
			float yfv_z = y * fv_z;

			//计算误差===============================
			float max_prob = 0;
			float scale_minus_1_half_ = (fscale_ - 1.0f) / 2.0f;
			int u, v;
			int x_, y_;
			//cout << "===start===" << endl;
			for (int i = 0; i < distribution_length_; ++i) {
				for (int j = 0; j < scale_; ++j) {
					if (std::fabs(data_line.normal_u) > std::fabs(data_line.normal_v)) {
						u = int(
							data_line.center_u +
							float(sgn(data_line.normal_u)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						v = int(data_line.center_v +
							(float(u) - data_line.center_u) *
							(data_line.normal_v / data_line.normal_u) +
							0.5f);
					}
					else {
						v = int(
							data_line.center_v +
							float(sgn(data_line.normal_v)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						u = int(data_line.center_u +
							(float(v) - data_line.center_v) *
							(data_line.normal_u / data_line.normal_v) +
							0.5f);
					}
					float prob = data_line.distribution[i];
					
					/*cv::Vec3b s1 = { 24, 184, 234 };
					cv::Vec3b s2 = { 61, 63, 179 };
					float color_ratio = std::min(3 * data_line.distribution[i], 1.0f);
					img.at<cv::Vec3b>(v, u) = color_ratio * s2 +
						(1.0f - color_ratio) * s1;*/
					//cout << prob << endl;
					/*cv::namedWindow("img", 0);
					cv::imshow("img", img);
					cv::waitKey(0);*/
					if (prob > max_prob)
					{
						max_prob = prob;
						x_ = u;
						y_ = v;
					}
				}
			}
			/*cv::namedWindow("img",0);
			cv::imshow("img", img);
			cv::waitKey(0);*/
			//cout<<"======================"<<endl;

			contourPts1.push_back(cv::Point(x_, y_));
			contourPts2.push_back(cv::Point(data_line.center_u, data_line.center_v));
			if (max_prob >= 0.45)  //0.3  0.55 * scale_   0.4
			{
				match_num++;
			}
			//if (max_prob <= 0.25)  //0.3  0.55 * scale_
			//{
			//	continue;
			//}
			//意义：dsi
			// normal_component_to_scale   ni/s
			//cout<<"u1:"<< xfu_z + ppu_ <<endl;
			//cout << "u2:" << data_line.center_u << endl;
			/*cv::circle(img, cv::Point(xfu_z + ppu_, yfv_z + ppv_), 1, cv::Scalar(0, 0, 255), -1);
			cv::circle(img, cv::Point(data_line.center_u, data_line.center_v), 1, cv::Scalar(0, 255, 0), -1);
			cv::imshow("img", img);
			cv::waitKey(0);*/
			//计算前后两次的变化量
			float delta_cs = (data_line.normal_u * (xfu_z + ppu_ - data_line.center_u) +
				data_line.normal_v * (yfv_z + ppv_ - data_line.center_v) -
				data_line.delta_r) *
				data_line.normal_component_to_scale;

			//=======确定是否过滤====
#if flag_use_filter
			//这个阈值可能需要调控
			
			if (error_count_ > 3.0)
			{
				float angle_v2r = angle_2(contourPts1[index_dataline], contourPts2[index_dataline]);
				angle_v2r = angle_v2r * 180 / CV_PI;
				//cout<< angle_v2r <<endl;
				//===判断左右方向
				float div_left_right = float(angle_left_num) / float(angle_right_num);
				//===向左的优化的概率较大
				if (div_left_right > 1.5)
				{			
					if ((angle_v2r >= -180 && angle_v2r <= -160) || (angle_v2r >= 160 && angle_v2r <= 180))
					{
						index_dataline++;
						continue;
					}
				}
				if(div_left_right < 0.667)
				{
					if (angle_v2r >= -20 && angle_v2r <= 20)
					{
						index_dataline++;
						continue;
					}
				}
				//===判断上下方向
				float div_up_down = float(angle_up_num) / float(angle_down_num);
				//主要是向上
				if (div_up_down > 1.5)
				{
					if (angle_v2r >= -110 && angle_v2r <= -70)
					{
						index_dataline++;
						continue;
					}
				}
				if(div_up_down < 0.667)
				{
					if (angle_v2r >= 70 && angle_v2r <= 110)
					{
						index_dataline++;
						continue;
					}
				}
				
			}
#endif
			/*cv::circle(img, contourPts1[index_dataline], 1, cv::Scalar(0, 255, 255), -1);
			cv::circle(img, contourPts2[index_dataline], 1, cv::Scalar(0, 0, 255), -1);
			cv::line(img, contourPts1[index_dataline], contourPts2[index_dataline], cv::Scalar(255, 0, 0), 1);*/
			//cv::imshow("img", img);
			//cv::waitKey(0);

			//分布的变化太大，可能是异常点
			/*if (mean_vector_[index_dataline] > 3)
			{
				index_dataline++;
				continue;
			}*/
			index_dataline++;			
			//cout<< max_prob <<endl;			
			// Calculate first derivative of loglikelihood with respect to delta_cs
			float dloglikelihood_ddelta_cs;
			/*dloglikelihood_ddelta_cs =
				(data_line.mean - delta_cs) / (data_line.variance + 0.00001);*/
			//cout << "data_line.mean: " << data_line.mean << endl;
			//cout << "delta_cs: " << delta_cs << endl;
			//cout << "data_line.variance: " << data_line.variance << endl;
			//n_newton_iterations_
			//cout<<"data_line.mean: "<< data_line.mean <<endl;
			//cout<<"data_line.variance: "<<data_line.variance <<endl;
			error_count = error_count + fabs((data_line.mean));
			if (update_iteration < n_newton_iterations_) {
				dloglikelihood_ddelta_cs =
					(data_line.mean - delta_cs) / data_line.variance;
			}
			else {
				// Calculate distribution indexes
				// Note: (distribution_length - 1) / 2 + 1 = (distribution_length + 1) / 2
				//匹配点基本就在中心位置，进行更加精确的计算？
				int dist_idx_upper = int(delta_cs + distribution_length_plus_1_half_);
				int dist_idx_lower = dist_idx_upper - 1;
				if (dist_idx_lower < 0 || dist_idx_upper >= distribution_length_)
					continue;

				//加入学习率
				dloglikelihood_ddelta_cs =
					(std::log(data_line.distribution[dist_idx_upper]) -
						std::log(data_line.distribution[dist_idx_lower])) *
					learning_rate_ / (data_line.variance + 0.00001);

				//cout<<"log-upper"<< data_line.distribution[dist_idx_upper] <<endl;
				//cout << "log-lower" << data_line.distribution[dist_idx_lower] << endl;
				//cout << "local:" << dloglikelihood_ddelta_cs << endl;
			}
			
			// Calculate first order derivative of delta_cs with respect to theta
			Eigen::RowVector3f ddelta_cs_dcenter{
				data_line.normal_component_to_scale * data_line.normal_u * fu_z,
				data_line.normal_component_to_scale * data_line.normal_v * fv_z,
				data_line.normal_component_to_scale *
				(-data_line.normal_u * xfu_z - data_line.normal_v * yfv_z) / z };
			Eigen::RowVector3f ddelta_cs_dtranslation{ ddelta_cs_dcenter *
				body2camera_rotation_ };

			//求解的是XC对thta的一阶导数
			Eigen::Matrix<float, 1, 6> ddelta_cs_dtheta;
			ddelta_cs_dtheta << data_line.center_f_body.transpose().cross(
				ddelta_cs_dtranslation),
				ddelta_cs_dtranslation;

			// Calculate gradient and hessian
			//梯度
			//cout << "dloglikelihood_ddelta_cs: " << dloglikelihood_ddelta_cs << endl;
			//cout << "ddelta_cs_dtheta.transpose(): " << ddelta_cs_dtheta.transpose() << endl;

			// Calculate weight
			// 权重设计
			float weight = min_variance_ /
				(data_line.normal_component_to_scale * data_line.normal_component_to_scale * variance_);
			
			//cout << "edge_weight: "<< weight << endl;
			//权重很重要--实际上其中已经加了权重--如果再加，那就是为了平衡两种方法

			gradient_edge += dloglikelihood_ddelta_cs * ddelta_cs_dtheta.transpose();
			
			//cout << "gradient_edge: " << gradient_edge << endl;
			//除以标准差
			ddelta_cs_dtheta /= data_line.standard_deviation;
			//计算hessian矩阵
			//triangularView<Eigen::Lower>()下三角矩阵，其他部分为0
			hessian_edge.triangularView<Eigen::Lower>() -= 
				ddelta_cs_dtheta.transpose() * ddelta_cs_dtheta;

		}

		//===利用向量图进行高斯牛顿优化====
#if 0
        //关键点坐标
		std::vector<Eigen::Vector3f> points3ds_body = points3d_body;
		std::vector<Eigen::Vector2f> pre_2d_points;
		std::vector<Eigen::Vector2f> pre_2d_vector;
		pre_2d_points = pre_heatmap_2dpoints;
		pre_2d_vector = pre_edge_vector_2d;
		//一共八个点
		int index_edge_vector = 0;
		for (int i = 0; i< points3ds_body.size() - 1;i++)
		{
			for (int j = i + 1; j < points3ds_body.size(); j++)
			{
				//开始的点、结束的点
				Eigen::Vector3f points3d_body_start = points3ds_body[i];
				Eigen::Vector3f points3d_body_end = points3ds_body[j];
				Eigen::Vector3f points3d_cameara_start = body2camera_pose_ * points3d_body_start;
				Eigen::Vector3f points3d_cameara_end = body2camera_pose_ * points3d_body_end;
				//cout<< body2camera_pose_.matrix() <<endl;
				float x = points3d_cameara_start(0);
				float y = points3d_cameara_start(1);
				float z = points3d_cameara_start(2);
				float z2 = z * z;
				// 计算关键点的预测和当前的偏差
				float center_u_start = points3d_cameara_start(0) * fu_ / points3d_cameara_start(2) + ppu_;
				float center_v_start = points3d_cameara_start(1) * fv_ / points3d_cameara_start(2) + ppv_;

				float center_u_end = points3d_cameara_end(0) * fu_ / points3d_cameara_end(2) + ppu_;
				float center_v_end = points3d_cameara_end(1) * fv_ / points3d_cameara_end(2) + ppv_;

				//std::cout << center_u_start<<", "<< center_v_start << std::endl;
				//std::cout << center_u_end << ", " << center_v_end << std::endl;

				Eigen::Vector2f edge_vector = pre_2d_vector[index_edge_vector];
				index_edge_vector++;
				//std::cout << edge_vector << std::endl;

				Eigen::Vector2f point_pre = pre_2d_points[j];
				Eigen::Vector2f diff{ edge_vector(0) - (center_u_end - center_u_start), edge_vector(1) - (center_v_end - center_v_start) };
				//std::cout << diff << std::endl;
				float squared_error = diff.squaredNorm();
				float error = sqrtf(squared_error);

				// Calculate weight with Tukey norm
				float weight = 1.0f / variance_;
				if (error > std::numeric_limits<float>::min())
					weight = (TukeyNorm(error) / squared_error) / variance_;

				//cout<< weight <<endl;
				//error_count += fabs(error) * weight;
				// Calculate derivatives
				Eigen::MatrixXf  dx_dX(2, 3);
				dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z, -y * fv_ / z2;
				Eigen::MatrixXf dx_dtranslation{ dx_dX * body2camera_rotation_ };
				Eigen::MatrixXf dx_dtheta(2, 6);
				dx_dtheta << -dx_dtranslation * Vector2Skewsymmetric(points3d_body_start),
					dx_dtranslation;

				// Calculate gradient and hessian
				gradient_edge -= (weight * diff.transpose()) * dx_dtheta;
				hessian_edge.triangularView<Eigen::Lower>() -=
					(weight * dx_dtheta.transpose()) * dx_dtheta;
			}
		}
		
#endif

		//特征误差计算
		hessian_edge = hessian_edge.selfadjointView<Eigen::Lower>();
		error_count_ = error_count / data_lines_.size();  //当前线长 / distribution_length_
		//cv::Ptr<cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();
		//float dis = mysc->computeDistance(contourPts1, contourPts2);
		shape_cost_ = 0;
		match_ratio_ = float(match_num) / data_lines_.size();
		//cout << float(match_num) / data_lines_.size() << endl;
		//cout <<"edge: "<< error_count_ << endl;
		/*cv::imshow("img", img);
		cv::waitKey(0);*/
		
#endif
		//=========ransac剔除误差点===================
#if 0
		//将区域分为多个段，并进行ransac剔除误差点，然后再进行匹配
		vector <cv::Point> contourPts1_ransac;
		vector <cv::Point> contourPts2_ransac;
		vector<int> contour_index_vector; 
		vector <vector<cv::Point>> Candidate_points_datelines_;
		vector <vector<float>> Candidate_points_weight;
		cv::Mat img_show = camera_ptr_->image().clone();
		int contour_index_ = 0;
		for (int i = 0; i < data_lines_.size(); i++) {
			DataLine data_line = data_lines_[i];
			float max_prob = 0;
			float scale_minus_1_half_ = (fscale_ - 1.0f) / 2.0f;
			int u, v;
			int x_, y_;
			std::vector<std::pair<cv::Point, float>> prob_v;
			for (int i = 0; i < distribution_length_; ++i) {
				for (int j = 0; j < scale_; ++j) {
					if (std::fabs(data_line.normal_u) > std::fabs(data_line.normal_v)) {
						u = int(
							data_line.center_u +
							float(sgn(data_line.normal_u)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						v = int(data_line.center_v +
							(float(u) - data_line.center_u) *
							(data_line.normal_v / data_line.normal_u) +
							0.5f);
					}
					else {
						v = int(
							data_line.center_v +
							float(sgn(data_line.normal_v)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						u = int(data_line.center_u +
							(float(v) - data_line.center_v) *
							(data_line.normal_u / data_line.normal_v) +
							0.5f);
					}
					float prob = data_line.distribution[i];
					prob_v.push_back(std::make_pair(cv::Point(x_, y_), prob));
					if (prob > max_prob)
					{
						max_prob = prob;
						x_ = u;
						y_ = v;
					}
				}
			}
			
			sort(prob_v.begin(), prob_v.end(), [](std::pair<cv::Point, float> x, std::pair<cv::Point, float> y) {return x.second > y.second; });
			//prob_v[0].first
			//选择五个候选点
			vector<cv::Point> Candidate_points;
			vector<float> Candidate_w;
			for (int i = 0;i < 10;i++)
			{
				cv::Point p = cv::Point(prob_v[i].first.x, prob_v[i].first.y);
				float w = prob_v[i].second;
				Candidate_points.push_back(p);
				Candidate_w.push_back(w);
			}
			Candidate_points_datelines_.push_back(Candidate_points);
			Candidate_points_weight.push_back(Candidate_w);
			/*
			img_show.at<cv::Vec3b>(prob_v[0].first.y, prob_v[0].first.x) = cv::Vec3b(0, 0, 255);
			img_show.at<cv::Vec3b>(prob_v[1].first.y, prob_v[1].first.x) = cv::Vec3b(255, 0, 0);
			img_show.at<cv::Vec3b>(prob_v[2].first.y, prob_v[2].first.x) = cv::Vec3b(0, 255, 0);
			img_show.at<cv::Vec3b>(prob_v[3].first.y, prob_v[3].first.x) = cv::Vec3b(0, 255, 144);
			img_show.at<cv::Vec3b>(prob_v[4].first.y, prob_v[4].first.x) = cv::Vec3b(144, 255, 144);
			*/

			contourPts1_ransac.push_back(cv::Point(x_, y_));
			contourPts2_ransac.push_back(cv::Point(data_line.center_u, data_line.center_v));
			contour_index_vector.push_back(data_line.index_contour);
			/*if (data_line.index_contour == 0)
			{
				img_show.at<cv::Vec3b>(data_line.center_v, data_line.center_u) = cv::Vec3b(0, 0, 255);
			}
			if (data_line.index_contour == 1)
			{
				img_show.at<cv::Vec3b>(data_line.center_v, data_line.center_u) = cv::Vec3b(0, 255, 0);
			}
			if (data_line.index_contour == 2)
			{
				img_show.at<cv::Vec3b>(data_line.center_v, data_line.center_u) = cv::Vec3b(255, 0, 0);
			}*/
		}

		/*cv::namedWindow("img_show", 0);
		cv::imshow("img_show", img_show);
		cv::waitKey(0);*/

		//执行RANSAC过程
		std::vector<uchar> status;
		/****************基于连续性进行候选点选择---效果不好*****************/
#if 0
		int index = 0;
		std::vector<std::vector<cv::Point>> temp_Candidate_points;
		std::vector<cv::Point2f> temp_contours_point_project;
		std::vector<uchar> temp_status;
		for (int i = 0; i < contour_index_vector.size(); i++)
		{
			if (contour_index_vector[i] == index)
			{
				temp_Candidate_points.push_back(Candidate_points_datelines_[i]);
				temp_contours_point_project.push_back(contourPts2_ransac[i]);
			}
			if (contour_index_vector[i] != index || i == contour_index_vector.size() - 1)
			{
				//=======候选点计算=========
				vector<int> candidate_index;
				int current_point_index = 0;
				candidate_index.push_back(current_point_index);

				for (int n_p = 0;n_p < temp_Candidate_points.size() - 1; n_p++)
				{
					float min_energy = std::numeric_limits<float>::max();
					float temp_index = 0;
					for (int n_n = 0;n_n < temp_Candidate_points[n_p].size();n_n++)
					{
						cv::Point current_point = temp_Candidate_points[n_p][current_point_index];
						cv::Point next_point = temp_Candidate_points[n_p + 1][n_n];
						/*计算损失函数*/
						float energy_w_current = std::exp((1 - Candidate_points_weight[n_p][current_point_index])/10);
						float energy_w_next = std::exp((1 - Candidate_points_weight[n_p + 1][n_n])/10);
						float energy_dis = std::exp(-dist(current_point, next_point)/1000);
						//cout<< energy_w_current <<endl;
						//cout<< energy_w_next <<endl;
						//cout<< energy_dis <<endl;
						float energy_all = energy_dis;
						if (energy_all < min_energy)
						{
							min_energy = energy_all;
							//找到最小的点
							temp_index = n_n;
						}
					}
					current_point_index = temp_index;
					candidate_index.push_back(current_point_index);
				}
				cv::Mat img_show_candiate = camera_ptr_->image().clone();
				for (int n_p = 0; n_p < temp_Candidate_points.size(); n_p++)
				{
					img_show_candiate.at<cv::Vec3b>(temp_Candidate_points[n_p][candidate_index[n_p]].y,
						temp_Candidate_points[n_p][candidate_index[n_p]].x) = cv::Vec3b(0, 0, 255);
					/*cv::circle(img_show_candiate, cv::Point(temp_Candidate_points[n_p][candidate_index[n_p]].x,
						temp_Candidate_points[n_p][candidate_index[n_p]].y), 3, cv::Scalar(0, 0, 255), 1);*/

					/*img_show_candiate.at<cv::Vec3b>(temp_Candidate_points[n_p][0].y,
						temp_Candidate_points[n_p][0].x) = cv::Vec3b(0, 255, 0);
					img_show_candiate.at<cv::Vec3b>(temp_Candidate_points[n_p][1].y,
						temp_Candidate_points[n_p][1].x) = cv::Vec3b(0, 255, 0);
					img_show_candiate.at<cv::Vec3b>(temp_Candidate_points[n_p][2].y,
						temp_Candidate_points[n_p][2].x) = cv::Vec3b(0, 255, 0);
					img_show_candiate.at<cv::Vec3b>(temp_Candidate_points[n_p][3].y,
						temp_Candidate_points[n_p][3].x) = cv::Vec3b(0, 255, 0);
					img_show_candiate.at<cv::Vec3b>(temp_Candidate_points[n_p][4].y,
						temp_Candidate_points[n_p][4].x) = cv::Vec3b(0, 255, 0);*/

					img_show_candiate.at<cv::Vec3b>(temp_contours_point_project[n_p].y,
						temp_contours_point_project[n_p].x) = cv::Vec3b(255, 0, 0);
				}
				cv::namedWindow("img_show_candiate", 0);
				cv::imshow("img_show_candiate", img_show_candiate);
				cv::waitKey(0);
				//====下一个轮廓===
				//cout<< index <<endl;
				index = contour_index_vector[i];
				temp_Candidate_points.clear();
				temp_contours_point_project.clear();
				temp_status.clear();
				temp_Candidate_points.push_back(Candidate_points_datelines_[i]);
				temp_contours_point_project.push_back(contourPts2_ransac[i]);
			}
		}

#endif
		/******************随机的边缘分布时的ransac******************/
#if 0
		//分区的大小  20的设定才有效果提升
		int num_edge_ransac = 10;
		//每个分区的点数
		int step_ = data_lines_.size() / num_edge_ransac;
		for (int i = 0;i < num_edge_ransac;i++)
		{
			std::vector<cv::Point2f> temp_contours_point1;
			std::vector<cv::Point2f> temp_contours_point2;
			std::vector<uchar> temp_status;
			for (int j = 0;j < step_;j++)
			{
				temp_contours_point1.push_back(contourPts1_ransac[(i * step_) + j]);
				temp_contours_point2.push_back(contourPts2_ransac[(i * step_) + j]);
			}
			cv::findFundamentalMat(temp_contours_point2, temp_contours_point1, cv::FM_RANSAC, 3.0, 0.99, temp_status);

			//cv::Mat trans = cv::findHomography(temp_contours_point2, temp_contours_point1, cv::RANSAC, 3.0, temp_status, 100, 0.9);
			std::vector<cv::Point2f> result_pts;
			//cout<< trans <<endl;
			//cv::perspectiveTransform(temp_contours_point2, result_pts, trans);
			//cv::Mat img11 = camera_ptr_->image().clone();
			for (int i = 0;i < temp_contours_point2.size();i++)
			{
				if (temp_status[i] != 0)
				{
					//=====查找匹配点   红色=======
					//img11.at<cv::Vec3b>(temp_contours_point1[i].y, temp_contours_point1[i].x) = cv::Vec3b(0, 0, 255);
					//===模型投影点   蓝色====
					//img11.at<cv::Vec3b>(temp_contours_point2[i].y, temp_contours_point2[i].x) = cv::Vec3b(255, 0, 0);
					//cv::line(img11, temp_contours_point1[i], temp_contours_point2[i], cv::Scalar(255, 255, 255), 1);
				}
			}
			/*cv::namedWindow("img11", 0);
		    cv::imshow("img11", img11);
		    cv::waitKey(0);*/

			for (int k = 0; k < step_; k++)
			{
				status.push_back(temp_status[k]);
			}
		}
#endif
		/*************按照边缘轮廓的分类进行ransac，速度较慢**************/
#if 0
		std::vector<cv::Point2f> temp_contours_point1;
		std::vector<cv::Point2f> temp_contours_point2;
		std::vector<uchar> temp_status;
		int index = 0;
		cv::Mat img11 = camera_ptr_->image().clone();

		for (int i = 0;i < contour_index_vector.size();i++)
		{
			//====找同一个轮廓下的点====
			//img11.at<cv::Vec3b>(contourPts1_ransac[i].y, contourPts1_ransac[i].x) = cv::Vec3b(0, 0, 255);
			if (contour_index_vector[i] == index)
			{
				temp_contours_point1.push_back(contourPts1_ransac[i]);
				temp_contours_point2.push_back(contourPts2_ransac[i]);
			}
			if(contour_index_vector[i] != index || i == contour_index_vector.size() - 1)
			{	
				//=======当前轮廓的ransac=========
				/********点数太少时******/
				if (temp_contours_point2.size() < 10)
				{
					for (int j = 0;j < temp_contours_point2.size();j++)
					{
						status.push_back(0);
						//img11.at<cv::Vec3b>(temp_contours_point1[j].y, temp_contours_point1[j].x) = cv::Vec3b(0, 0, 255);
					}
				}
				/********点数>3******/
				else
				{
					//cv::findFundamentalMat(temp_contours_point2, temp_contours_point1, cv::FM_RANSAC, 1.0, 0.99, temp_status);
					cv::Mat trans = cv::findHomography(temp_contours_point2, temp_contours_point1, cv::RANSAC, 3.0, temp_status, 100, 0.99);
					
					if (temp_status.size() == 0)
					{
						for (int k = 0; k < temp_contours_point2.size(); k++)
						{
							status.push_back(0);
							//img11.at<cv::Vec3b>(temp_contours_point1[k].y, temp_contours_point1[k].x) = cv::Vec3b(0, 0, 255);
						}
					}
					else
					{
						std::vector<cv::Point2f> result_pts;
						//cv::perspectiveTransform(temp_contours_point2, result_pts, trans);
						for (int k = 0; k < temp_contours_point2.size(); k++)
						{
							status.push_back(temp_status[k]);
							if (temp_status[k] != 0)//temp_status[k] != 0
							{
								//=====查找匹配点   红色=======
								img11.at<cv::Vec3b>(temp_contours_point1[k].y, temp_contours_point1[k].x) = cv::Vec3b(0, 0, 255);
								//===模型投影点   蓝色====
								img11.at<cv::Vec3b>(temp_contours_point2[k].y, temp_contours_point2[k].x) = cv::Vec3b(255, 0, 0);
								//===转换后   绿色====
								//img11.at<cv::Vec3b>(result_pts[k].y, result_pts[k].x) = cv::Vec3b(0, 255, 0);
							}
						}
						/*cv::namedWindow("img11", 0);
						cv::imshow("img11", img11);
						cv::waitKey(0);*/
					}	
				}		
				//====下一个轮廓===
				//cout<< index <<endl;
				index = contour_index_vector[i];
				temp_contours_point1.clear();
				temp_contours_point2.clear();
				temp_status.clear();

				temp_contours_point1.push_back(contourPts1_ransac[i]);
				temp_contours_point2.push_back(contourPts2_ransac[i]);
			}
			
		}
#endif
		/**************点集聚类算法****************************/
#if 0
			//聚类
			// 生成随机点集
			cv::Mat data(500, 2, CV_32F);
			cv::RNG rng;
			rng.fill(data, cv::RNG::NORMAL, cv::Scalar(0, 0), cv::Scalar(0.1, 0.1));

			// 自适应选择簇数
			int max_k = 10;
			double best_silhouette = -1;
			int best_k = 2;
			for (int k = 2; k <= max_k; ++k)
			{
				cv::Mat labels, centers;
				kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

				// 计算轮廓系数
				cv::Mat silhouette;
				double avg_silhouette = 0;
				silhouette.create(data.rows, 1, CV_32F);
				for (int i = 0; i < data.rows; ++i)
				{
					float dist_in = 0, dist_out = FLT_MAX;
					int label = labels.at<int>(i);
					for (int j = 0; j < data.rows; ++j)
					{
						if (labels.at<int>(j) == label && i != j)
							dist_in += norm(data.row(i), data.row(j), cv::NORM_L2);
						else
						{
							int other_label = labels.at<int>(j);
							float dist = norm(data.row(i), centers.row(other_label), cv::NORM_L2);
							if (dist < dist_out)
								dist_out = dist;
						}
					}
					float silh = (dist_out - dist_in) / max(dist_in, dist_out);
					silhouette.at<float>(i) = silh;
					avg_silhouette += silh;
				}
				avg_silhouette /= data.rows;
				if (avg_silhouette > best_silhouette)
				{
					best_silhouette = avg_silhouette;
					best_k = k;
				}
			}

			// 使用自适应的簇数进行聚类
			cv::Mat labels, centers;
			kmeans(data, best_k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

			// 可视化聚类结果
			cv::Mat img = cv::Mat::zeros(500, 500, CV_8UC3);
			for (int i = 0; i < data.rows; ++i)
			{
				int label = labels.at<int>(i);
				cv::Point2f pt(data.at<float>(i, 0) * 200 + 250, data.at<float>(i, 1) * 200 + 250);
				cv::circle(img, pt, 2, 
					cv::Scalar(label == 0 ? 255 : label == 1 ? 128 : label == 2 ? 64 : 0, label == 0 ? 0 : label == 1 ? 128 : label == 2 ? 192 : 0, label == 0 ? 0 : label == 1 ? 128 : label == 2 ? 192 : 255), cv::FILLED);
			}
			cv::imshow("k-means clustering", img);
			cv::waitKey(0);
#endif			
		/***********基于点集相互之间向量的误差点剔除-----还没写完****************/
#if 0
			std::vector<cv::Point> temp_contours_point1;
			std::vector<cv::Point> temp_contours_point2;
			std::vector<uchar> temp_status;
			int index = 0;
			cv::Mat img11 = camera_ptr_->image().clone();

			for (int i = 0; i < contour_index_vector.size(); i++)
			{
				//====找同一个轮廓下的点====
				//img11.at<cv::Vec3b>(contourPts1_ransac[i].y, contourPts1_ransac[i].x) = cv::Vec3b(0, 0, 255);
				if (contour_index_vector[i] == index)
				{
					temp_contours_point1.push_back(contourPts1_ransac[i]);
					temp_contours_point2.push_back(contourPts2_ransac[i]);
				}
				if (contour_index_vector[i] != index || i == contour_index_vector.size() - 1)
				{
					//=======当前轮廓的ransac=========
					/********点数太少时******/
					if (temp_contours_point2.size() < 5)
					{
						for (int j = 0; j < temp_contours_point2.size(); j++)
						{
							status.push_back(0);
							//img11.at<cv::Vec3b>(temp_contours_point1[j].y, temp_contours_point1[j].x) = cv::Vec3b(0, 0, 255);
						}
					}
					/********点数>3******/
					else
					{
						/*剔除误差点*/
						try
						{
							for (int j = 0; j < temp_contours_point2.size(); j++)
							{

								status.push_back(1);
								img11.at<cv::Vec3b>(temp_contours_point1[j].y, temp_contours_point1[j].x) = cv::Vec3b(0, 0, 255);
							}
						}
						catch (cv::Exception& e)
						{

							const char* err_msg = e.what();
							cout<< err_msg <<endl;
						}
						cv::namedWindow("img11", 0);
						cv::imshow("img11", img11);
						cv::waitKey(0);
					}
					//====下一个轮廓===
					//cout<< index <<endl;
					index = contour_index_vector[i];
					temp_contours_point1.clear();
					temp_contours_point2.clear();
					temp_status.clear();

					temp_contours_point1.push_back(contourPts1_ransac[i]);
					temp_contours_point2.push_back(contourPts2_ransac[i]);
				}

			}
#endif
		//cv::findFundamentalMat(contourPts1_ransac, contourPts2_ransac, cv::FM_RANSAC, 3.0, 0.5, status);

		//cv::Mat img11 = camera_ptr_->image().clone();
		//for (int i = 0; i < contourPts1_ransac.size(); i++)
		//{
		//	//cv::circle(img11, contourPts1_ransac[i], 1, cv::Scalar(0, 255, 0), -1);
		//	//cv::circle(img11, contourPts2_ransac[i], 1, cv::Scalar(0, 0, 255), -1);
		//	if (status[i] != 0)
		//	{
		//		cv::circle(img11, contourPts1_ransac[i], 1, cv::Scalar(255, 255, 144), -1);
		//		cv::circle(img11, contourPts2_ransac[i], 1, cv::Scalar(250, 0, 120), -1);
		//	}
		//}
		//cv::namedWindow("img11", 0);
		//cv::imshow("img11", img11);
		//cv::waitKey(0);
#endif
//=================turky==============
#if 0
		int debug = 0;
		cv::Mat img;
		if(debug)
		  img = camera_ptr_->image().clone();
		vector <cv::Point> contourPts1;
		vector <cv::Point> contourPts2;
		float error_count = 0.0f;
		float nerr = 0.0f;
		//匹配的数量
		float match_num = 0;
		int index_point = 0;
		for (auto &data_line : data_lines_) {

			data_line.center_f_camera = body2camera_pose_ * data_line.center_f_body;
			//cout<< body2camera_pose_.matrix() <<endl;
			//估计的轮廓？？？
			float x = data_line.center_f_camera(0);
			float y = data_line.center_f_camera(1);
			float z = data_line.center_f_camera(2);
			float z2 = z * z;

			float max_prob = 0;
			float scale_minus_1_half_ = (fscale_ - 1.0f) / 2.0f;
			int u, v;
			int x_, y_;
			std::vector<std::pair<cv::Point, float>> prob_v;
			for (int i = 0; i < distribution_length_; ++i) {
				for (int j = 0; j < scale_; ++j) {
					if (std::fabs(data_line.normal_u) > std::fabs(data_line.normal_v)) {
						u = int(
							data_line.center_u +
							float(sgn(data_line.normal_u)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						v = int(data_line.center_v +
							(float(u) - data_line.center_u) *
							(data_line.normal_v / data_line.normal_u) +
							0.5f);
					}
					else {
						v = int(
							data_line.center_v +
							float(sgn(data_line.normal_v)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						u = int(data_line.center_u +
							(float(v) - data_line.center_v) *
							(data_line.normal_u / data_line.normal_v) +
							0.5f);
					}
					float prob = data_line.distribution[i];
					prob_v.push_back(std::make_pair(cv::Point(x_, y_), prob));
					if (prob > max_prob)
					{
						max_prob = prob;
						x_ = u;
						y_ = v;
					}
				}
			}
			//cout << max_prob << endl;
			//经验值
			//if (max_prob <= 0.15 || status[index_point] == 0) //  max_prob <= 0.15 || status[index_point] == 0
			//{
			//	index_point++;
			//	continue;
			//}
			if (max_prob >= 0.4)
			{
				match_num++;
			}
			index_point++;
			
			contourPts1.push_back(cv::Point(x_, y_));
			contourPts2.push_back(cv::Point(data_line.center_u, data_line.center_v));
			if (debug)
			{
				
				//cv::circle(img, cv::Point(data_line.center_u, data_line.center_v), 1, cv::Scalar(0, 0, 255), -1);
				//cv::circle(img, cv::Point(x_, y_), 1, cv::Scalar(0, 255, 255), -1);
				img.at<cv::Vec3b>(data_line.center_v, data_line.center_u) = cv::Vec3b(0, 0, 255);
				img.at<cv::Vec3b>(y_, x_) = cv::Vec3b(0, 255, 255);
				//cv::line(img, cv::Point(x_, y_), cv::Point(data_line.center_u, data_line.center_v),cv::Scalar(0, 255, 0), 1);
			
				//sort(prob_v.begin(), prob_v.end(), [](std::pair<cv::Point, float> x, std::pair<cv::Point, float> y) {return x.second > y.second; });

				//cv::circle(img, prob_v[0].first, 1, cv::Scalar(0, 255, 255), -1);
				//cv::circle(img, prob_v[1].first, 1, cv::Scalar(0, 255, 0), -1);
				//cv::circle(img, prob_v[2].first, 1, cv::Scalar(255, 0, 0), -1);

				//cv::line(img, prob_v[0].first, cv::Point(data_line.center_u, data_line.center_v), cv::Scalar(255, 255, 255), 1);
				//cv::line(img, prob_v[1].first, cv::Point(data_line.center_u, data_line.center_v), cv::Scalar(255, 255, 255), 1);
				//cv::line(img, prob_v[2].first, cv::Point(data_line.center_u, data_line.center_v), cv::Scalar(255, 255, 255), 1);
			}
			//cv::namedWindow("img",0);
			//cv::imshow("img", img);
			//cv::waitKey(0);

			Eigen::Vector2f diff{ (data_line.center_u - x_), (data_line.center_v - y_) };
			float squared_error = diff.squaredNorm();
			float error = sqrtf(squared_error);

			// Calculate weight with Tukey norm
			float weight = 1.0f / variance_;
			if (error > std::numeric_limits<float>::min())
				weight = (TukeyNorm(error) / squared_error) / variance_;

			//cout<< weight <<endl;
			error_count += fabs(error) * weight;
			nerr += weight;
			// Calculate derivatives
			Eigen::MatrixXf  dx_dX(2, 3);
			dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z, -y * fv_ / z2;
			Eigen::MatrixXf dx_dtranslation{ dx_dX * body2camera_rotation_ };
			Eigen::MatrixXf dx_dtheta(2, 6);
			dx_dtheta << -dx_dtranslation * Vector2Skewsymmetric(data_line.center_f_body),
				dx_dtranslation;

			// Calculate gradient and hessian
			gradient_edge -= (weight * diff.transpose()) * dx_dtheta;
			hessian_edge.triangularView<Eigen::Lower>() -=
				(weight * dx_dtheta.transpose()) * dx_dtheta;
		}
		//特征误差计算
		hessian_edge = hessian_edge.selfadjointView<Eigen::Lower>();
		error_count_ = error_count / (nerr * match_num);

		match_ratio_ = float(match_num) / data_lines_.size();
		//cout << "================" << endl;
		//cout << "edge error_count_: " << error_count_ << endl;
		//===============================
		//CalShape(contourPts1, contourPts2);
		
		//cout << "================" << endl;
		if (debug)
		{
			cv::Ptr<cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();	
			float dis1 = mysc->computeDistance(contourPts1, contourPts2);
			//cout << "shape Context distance" << dis1 << endl;
			//cv::Ptr<cv::HausdorffDistanceExtractor> mysd = cv::createHausdorffDistanceExtractor();
			//float dis2 = mysd->computeDistance(contourPts1, contourPts2);
			//cout << "Hausdorff distance" << dis2 << endl;
			cv::namedWindow("img",0);
			cv::imshow("img", img);
		    cv::waitKey(0);
		}
		
#endif
#if 0
		
		cv::Matx61f JT;
		cv::Matx66f wJTJ;
		JT = cv::Matx61f::zeros();
		wJTJ = cv::Matx66f::zeros();

		float* JT_float = JT.val;
		float* wJTJ_float = wJTJ.val;



		cv::Mat visualization_image{ camera_ptr_->image().clone() };
		//cv::Mat visualization_image(Canny_Mat_.clone());
		//cv::cvtColor(visualization_image, visualization_image, CV_GRAY2BGR);

		// Iterate over correspondence lines
		//计算每个搜索线的优化函数
		//std::ofstream outFile;
		//outFile.open("out.xyz", std::ofstream::out | std::ofstream::trunc);
		int u, v;
		float scale_minus_1_half_ = (fscale_ - 1.0f) / 2.0f;

		try {
			for (auto &data_line : data_lines_) {
				// Calculate point coordinates in camera frame
				data_line.center_f_camera = body2camera_pose_ * data_line.center_f_body;
				float x = data_line.center_f_camera(0);
				float y = data_line.center_f_camera(1);
				float z = data_line.center_f_camera(2);

				float fu_z = fu_ / z;
				float fv_z = fv_ / z;
				//
				float xfu_z = x * fu_z;
				float yfv_z = y * fv_z;

				/*std::ostringstream oss1;
				oss1 << x;
				std::string strx(oss1.str());
				std::ostringstream oss2;
				oss2 << y;
				std::string stry(oss2.str());
				std::ostringstream oss3;
				oss3 << z;
				std::string strz(oss3.str());
				outFile << strx + " " + stry + " " + strz + " " << std::endl;*/

				float zc2 = z * z;
				float J[6];
				//cout << data_line.normal_u<<"   "<< data_line.normal_v << endl;

				J[0] = data_line.normal_u * (-x*fu_*y / zc2) + data_line.normal_v * (-fv_ - y*y*fv_ / zc2);
				J[1] = data_line.normal_u * (fu_ + x*x*fu_ / zc2) + data_line.normal_v * (x*y*fv_ / zc2);
				J[2] = data_line.normal_u * (-fu_*y / z) + data_line.normal_v * (x*fv_ / z);

				//后三项与region方法一样
				J[3] = data_line.normal_u * (fu_ / z);
				J[4] = data_line.normal_v * (fv_ / z);
				J[5] = data_line.normal_u * (-x*fu_ / zc2) + data_line.normal_v * (-y*fv_ / zc2);

#if 0
				//计算分布最大的那个点
				float distribution = 0;
				float ex = 1000;
				float turkey_weiht = 30 * fscale_;
				int x_img, y_img;
				int index = 0;
				for (int i = 0; i < distribution_length_; ++i) {
					for (int j = 0; j < scale_; ++j) {
						float temp_distribution = data_line.distribution[index];
						index++;
						//cout << temp_distribution << endl;
						float turkey = 0;
						if (std::fabs(data_line.normal_u) > std::fabs(data_line.normal_v)) {
							u = int(
								data_line.center_u +
								float(sgn(data_line.normal_u)) *
								(fscale_ * (float(i) - distribution_length_minus_1_half_) +
									float(j) - scale_minus_1_half_) +
								0.5f);
							v = int(data_line.center_v +
								(float(u) - data_line.center_u) *
								(data_line.normal_v / data_line.normal_u) +
								0.5f);
						}
						else {
							v = int(
								data_line.center_v +
								float(sgn(data_line.normal_v)) *
								(fscale_ * (float(i) - distribution_length_minus_1_half_) +
									float(j) - scale_minus_1_half_) +
								0.5f);
							u = int(data_line.center_u +
								(float(v) - data_line.center_v) *
								(data_line.normal_u / data_line.normal_v) +
								0.5f);
						}
						//cout << "temp_distribution:" << temp_distribution << endl;
						//cv::circle(visualization_image, cv::Point(u, v), 1, cv::Scalar(0, 0, 255), -1);
						if (temp_distribution == 9999)
						{
							continue;
						}

						if (fabs(temp_distribution) <= turkey_weiht)
						{
							turkey = (1 - (temp_distribution / turkey_weiht)*(temp_distribution / turkey_weiht))
								*(1 - (temp_distribution / turkey_weiht)*(temp_distribution / turkey_weiht));

							//turkey = tukey_cost(temp_distribution, turkey_weiht);
						}
						else
							turkey = 0;

						if (v < 0 || v > image_height_minus_1_ || u < 0 || u > image_width_minus_1_) {
							continue;
						}

						if (distribution < turkey)
						{
							distribution = turkey;
							ex = temp_distribution;
							x_img = u;
							y_img = v;
						}
					}
				}
				if (y_img < 0 || y_img > image_height_minus_1_ || x_img < 0 || x_img > image_width_minus_1_)
				{
					continue;
				}

#endif
				
				//for (int n = 0; n < 6; n++) {
				//	//cout << "J:" << J[n] << endl;
				//	JT_float[n] += distribution * ex * J[n]; //distribution * ex * J[n]
				//	
				//}
				////cout <<"JT:"<< JT << endl;
				//float wJTJ[36];
				//for (int n = 0; n < 6; n++)
				//	for (int m = n; m < 6; m++) {
				//		wJTJ_float[n * 6 + m] += distribution * J[n] * J[m]; //distribution * J[n] * J[m]
				//	}
				float distribution = 0;
				float ex = 1000;
				//
				float turkey_weiht = 15 * fscale_;
				int x_img, y_img;
				int index = 0;
				float dltad = 0;
				for (int i = 0; i < distribution_length_; ++i) {
					for (int j = 0; j < scale_; ++j) {
						float temp_distribution = data_line.distribution[index];
						//cout << temp_distribution << endl;
						index++;
						//cout << temp_distribution << endl;
						float turkey = 0;
						if (std::fabs(data_line.normal_u) > std::fabs(data_line.normal_v)) {
							u = int(
								data_line.center_u +
								float(sgn(data_line.normal_u)) *
								(fscale_ * (float(i) - distribution_length_minus_1_half_) +
									float(j) - scale_minus_1_half_) +
								0.5f);
							v = int(data_line.center_v +
								(float(u) - data_line.center_u) *
								(data_line.normal_v / data_line.normal_u) +
								0.5f);
						}
						else {
							v = int(
								data_line.center_v +
								float(sgn(data_line.normal_v)) *
								(fscale_ * (float(i) - distribution_length_minus_1_half_) +
									float(j) - scale_minus_1_half_) +
								0.5f);
							u = int(data_line.center_u +
								(float(v) - data_line.center_v) *
								(data_line.normal_u / data_line.normal_v) +
								0.5f);
						}
											
						cv::circle(visualization_image, cv::Point(u, v), 1, cv::Scalar(0, 0, 255), -1);

						if (v < 0 || v > image_height_minus_1_ || u < 0 || u > image_width_minus_1_) {
							continue;
						}
						if (distribution < fabs(temp_distribution))
						{
							distribution = temp_distribution;
							x_img = u;
							y_img = v;						
							float ex = data_line.normal_u * (data_line.center_u - u) + data_line.normal_v * (data_line.center_v - v);
							
							dltad = ex;
						}
					}
				}
				if (y_img < 0 || y_img > image_height_minus_1_ || x_img < 0 || x_img > image_width_minus_1_)
				{
					continue;
				}

				//visualization_image.at<cv::Vec3b>(y_img, x_img) = cv::Vec3b(0, 255, 0);
				//cout << dltad << endl;
				cv::circle(visualization_image,cv::Point(x_img, y_img),1,cv::Scalar(0,255,0),-1);
				if (update_iteration < n_newton_iterations_)
				{
					dltad = dltad;
				}
				else
					dltad = dltad * learning_rate_;
				//cv::imwrite("visualization_image.jpg", visualization_image);

				//**************以下计算过程为收敛过程，比后一种估计的结果更好****************
				//ds对相机坐标下的点X的求导
				Eigen::RowVector3f ddelta_cs_dcenter{
					data_line.normal_u * fu_z,
					data_line.normal_v * fv_z,
					(-data_line.normal_u * xfu_z - data_line.normal_v * yfv_z) / z };
				//计算相机坐标系点X对thta的导数
				Eigen::RowVector3f ddelta_cs_dtranslation{ ddelta_cs_dcenter *
					body2camera_rotation_ };
				Eigen::Matrix<float, 1, 6> ddelta_cs_dtheta;
				ddelta_cs_dtheta << data_line.center_f_body.transpose().cross(ddelta_cs_dtranslation), ddelta_cs_dtranslation;

				gradient_edge += -distribution * dltad * ddelta_cs_dtheta.transpose();
				
				//ddelta_cs_dtheta /= data_line.standard_deviation;
				hessian_edge.triangularView<Eigen::Lower>() -=
					distribution * ddelta_cs_dtheta.transpose() * ddelta_cs_dtheta;
				//cout << "gradient_edge: "<<gradient_edge << endl;
				//============================================================
			}
			//特征误差计算
			hessian_edge = hessian_edge.selfadjointView<Eigen::Lower>();
			//outFile.close();
			if (0)
			{
				// Optimize and update pose
				Eigen::FullPivLU<Eigen::Matrix<float, 6, 6>> lu{ tikhonov_matrix_ - hessian_edge };
				if (lu.isInvertible()) {
					//theta为六维位姿变化量
					Eigen::Matrix<float, 6, 1> theta{ lu.solve(gradient_edge) };
					Transform3fA pose_variation{ Transform3fA::Identity() };
					//Vector2Skewsymmetric对称矩阵
					pose_variation.rotate(Vector2Skewsymmetric(theta.head<3>()).exp());
					pose_variation.translate(theta.tail<3>());
					body_ptr_->set_body2world_pose(body_ptr_->body2world_pose() *
						pose_variation);
					//cout << "pose_variation:"<<pose_variation.matrix() << endl;
				}
			}
			if (0)
			{
				//以下过程也可以收敛了，但需要注意的是这里算出来的矩阵为***左乘**====
				for (int i = 0; i < wJTJ.rows; i++)
					for (int j = i + 1; j < wJTJ.cols; j++) {
						wJTJ(j, i) = wJTJ(i, j);
					}
				Transform3fA pose_variation{ Transform3fA::Identity() };
				cv::Matx44f pose_temp = exp(-wJTJ.inv(cv::DECOMP_CHOLESKY)*JT);
				pose_variation.matrix() << pose_temp(0, 0), pose_temp(0, 1), pose_temp(0, 2), pose_temp(0, 3),
					pose_temp(1, 0), pose_temp(1, 1), pose_temp(1, 2), pose_temp(1, 3),
					pose_temp(2, 0), pose_temp(2, 1), pose_temp(2, 2), pose_temp(2, 3),
					pose_temp(3, 0), pose_temp(3, 1), pose_temp(3, 2), pose_temp(3, 3);
				//cout <<"pose_variation: "<<pose_variation.matrix() << endl;
				body_ptr_->set_body2world_pose(pose_variation * body_ptr_->body2world_pose());
			}





		}
		catch (cv::Exception& e)
		{
			const char* msg_e = e.what();

			cout << msg_e << endl;
		}
		cv::namedWindow("img_cal", 0);
		cv::imshow("img_cal", visualization_image);
		cv::waitKey(0);
#endif
		return true;
	}
	void EdgeModality::CalShape(std::vector<cv::Point> contourPts1, std::vector<cv::Point> contourPts2)
	{
		vector< vector<double> > chiStatistics;
		if (contourPts1.size() == 0)
		{
			return;
		}
		//    return 0;
		//获取两组点集的最多和最少数量
		int minsize = min(contourPts1.size(), contourPts2.size());
		int maxsize = max(contourPts1.size(), contourPts2.size());
		//采样率 = max/min
		int samplingRate = maxsize / minsize;
		//若最多的是第一组点，采样第一组点，否则采样第二组点
		if (contourPts1.size() == maxsize)
			contourPts1 = getSampledPoints(contourPts1, samplingRate);
		else
			contourPts2 = getSampledPoints(contourPts2, samplingRate);

		//std::cout << contourPts1.size() << " " << contourPts2.size() << std::endl;
		//进行采样后两两组点之间的数量相当

		pair<cv::Point, cv::Point> mm = getMinMax(contourPts1, contourPts2);
		//最大的x，y加上最大减去最小的0.1倍  作为size
		int sizex = mm.second.x + (int)((mm.second.x - mm.first.x) * 0.1);
		int sizey = mm.second.y + (int)((mm.second.y - mm.first.y) * 0.1);

		//	int **histogram1 = getHistogramFromContourPts(contourPts1);
		std::vector< std::vector<int>> histogram1 = getHistogramFromContourPts(contourPts1);
		int size1 = contourPts1.size();
		//std::cout << " size1 = " << size1 << std::endl;
		//	int **histogram2 = getHistogramFromContourPts(contourPts2);
		vector< vector<int> > histogram2 = getHistogramFromContourPts(contourPts2);
		int size2 = contourPts2.size();
		//std::cout << "size2 = " << size2 << std::endl;
		int size = max(size1, size2);

		for (int i = 0; i < size; ++i)
		{
			vector<double> tmp;
			for (int j = 0; j < size; ++j)
			{
				tmp.push_back(-1.0);
			}
			chiStatistics.push_back(tmp);
		}
		//计算代价矩阵chiStatistics
		getChiStatistic(chiStatistics, histogram1, size1, histogram2, size2);

		vector<double> u, v;
		vector<int> colsol, rowsol;
		for (int i = 0; i < size; ++i)
		{
			u.push_back(0);
			v.push_back(0);
			rowsol.push_back(0);
			colsol.push_back(0);
		}
		//struct timeval t1, t2;
		//gettimeofday(&t1, NULL);
		//==dim维度
		//=cost matrix代价矩阵
		//==行到列的解决方案
		//==列到行的解决方案
		//==对偶变量，行减少数
		//==对偶变量，列减少数
		//=========
		//====匈牙利算法会陷入死循环，分析原因为比较两个数的大小时存在问题，通过fabs(a-b)>1-e10解决
		//===========
		double cost = lap(size, chiStatistics, rowsol, colsol, u, v);
		//std::cout << "shape cost:" << cost / size1 << std::endl;

		shape_cost_ = cost / size1;
		//gettimeofday(&t2, NULL);
		//cout << "time taken :: " << (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0 << " s" << endl;
		double max_dist = -1, min_dist = 1000;

		cv::Mat resimg = cv::Mat::zeros(sizey, sizex, CV_8UC3);
		cv::Mat resimg2 = cv::Mat::zeros(sizey, sizex, CV_8UC3);
		int count = 0;
		for (int i = 0; i < size; i++)
		{
			int j = rowsol[i];
			if (i >= contourPts1.size() || j >= contourPts2.size())
				continue;

			cv::Point p1 = contourPts1[i];
			cv::Point p2 = contourPts2[j];

			double tmp_dist = dist(p1, p2);
			if (tmp_dist > max_dist)
			{
				max_dist = tmp_dist;
			}
			if (tmp_dist < min_dist)
			{
				min_dist = tmp_dist;
			}

			if (count % 1 == 0)
			{
				cv::circle(resimg, p1, 1, cv::Scalar(0,0,255), 1, 8);
				cv::circle(resimg, p2, 1, cv::Scalar(0, 255, 0), 1, 8);
				cv::line(resimg, p1, p2, cv::Scalar(255, 255, 255), 1, 8);
			}
			count = count + 1;
		}

		vector<int> link1, link2;
		for (int i = 0; i < size; ++i)
		{
			int j = rowsol[i];
			if (i >= contourPts1.size() || j >= contourPts2.size())
				continue;

			cv::Point p1 = contourPts1[i];
			cv::Point p2 = contourPts2[j];

			double tmp_dist = dist(p1, p2);

			//1.7        10
			//后面这个参数越小，匹配保留的点数越多
			//求两个匹配点的距离，若距离小于最小距离的两倍或小于最大距离的1/10，则认为合理
			//一般调后面的参数即可

			//if (tmp_dist < 1.7 * min_dist || tmp_dist < max_dist / 5)
			if (tmp_dist < 20.0)
			{
				link1.push_back(i);
				link2.push_back(j);
			}
		}
		int num = 0;
		for (int i = 0; i < link1.size(); ++i)
		{

			cv::Point p1 = contourPts1[link1[i]];
			cv::Point p2 = contourPts2[link2[i]];
			//p1.x = p1.x - 100;
			//p2.y = p2.y - 100;
			if (num % 1 == 0)
			{
				cv::circle(resimg2, p1, 1, cv::Scalar(0, 0, 255), 1, 8);
				cv::circle(resimg2, p2, 1, cv::Scalar(0, 255, 0), 1, 8);
				cv::line(resimg2, p1, p2, cv::Scalar(255, 255, 255), 1, 8);
			}
			num = num + 1;
		}
		float match_ratio = 0;
		match_ratio = float(link1.size()) / size;
		//cout<<"edge match_ratio： "<< match_ratio <<endl;
		cv::imshow("mapping_edge", resimg);
	    cv::imshow("map_clear_edge", resimg2);
		cv::waitKey(0);
	}

	cv::Matx44f  EdgeModality::exp(cv::Matx61f xi)
	{
		cv::Matx44f T = cv::Matx44f::eye();

		// rotational part of the twist coordinates (orientation)
		cv::Vec3f r = cv::Vec3f(xi(0, 0), xi(1, 0), xi(2, 0));

		// translational part of the twist coordinates (velocity)
		cv::Vec3f v = cv::Vec3f(xi(3, 0), xi(4, 0), xi(5, 0));

		// angle of the twist/rotation
		float theta = norm(r);

		// return the identity group element for theta == 0, as there is no motion
		if (abs(theta) < FLT_EPSILON)
		{
			return T;
		}
		else
		{
			// compute the rotation matrix as the matrix exponential of r
			cv::Matx33f R;
			cv::Rodrigues(r, R);

			// copy R to final pose
			T(0, 0) = R(0, 0); T(0, 1) = R(0, 1); T(0, 2) = R(0, 2);
			T(1, 0) = R(1, 0); T(1, 1) = R(1, 1); T(1, 2) = R(1, 2);
			T(2, 0) = R(2, 0); T(2, 1) = R(2, 1); T(2, 2) = R(2, 2);

			// compute the translation vector t
			cv::Matx33f I = cv::Matx33f::eye();
			cv::Vec3f w = r / theta;
			cv::Matx33f w_x = axiator(w);
			v /= theta;

			cv::Vec3f t = (I - R)*w_x*v + w*w.t()*v*theta;

			// copy t to final pose
			T(0, 3) = t[0];
			T(1, 3) = t[1];
			T(2, 3) = t[2];
		}

		return T;
	}
	cv::Matx33f EdgeModality::axiator(cv::Vec3f a)
	{
		float a1 = a[0];
		float a2 = a[1];
		float a3 = a[2];

		return cv::Matx33f(0, -a3, a2,
			a3, 0, -a1,
			-a2, a1, 0);
	}
	bool EdgeModality::VisualizePoseUpdate(int save_idx) {
		if (!IsSetup()) return false;

		if (visualize_points_pose_update_) {
			UpdateLineCentersWithCurrentPose();
			VisualizePointsCameraImage("camera_image_pose_update", save_idx);
		}
		if (visualize_points_histogram_image_pose_update_) {
			UpdateLineCentersWithCurrentPose();
			VisualizePointsHistogramImage("histogram_image_pose_update", save_idx);
		}
		return true;
	}

	bool EdgeModality::VisualizeResults(int save_idx) {
		if (!IsSetup()) return false;

		if (visualize_points_result_) {
			UpdateLineCentersWithCurrentPose();
			VisualizePointsCameraImage("camera_image_result", save_idx);
		}
		if (visualize_points_histogram_image_result_) {
			UpdateLineCentersWithCurrentPose();
			VisualizePointsHistogramImage("histogram_image_result", save_idx);
		}
		return true;
	}

	const std::string &EdgeModality::name() const { return name_; }

	std::shared_ptr<Body> EdgeModality::body_ptr() const { return body_ptr_; }

	std::shared_ptr<EdgeModel> EdgeModality::EdgeModel_ptr() const { return EdgeModel_ptr_; }

	std::shared_ptr<Camera> EdgeModality::camera_ptr() const {
		return camera_ptr_;
	}

	std::shared_ptr<OcclusionRenderer> EdgeModality::occlusion_renderer_ptr()
		const {
		return occlusion_renderer_ptr_;
	}

	bool EdgeModality::imshow_correspondence() const {
		return imshow_correspondence_;
	}

	bool EdgeModality::imshow_pose_update() const { return imshow_pose_update_; }

	bool EdgeModality::imshow_result() const { return imshow_result_; }

	bool EdgeModality::set_up() const { return set_up_; }

	void EdgeModality::PrecalculateFunctionLookup() {
		
		/*function_lookup_f_.resize(function_length_);
		function_lookup_b_.resize(function_length_);
		for (int i = 0; i < function_length_; ++i) {
			float x = float(i) - float(function_length_ - 1) / 2.0f;
			if (function_slope_ == 0.0f)
				function_lookup_f_[i] =
				0.5f - function_amplitude_ * ((0.0f < x) - (x < 0.0f));
			else
				function_lookup_f_[i] =
				0.5f - function_amplitude_ * std::tanh(x / (2.0f * function_slope_));
			function_lookup_b_[i] = 1.0f - function_lookup_f_[i];
		}*/

		//应用正太分布估计
		function_lookup_edge_.resize(function_length_);

		for (int i = 0; i < function_length_; ++i) {

			float x = float(i) - float(function_length_ - 1) / 2.0f;
			//function_lookup_edge_[i] = 1/(2.506628 * sigma_) * std::exp(-(pow(x - mu_,2)/(2*sigma_*sigma_)));
			function_lookup_edge_[i] = std::exp(-(pow(x - mu_, 2) / (2 * sigma_*sigma_)));
			//cout<< function_lookup_edge_[i] <<endl;
		}
		
	}

	void EdgeModality::PrecalculateDistributionVariables() {
		//为什么要相加？
		line_length_in_segments_ = function_length_ + distribution_length_ - 1;
		//line_length_in_segments_ = distribution_length_ + 1;
		distribution_length_minus_1_half_ =
			(float(distribution_length_) - 1.0f) / 2.0f;
		distribution_length_plus_1_half_ =
			(float(distribution_length_) + 1.0f) / 2.0f;
		float min_variance_laplace =
			1.0f / (2.0f * powf(std::atanhf(2.0f * function_amplitude_), 2.0f));
		float min_variance_gaussian = function_slope_;
		min_variance_ = std::max(min_variance_laplace, min_variance_gaussian);
		//cout<<"min_variance_: "<<min_variance_ <<endl;
	}

	void EdgeModality::PrecalculateHistogramBinVariables() {
		n_histogram_bins_squared_ = pow_int(n_histogram_bins_, 2);
		//std::cout << "n_histogram_bins_squared_:" << n_histogram_bins_squared_ << std::endl;
		n_histogram_bins_cubed_ = pow_int(n_histogram_bins_, 1);  //3
		//std::cout << "n_histogram_bins_cubed_:" << n_histogram_bins_cubed_ << std::endl;
		temp_histogram_f_.resize(n_histogram_bins_cubed_);
		temp_histogram_b_.resize(n_histogram_bins_cubed_);
		histogram_f_.resize(n_histogram_bins_cubed_);
		histogram_b_.resize(n_histogram_bins_cubed_);
	}

	void EdgeModality::SetImshowVariables() {
		imshow_correspondence_ = visualize_lines_correspondence_ ||
			(visualize_points_occlusion_mask_correspondence_ &&
				use_occlusion_handling_);
		imshow_pose_update_ = visualize_points_pose_update_ ||
			visualize_points_histogram_image_pose_update_;
		imshow_result_ =
			visualize_points_result_ || visualize_points_histogram_image_result_;
	}

	void EdgeModality::PrecalculateBodyVariables() {
		if (use_occlusion_handling_)
			encoded_occlusion_id_ = (uchar(1) << unsigned(body_ptr_->occlusion_id()));
	}

	void EdgeModality::PrecalculateCameraVariables() {
		fu_ = camera_ptr_->intrinsics().fu;
		fv_ = camera_ptr_->intrinsics().fv;
		ppu_ = camera_ptr_->intrinsics().ppu;
		ppv_ = camera_ptr_->intrinsics().ppv;
		image_width_minus_1_ = camera_ptr_->image().cols - 1;
		image_height_minus_1_ = camera_ptr_->image().rows - 1;
		image_width_minus_2_ = camera_ptr_->image().cols - 2;
		image_height_minus_2_ = camera_ptr_->image().rows - 2;
		//std::cout << "fu_,fv_:"<< fu_ <<" "<< fv_ << std::endl;
	}

	void EdgeModality::PrecalculateExtractEdge()
	{
		const cv::Mat &image{ camera_ptr_->image()};

		const cv::Mat &image_edge{ camera_ptr_->image_edge() };
		//cv::rectangle(canny_mat, rect_roi_, cv::Scalar(255), 1);
		//计算截取区域
		int expend_length = 40;  //40
		cv::Rect roi_new = cv::Rect(rect_roi_.x - expend_length, rect_roi_.y - expend_length, 
			rect_roi_.width+ 2 * expend_length, rect_roi_.height + 2 * expend_length);
		if (roi_new.x < 0)
		{
			roi_new.x = 0;
		}
		if (roi_new.y < 0)
		{
			roi_new.y = 0;
		}
		if (roi_new.x + roi_new.width > image.cols)
		{
			roi_new.width = image.cols - roi_new.x;
		}
		if (roi_new.y + roi_new.height > image.rows)
		{
			roi_new.height = image.rows - roi_new.y;
		}
		
		
		//cv::rectangle(canny_mat, roi_new, cv::Scalar(255), 1);
		cv::Mat canny_mat;
		
		cv::Mat ori_img = image(roi_new);
		rect_roi_ = roi_new;
		
		cv::Mat preEdge = image_edge;
		preEdge = preEdge(roi_new);
		if (preEdge.channels() == 3)
			cv::cvtColor(preEdge, preEdge, CV_BGR2GRAY);
		//normalize(preEdge, preEdge, 0.0, 255.0, cv::NORM_MINMAX);
		//std::cout << preEdge.channels() << std::endl;
		/*cv::imshow("ori_img", ori_img);
		cv::waitKey(0);*/
		cv::cvtColor(ori_img, canny_mat, CV_BGR2GRAY);
		//GMMCal(image);
		//GMMCal(ori_img);
		
		//Hog_img = canny_mat;
#if 0
		cv::Canny(canny_mat, canny_mat, 20, 40);   //50,100
		//对较小边缘进行过滤
		std::vector<std::vector<cv::Point2i>> contours1;
		cv::findContours(canny_mat, contours1, cv::RetrievalModes::RETR_LIST,
			cv::ContourApproximationModes::CHAIN_APPROX_NONE);
		contours1.erase(std::remove_if(begin(contours1), end(contours1),
			[](const std::vector<cv::Point2i> &contour) {
			return contour.size() < 100;
		}),	end(contours1));

		cv::Mat contours_filter = cv::Mat::zeros(canny_mat.size(), CV_8UC1);
		cv::drawContours(contours_filter, contours1, -1, cv::Scalar::all(255));
		if (contours_filter.channels() == 3)
		{
			cv::cvtColor(contours_filter, contours_filter, CV_BGR2GRAY);
		}

		Canny_Mat_ = contours_filter;  //canny_mat

		std::vector<std::vector<cv::Point2i>> contours;
		cv::findContours(canny_mat, contours, cv::RetrievalModes::RETR_LIST,
			cv::ContourApproximationModes::CHAIN_APPROX_NONE);

		//计算每个点的方向
		line_angle = cv::Mat::zeros(canny_mat.size(), CV_32FC1);
		for (auto contour : contours)
		{
			for (auto point : contour)
			{
				Eigen::Vector2f normal = { -float(contour.back().y - contour.front().y),float(contour.back().x - contour.front().x) };
				normal = normal.normalized();
				cv::Point2i p = point;
				float angle = atan2(normal.y(), normal.x());
				line_angle.at<float>(p) = angle;
			}
		}

#endif
		//进行sobel计算
#if 1
		//进行sobel计算
		cv::Mat sobel_dx;
		cv::Mat sobel_dy;
		cv::Sobel(canny_mat, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
		cv::Sobel(canny_mat, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_DEFAULT);

		/*cv::Mat sobel_p45;
		cv::Mat sobel_n45;
		cv::Sobel(canny_mat, sobel_p45, CV_32F, 1, 1, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
		cv::Sobel(canny_mat, sobel_n45, CV_32F, 1, -1, 3, 1.0, 0.0, cv::BORDER_DEFAULT);*/

		cv::Mat mag = cv::Mat::zeros(canny_mat.size(), CV_32FC1);
		cv::Mat ori = cv::Mat::zeros(canny_mat.size(), CV_32FC1);
		
		//==为true时表示以角度表示====
		cv::cartToPolar(sobel_dx, sobel_dy, mag, ori, false);//true
#if 0
		// 创建显示图像
		cv::Mat arrowImage;
		arrowImage = ori_img.clone();
		resize(arrowImage, arrowImage, cv::Size(ori.cols * 5, ori.rows * 5));
		cv::Mat mag_clone;
		mag_clone = mag.clone();
		cv::Mat mag_normal;
		//mag_clone.convertTo(mag_normal, CV_8U, 1.0);
		normalize(mag_clone, mag_clone, 0.0, 1.0, cv::NORM_MINMAX);
		//cv::imshow("arrowImage", arrowImage);
		// 绘制箭头
		double scale = 20.0;  // 箭头长度的缩放因子
		for (int y = 0; y < ori.rows; ++y)
		{
			for (int x = 0; x < ori.cols; ++x)
			{
				// 获取当前位置的梯度方向
				float dir = ori.at<float>(y, x);
				
				if (mag_clone.at<float>(y, x) > 0.03)
				{
					// 计算箭头终点坐标
					cv::Point2f arrowEnd(x * 5 + scale * cos(dir), y * 5 + scale * sin(dir));

					// 绘制箭头
					cv::arrowedLine(arrowImage, cv::Point(x * 5, y * 5), arrowEnd, cv::Scalar(0, 0, 255), 1);
					
				}
				
			}
		}
		// 显示图像
		cv::namedWindow("Arrow", 0);
		cv::imshow("Arrow", arrowImage);
		cv::waitKey(0);
#endif
		mag.convertTo(gray_roi_, CV_8U, 1.0);
		//cv::imshow("ori", ori);
		//cv::waitKey(0);
		//cv::resize(ori, ori, cv::Size(ori_img.rows * 3, ori_img.cols * 3));
		//cv::resize(mag, mag, cv::Size(ori_img.rows * 3, ori_img.cols * 3));

		cv::Mat ori_temp;
		//弧度制
		//phase(sobel_dx, sobel_dy, ori_temp);
		//计算幅值
		//magnitude(sobel_dx, sobel_dy, mag_);
		//取绝对值,因为白到黑的导数为正数，黑到白的导数为负数，负数会被自动置为0（黑色），所以只能看到左侧的边缘
		//sobel_dx = cv::abs(sobel_dx);
		//sobel_dy = cv::abs(sobel_dy);
		//cv::convertScaleAbs(sobel_dx, sobel_dx);
		//cv::convertScaleAbs(sobel_dy, sobel_dy);
		//cv::addWeighted(sobel_dx, 0.5, sobel_dy, 0.5, 0, mag);	
		//cout<< mag_ .type()<<endl;

		ori_ = ori;
		normalize(mag, mag, 0.0, 1.0, cv::NORM_MINMAX); //NORM_L2  NORM_MINMAX

		cv::Mat result;
		cv::Mat oriEdge;
		mag.convertTo(oriEdge, CV_8U, 255.0 , 0);
		//cv::imshow("oriEdge", oriEdge);
		//cv::imshow("preEdge", preEdge);
		//cv::waitKey(1);
		//cv::addWeighted(oriEdge, 0.9, preEdge, 0.1, 0, result);
		////cv::imshow("result", result);
		////cv::waitKey(0);
		//result.convertTo(result, CV_32FC1, 1.0 / 255);
		//normalize(result, mag, 0.0, 1.0, cv::NORM_MINMAX);

		pre_edge_ = preEdge;
		mag_temp = oriEdge;
		//cv::imshow("mag_temp", mag_temp);
		//cv::waitKey(0);
		//mag_ = mag;

#endif
#if 0
		/******************取通道中的梯度最大的*************************/
		cv::Mat smoothed;
		// Compute horizontal and vertical image derivatives on all color channels separately
		static const int KERNEL_SIZE = 3;
		// For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
		cv::GaussianBlur(ori_img, smoothed, cv::Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, cv::BORDER_REPLICATE);
		cv::Mat magnitude;
		magnitude.create(ori_img.size(), CV_32F);

		// Allocate temporary buffers
		cv::Size size = ori_img.size();
		cv::Mat sobel_3dx;              // per-channel horizontal derivative
		cv::Mat sobel_3dy;              // per-channel vertical derivative
		cv::Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
		cv::Mat sobel_dy(size, CV_32F); // maximum vertical derivative
		cv::Mat sobel_ag;               // final gradient orientation (unquantized)

		cv::Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
		cv::Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

		short *ptrx = (short *)sobel_3dx.data;
		short *ptry = (short *)sobel_3dy.data;
		float *ptr0x = (float *)sobel_dx.data;
		float *ptr0y = (float *)sobel_dy.data;
		float *ptrmg = (float *)magnitude.data;

		const int length1 = static_cast<const int>(sobel_3dx.step1());
		const int length2 = static_cast<const int>(sobel_3dy.step1());
		const int length3 = static_cast<const int>(sobel_dx.step1());
		const int length4 = static_cast<const int>(sobel_dy.step1());
		const int length5 = static_cast<const int>(magnitude.step1());
		const int length0 = sobel_3dy.cols * 3;

		for (int r = 0; r < sobel_3dy.rows; ++r)
		{
			int ind = 0;

			for (int i = 0; i < length0; i += 3)
			{
				// Use the gradient orientation of the channel whose magnitude is largest
				//使用幅值最大的通道的方向
				int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
				int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
				int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

				if (mag1 >= mag2 && mag1 >= mag3)
				{
					ptr0x[ind] = ptrx[i];
					ptr0y[ind] = ptry[i];
					ptrmg[ind] = sqrtf((float)mag1);
				}
				else if (mag2 >= mag1 && mag2 >= mag3)
				{
					ptr0x[ind] = ptrx[i + 1];
					ptr0y[ind] = ptry[i + 1];
					ptrmg[ind] = sqrtf((float)mag2);
				}
				else
				{
					ptr0x[ind] = ptrx[i + 2];
					ptr0y[ind] = ptry[i + 2];
					ptrmg[ind] = sqrtf((float)mag3);
				}
				++ind;
			}
			ptrx += length1;
			ptry += length2;
			ptr0x += length3;
			ptr0y += length4;
			ptrmg += length5;
		}

		// Calculate the final gradient orientations
		//false时为弧度
		phase(sobel_dx, sobel_dy, sobel_ag, false);  //true

		ori_ = sobel_ag;
		//bitwise_not(magnitude, magnitude);
		
		normalize(magnitude, magnitude, 0.0, 1.0, cv::NORM_MINMAX);

		//cv::namedWindow("magnitude", 0);
		//cv::imshow("magnitude", magnitude);
		//cv::waitKey(0);

		mag_ = magnitude;

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
		fld->detect(canny_mat, lines);
		vector<cv::line_descriptor::KeyLine> keylines;
		cv::Mat line_image = cv::Mat::zeros(canny_mat.size(), CV_8UC1);
		for (int i = 0;i < lines.size();i++)
		{
			cv::Vec4f line = lines[i];
			cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 1);
		}
#endif
#if 0
		cv::Mat input_img = cv::imread("normal_image_.png");
		cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
		ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
		ed->params.GradientThresholdValue = 38;
		ed->params.AnchorThresholdValue = 8;
		vector<cv::Vec6d> ellipses;
		vector<cv::Vec4f> lines;
		lines.clear();
		//you should call this before detectLines() and detectEllipses()
		cv::cvtColor(input_img, input_img, CV_BGR2GRAY);
		ed->detectEdges(input_img);
		// Detect lines
		ed->detectLines(lines);
		cv::Mat edge_image_ed = cv::Mat::zeros(canny_mat.size(), CV_8UC1);
		/*for (int i = 0;i < lines.size();i++)
		{
			cv::Vec4f line = lines[i];
			cv::line(edge_image_ed, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 1);
		}*/
		
		int class_counter = -1;
		vector<cv::line_descriptor::KeyLine> keylines;
		for (int k = 0; k < (int)lines.size(); k++)
		{
			cv::line_descriptor::KeyLine kl;
			cv::Vec4f extremes = lines[k];
			/* check data validity */
			checkLineExtremes(extremes, edge_image_ed.size());
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
			cv::LineIterator li(edge_image_ed, cv::Point2f(extremes[0], extremes[1]), cv::Point2f(extremes[2], extremes[3]));
			kl.numOfPixels = li.count;

			kl.angle = atan2((kl.endPointY - kl.startPointY), (kl.endPointX - kl.startPointX));
			kl.class_id = ++class_counter;
			kl.octave = 0;
			kl.size = (kl.endPointX - kl.startPointX) * (kl.endPointY - kl.startPointY);
			kl.response = kl.lineLength / max(edge_image_ed.cols, edge_image_ed.rows);
			kl.pt = cv::Point2f((kl.endPointX + kl.startPointX) / 2, (kl.endPointY + kl.startPointY) / 2);
			keylines.push_back(kl);
		}
		int lsdNFeatures = 30;
		//cout << "filter lines" << endl;
		if (keylines.size()>lsdNFeatures)
		{
			sort(keylines.begin(), keylines.end(), sort_lines_by_response());
			keylines.resize(lsdNFeatures);
			for (int i = 0; i<lsdNFeatures; i++)
				keylines[i].class_id = i;
		}
		cv::Mat combine_img = cv::imread("edge_image_ed111.jpg",1);
		for (int i = 0;i < keylines.size();i++)
		{
			cv::line_descriptor::KeyLine line = keylines[i];
			cv::line(edge_image_ed, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(255), 1);

			cv::line(combine_img, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(0,0,255), 1);
		}

		cv::imwrite("edge_image_ed.jpg", edge_image_ed);
		cv::imwrite("combine_img.jpg", combine_img);
#endif
	}

	void EdgeModality::PrecalculatePoseVariables() {
		body2camera_pose_ =
			camera_ptr_->world2camera_pose() * body_ptr_->body2world_pose();
		//cout<< "PrecalculatePoseVariables"<<body2camera_pose_ .matrix()<<endl;
		body2camera_rotation_ = body2camera_pose_.rotation().matrix();
		body2camera_rotation_xy_ = body2camera_rotation_.topRows<2>();
	}

	void EdgeModality::PrecalculateScaleDependentVariables(int corr_iteration) {
		if (corr_iteration < int(scales_.size()))
			scale_ = scales_[corr_iteration];
		else
			scale_ = 1;
		fscale_ = float(scale_);
		line_length_ = line_length_in_segments_ * scale_;
		line_length_minus_1_ = line_length_ - 1;
		line_length_minus_1_half_ = float(line_length_ - 1) * 0.5f;
		line_length_half_minus_1_ = float(line_length_) * 0.5f - 1.0f;

		float standard_deviation =
			LastValidValue(standard_deviations_, corr_iteration);
		variance_ = powf(standard_deviation, 2.0f);

	}

	void EdgeModality::AddLinePixelColorsToTempHistograms() {
		const cv::Mat &image{ camera_ptr_->image() };

		cv::Mat sobel_dx;
		cv::Mat sobel_dy;
		cv::Mat gray_img;
		cv::cvtColor(image, gray_img, CV_BGR2GRAY);
		cv::Sobel(gray_img, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
		cv::Sobel(gray_img, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_DEFAULT);

		cv::Mat mag_temp = cv::Mat::zeros(image.size(), CV_32FC1);
		cv::Mat ori_temp = cv::Mat::zeros(image.size(), CV_32FC1);
		//==为true时表示以角度表示====
		cv::cartToPolar(sobel_dx, sobel_dy, mag_temp, ori_temp, false);
		cv::Mat mag_8u_;
		mag_temp.convertTo(mag_8u_, CV_8U, 1.0);
		//cv::imshow("mag_8u_", mag_8u_);
		//cv::waitKey(0);
		//cv::Mat show_img = image.clone();
		const EdgeModel::TemplateView *template_view;
		EdgeModel_ptr_->GetClosestTemplateView(body2camera_pose_, &template_view);
		// Iterate over all points
		std::fill(begin(temp_histogram_f_), end(temp_histogram_f_), 0.0f);
		//std::fill(begin(temp_histogram_b_), end(temp_histogram_b_), 0.0f);
		for (auto data_point = begin(template_view->data_points);
			data_point != begin(template_view->data_points) + n_lines_;
			++data_point) {
			// Project point data in camera frame
			//center_f_body为模型上的点
			Eigen::Vector3f center_f_camera{ body2camera_pose_ *
				data_point->center_f_body };
			//将其转换到图像
			Eigen::Vector2f center{
				center_f_camera(0) * fu_ / center_f_camera(2) + ppu_,
				center_f_camera(1) * fv_ / center_f_camera(2) + ppv_ };
			//将其转换到图像
			Eigen::Vector2f normal{
				(body2camera_rotation_xy_ * data_point->normal_f_body).normalized() };
			float foreground_distance =
				data_point->foreground_distance * fu_ / center_f_camera(2);
			float background_distance =
				data_point->background_distance * fu_ / center_f_camera(2);

			// Iterate over foreground pixels
			float u = center(0) - normal(0) * considered_line_length_/2;
			float v = center(1) - normal(1) * considered_line_length_/2;
			int n_iteration =
				int(considered_line_length_);
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				if (int(mag_8u_.at<uchar>(int(v), int(u))) > 20)
				{
					AddPixelColorToHistogram(mag_8u_.at<uchar>(int(v), int(u)),
						&temp_histogram_f_);
				}			
				u += normal(0);
				v += normal(1);
				//show_img.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(255, 0, 0);
			}
		}

		/*cv::namedWindow("show_img",0);
		cv::imshow("show_img", show_img);
		cv::waitKey(0);*/
	}
	/***在跟踪成功的基础上，构建边缘匹配成功的局部纹理表征，为相机位姿预测做准备***/
	void EdgeModality::AddEdgePixelGradientForCorrelation() {
		const cv::Mat &image{ camera_ptr_->image() };
		const EdgeModel::TemplateView *template_view;
		EdgeModel_ptr_->GetClosestTemplateView(body2camera_pose_, &template_view);
		//cv::Mat shoe_img = image.clone();
		//提取边缘图像
		//cv::Mat sobel_dx;
		//cv::Mat sobel_dy;
		//cv::Mat gray_img;
		//cv::cvtColor(image, gray_img, CV_BGR2GRAY);
		//cv::Sobel(gray_img, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
		//cv::Sobel(gray_img, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_DEFAULT);

		//cv::Mat mag_temp = cv::Mat::zeros(image.size(), CV_32FC1);
		//cv::Mat ori_temp = cv::Mat::zeros(image.size(), CV_32FC1);
		////==为true时表示以角度表示====
		//cv::cartToPolar(sobel_dx, sobel_dy, mag_temp, ori_temp, false);
		//normalize(mag_temp, mag_temp, 0.0, 255.0, cv::NORM_MINMAX);
		
		//cv::imwrite("mag_temp.jpg", mag_temp);
		std::vector<std::vector<cv::Vec3b>> SearchLinesColor_;
		// Iterate over all points
		SearchLinesGradient_.clear();
		lastFrame_CenterPoints_.clear();
		lastFrame_CenterNormal_.clear();

		cv::Mat temp_img_search_lines = cv::Mat::zeros(cv::Size(n_lines_, int(considered_line_length_ncc_ + 0.5)), CV_8UC3);

		int count_lines = 0;
		//points_lk_.clear();
		for (auto data_point = begin(template_view->data_points);
			data_point != begin(template_view->data_points) + n_lines_;
			++data_point) 
		{
#if 1
			//cout<<"iiii"<<endl;
			// Project point data in camera frame
			//center_f_body为模型上的点
			Eigen::Vector3f center_f_camera{ body2camera_pose_ *
				data_point->center_f_body };
			/*模型上的点*/
			lastFrame_CenterPoints_.push_back(data_point->center_f_body);
			//将其转换到图像
			Eigen::Vector2f center{
				center_f_camera(0) * fu_ / center_f_camera(2) + ppu_,
				center_f_camera(1) * fv_ / center_f_camera(2) + ppv_ };

			/*光流*/
			//points_lk_.push_back(cv::Point2f(center[0], center[1]));
			//Last_Frame = image.clone();

			//将其转换到图像
			Eigen::Vector2f normal{
				(body2camera_rotation_xy_ * data_point->normal_f_body).normalized() };
			lastFrame_CenterNormal_.push_back(normal);
			float normal_v = normal(1);
			float normal_u = normal(0);
			float center_u = center(0);
			float center_v = center(1);

			if (std::fabs(normal_v) < std::fabs(normal_u)) {
				// Calculate step and starting position
				float v_step = normal_v / normal_u;
				// Notice: u = int(center_u - (line_length / 2 - 0.5) + 0.5)
				int u = int(center_u - (considered_line_length_ncc_ / 2 - 1));
				int u_end = u + considered_line_length_ncc_ - 1;
				float v_f = center_v + v_step * (float(u) - center_u) + 0.5f;
				float v_f_end = v_f + v_step * float(considered_line_length_ncc_);

				// Iterate over all pixels of line and calculate probabilities
				if (normal_u > 0) {
					int jj = 0;
					for (; u <= u_end; ++u, v_f += v_step) {
						// Check if line is on image (margin of 1 for rounding errors of v_f_end)
						if (u < 0 || int(v_f) < 0 || int(u) > image_width_minus_1_ || int(v_f) > image_height_minus_1_)
							temp_img_search_lines.at<cv::Vec3b>(jj, count_lines) = cv::Vec3b(0, 0, 0);
						else
							temp_img_search_lines.at<cv::Vec3b>(jj, count_lines) = image.at<cv::Vec3b>(int(v_f), u);
						//cout << jj << endl;
						jj++;
					}
				}
				else {
					int jj = considered_line_length_ncc_ - 1;
					for (; u <= u_end; ++u, v_f += v_step) {
						if (u < 0 || int(v_f) < 0 || int(u) > image_width_minus_1_ || int(v_f) > image_height_minus_1_)
						{
							
							temp_img_search_lines.at<cv::Vec3b>(jj, count_lines) = cv::Vec3b(0, 0, 0);
						}		
						else
							temp_img_search_lines.at<cv::Vec3b>(jj, count_lines) = image.at<cv::Vec3b>(int(v_f), u);
						//cout << jj << endl;
						jj--;
					}
				}
			}
			else
			{
				// Calculate step and starting position
				float u_step = normal_u / normal_v;
				// Notice: v = int(center_v - (line_length / 2 - 0.5) + 0.5)
				int v = int(center_v - (considered_line_length_ncc_ / 2 - 1));
				int v_end = v + considered_line_length_ncc_ - 1;
				float u_f = center_u + u_step * (float(v) - center_v) + 0.5f;
				float u_f_end = u_f + u_step * float(considered_line_length_ncc_);

				// Iterate over all pixels of line and calculate probabilities
				if (normal_v > 0) {
					int jj = 0;
					for (; v <= v_end; ++v, u_f += u_step) {
						// Check if line is on image (margin of 1 for rounding errors of u_f_end)
						if (v < 0 || int(u_f) < 0 ||
							int(u_f) > image_width_minus_1_ || int(v) > image_height_minus_1_) {
							temp_img_search_lines.at<cv::Vec3b>(jj, count_lines) = cv::Vec3b(0, 0, 0);;
						}
						else
							temp_img_search_lines.at<cv::Vec3b>(jj, count_lines) = image.at<cv::Vec3b>(v, int(u_f));
						jj++;
					}
				}
				else {
					int jj = considered_line_length_ncc_ - 1;
					for (; v <= v_end; ++v, u_f += u_step) {
						if (v < 0 || int(u_f) < 0 ||
							int(u_f) > image_width_minus_1_ || int(v) > image_height_minus_1_) {
							//cout << int(normal_u) << endl;
							//cout << int(normal_v) << endl;
							temp_img_search_lines.at<cv::Vec3b>(jj, count_lines) = cv::Vec3b(0, 0, 0);
						}
						else
							temp_img_search_lines.at<cv::Vec3b>(jj, count_lines) = image.at<cv::Vec3b>(v, int(u_f));
						jj--;
					}
				}
			}
			count_lines++;
#endif
			/*float u = center(0) - normal(0) * considered_line_length_ / 2 + 0.5f;
			float v = center(1) - normal(1) * considered_line_length_ / 2 + 0.5f;
			std::vector<cv::Vec3b> temp_search_gradient;
			std::vector<cv::Vec3b> temp_search_color;*/

			//int n_iteration = int(considered_line_length_ + 0.5f);
			//for (int i = 0; i < n_iteration; ++i) {
			//	if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
			//		int(v) > image_height_minus_1_)
			//	{
			//		temp_search_gradient.push_back(0);
			//		break;
			//	}			
			//	temp_search_gradient.push_back(image.at<cv::Vec3b>(int(v), int(u)));
			//	//temp_search_color.push_back(image.at<cv::Vec3b>(int(v), int(u)));
			//	u += normal(0);
			//	v += normal(1);
			//	//shoe_img.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(0,0,255);
			//	/*cv::namedWindow("show_search_lines", 0);
			//	cv::imshow("show_search_lines", shoe_img);
			//	cv::waitKey(0);*/
			//}
			//SearchLinesGradient_.push_back(temp_search_gradient);
			//SearchLinesColor_.push_back(temp_search_color);
		}
		Lasr_Frame_Search_Line_img = temp_img_search_lines;
		/*cv::imshow("Lasr_Frame_Search_Line_img", Lasr_Frame_Search_Line_img);
		cv::imwrite("Lasr_Frame_Search_Line_img.jpg", Lasr_Frame_Search_Line_img);
		cv::waitKey(0);*/
		/*cv::Mat bundle_img = cv::Mat::zeros(cv::Size(SearchLinesGradient_.size(), SearchLinesGradient_[0].size()), CV_8UC3);
		cv::Mat bundle_img_color = cv::Mat::zeros(cv::Size(SearchLinesGradient_.size(), SearchLinesGradient_[0].size()), CV_8UC3);
		for (int i = 0;i < SearchLinesGradient_.size();i++)
		{
			for (int j = 0; j < SearchLinesGradient_[i].size();j++)
			{
				bundle_img.at<cv::Vec3b>(j,i) = SearchLinesGradient_[i][j];
				bundle_img_color.at<cv::Vec3b>(j, i) = SearchLinesColor_[i][j];
			}
		}
		cv::namedWindow("show_bundle", 0);
		cv::imshow("show_bundle", bundle_img);
		cv::imwrite("bundle_img.jpg", bundle_img);
		cv::imwrite("bundle_img_color.jpg", bundle_img_color);
		cv::waitKey(0);*/
	}

	void EdgeModality::AddPixelColorToHistogram(
		const uchar &pixel_color,
		std::vector<float> *enlarged_histogram) const {
		//cout<< int(pixel_color[0]) <<endl;
		(*enlarged_histogram)[(pixel_color >> histogram_bitshift_)] += 1.0f;
	}

	bool EdgeModality::CalculateHistogram(
		float learning_rate, const std::vector<float> &temp_histogram,
		std::vector<float> *histogram) {
		// Calculate sum for normalization
		float sum = 0.0f;
#ifndef _DEBUG
#pragma omp simd
#endif
		for (int i = 0; i < n_histogram_bins_cubed_; i++) {
			sum += temp_histogram[i];
		}
		if (!sum) return false;

		// Calculate histogram
		float complement_learning_rate = 1.0f - learning_rate;
		float learning_rate_divide_sum = learning_rate / sum;
#ifndef _DEBUG
#pragma omp simd
#endif
		for (int i = 0; i < n_histogram_bins_cubed_; i++) {
			(*histogram)[i] *= complement_learning_rate;
			(*histogram)[i] += temp_histogram[i] * learning_rate_divide_sum;
		}
		return true;
	}

	void EdgeModality::CalculateBasicLineData(const EdgeModel::PointData &data_point,
		DataLine *data_line) const {
		try{
			Eigen::Vector3f center_f_camera{ body2camera_pose_ * data_point.center_f_body };
			//cout << body2camera_pose_.matrix() << endl;
			Eigen::Vector2f normal_f_camera{
				(body2camera_rotation_xy_ * data_point.normal_f_body).normalized() };

			data_line->center_f_body = data_point.center_f_body;
			data_line->center_f_camera = center_f_camera;
			data_line->center_u = center_f_camera(0) * fu_ / center_f_camera(2) + ppu_;
			data_line->center_v = center_f_camera(1) * fv_ / center_f_camera(2) + ppv_;
			data_line->normal_u = normal_f_camera(0);
			data_line->normal_v = normal_f_camera(1);
			//cout<< atan2(data_line->normal_v, data_line->normal_u) * 180 / CV_PI<<endl;
			data_line->continuous_distance =
				std::min(data_point.background_distance, data_point.foreground_distance) *
				fu_ / (center_f_camera(2) * fscale_);

			data_line->index_contour = data_point.index_contour;
			//有错误
			
			//cout << data_point.vimg_desc_body << endl;
			//Eigen::VectorXf desc_temp{ data_point.vimg_desc_body };
			//cout << data_point.vimg_desc_body << endl;
			//data_line->vimg_desc = data_point.vimg_desc_body;
			/*Eigen::VectorXf desc_temp(36);
			desc_temp(0) = data_point.vimg_desc_body1(0);
			desc_temp(1) = data_point.vimg_desc_body1(1);
			desc_temp(2) = data_point.vimg_desc_body1(2);

			desc_temp(3) = data_point.vimg_desc_body2(0);
			desc_temp(4) = data_point.vimg_desc_body2(1);
			desc_temp(5) = data_point.vimg_desc_body2(2);

			desc_temp(6) = data_point.vimg_desc_body3(0);
			desc_temp(7) = data_point.vimg_desc_body3(1);
			desc_temp(8) = data_point.vimg_desc_body3(2);

			desc_temp(9) = data_point.vimg_desc_body4(0);
			desc_temp(10) = data_point.vimg_desc_body4(1);
			desc_temp(11) = data_point.vimg_desc_body4(2);

			desc_temp(12) = data_point.vimg_desc_body5(0);
			desc_temp(13) = data_point.vimg_desc_body5(1);
			desc_temp(14) = data_point.vimg_desc_body5(2);

			desc_temp(15) = data_point.vimg_desc_body6(0);
			desc_temp(16) = data_point.vimg_desc_body6(1);
			desc_temp(17) = data_point.vimg_desc_body6(2);

			desc_temp(18) = data_point.vimg_desc_body7(0);
			desc_temp(19) = data_point.vimg_desc_body7(1);
			desc_temp(20) = data_point.vimg_desc_body7(2);

			desc_temp(21) = data_point.vimg_desc_body8(0);
			desc_temp(22) = data_point.vimg_desc_body8(1);
			desc_temp(23) = data_point.vimg_desc_body8(2);

			desc_temp(24) = data_point.vimg_desc_body9(0);
			desc_temp(25) = data_point.vimg_desc_body9(1);
			desc_temp(26) = data_point.vimg_desc_body9(2);

			desc_temp(27) = data_point.vimg_desc_body10(0);
			desc_temp(28) = data_point.vimg_desc_body10(1);
			desc_temp(29) = data_point.vimg_desc_body10(2);

			desc_temp(30) = data_point.vimg_desc_body11(0);
			desc_temp(31) = data_point.vimg_desc_body11(1);
			desc_temp(32) = data_point.vimg_desc_body11(2);

			desc_temp(33) = data_point.vimg_desc_body12(0);
			desc_temp(34) = data_point.vimg_desc_body12(1);
			desc_temp(35) = data_point.vimg_desc_body12(2);
			
			data_line->vimg_desc = desc_temp;*/
			//cout<< data_line->vimg_desc <<endl;
		}	
		catch (cv::Exception& e)
		{
			const char* msg_e = e.what();

			cout << msg_e << endl;
		}
	}

	bool EdgeModality::IsLineValid(float u, float v,
		float continuous_distance) const {
		// Check if continuous distance is long enough
		//这个值怎么来的？？？
		//if (continuous_distance < min_continuous_distance_) return false;

		// Check if image coordinate is on image
		int i_u = int(u + 0.5f);
		int i_v = int(v + 0.5f);
		if (i_u < 0 || i_u > image_width_minus_1_ || i_v < 0 ||
			i_v > image_height_minus_1_)
			return false;

		// Check if line center is on mask
		if (use_occlusion_handling_) {
			//std::cout << int(occlusion_renderer_ptr_->GetValue(i_v, i_u)) << std::endl;
			//std::cout << int(encoded_occlusion_id_) <<std::endl;
			//std::cout << int(occlusion_renderer_ptr_->GetValue(i_v, i_u) & encoded_occlusion_id_) << std::endl;
			const cv::Mat &image{ camera_ptr_->image_mask_dilate() };
			/*cv::imshow("image", image);
			cv::waitKey(0);*/
			float mask_prob = image.at<float>(i_v, i_u);
			//std::cout << mask_prob <<std::endl;
			bool flag_occ = false;
			if (mask_prob > 0.60)
				flag_occ = true;
			//return occlusion_renderer_ptr_->GetValue(i_v, i_u) & encoded_occlusion_id_;

			return (occlusion_renderer_ptr_->GetValue(i_v, i_u) & encoded_occlusion_id_) && flag_occ;
		}
		return true;
	}
	bool EdgeModality::CalculateSegmentProbabilities_edge(
		float center_u, float center_v, float normal_u, float normal_v,
		std::vector<float> *segment_probabilities_f,
		std::vector<float> *segment_probabilities_b,
		float *normal_component_to_scale, float *delta_r, std::vector<float> *distribution, float *mean,
		float *standard_deviation, float *variance, Eigen::VectorXf vimg_desc) {

		const cv::Mat &image{ camera_ptr_->image_mask() };
		
		//cv::Mat region_result_sement_img = cv::imread("visualization_image.jpg",0);
		//cv::imshow("image", image);
		//cv::waitKey(0);
		//clone会非常慢
		int debug = 0;
		cv::Mat show_img;
		if (debug) //debug
		{
			show_img = image.clone();
		}		
		/*当前搜索线图像*/
		//cout<< line_length_minus_1_ <<endl;
		cv::Mat current_img = cv::Mat::zeros(cv::Size(1, line_length_minus_1_ + 1), CV_8UC3);

		// Select case if line is more horizontal or vertical
		if (std::fabs(normal_v) < std::fabs(normal_u)) {
			// Calculate step and starting position
			float v_step = normal_v / normal_u;
			// Notice: u = int(center_u - (line_length / 2 - 0.5) + 0.5)
			int u = int(center_u - line_length_half_minus_1_);
			int u_end = u + line_length_minus_1_;
			float v_f = center_v + v_step * (float(u) - center_u) + 0.5f;
			float v_f_end = v_f + v_step * float(line_length_minus_1_);

			// Check if line is on image (margin of 1 for rounding errors of v_f_end)
			if (u < 0 || u_end > image_width_minus_1_ || int(v_f) < 0 ||
				int(v_f) > image_height_minus_1_ || int(v_f_end) < 1 ||
				int(v_f_end) > image_height_minus_2_) {
				return false;
			}

			// Iterate over all pixels of line and calculate probabilities
			if (normal_u > 0) {

				float *segment_probability_f = segment_probabilities_f->data();

				//float *segment_probability_b = segment_probabilities_b->data();
				*segment_probability_f = 1.0f;
				//*segment_probability_b = 1.0f;
				int col_index = 0;
				int segment_idx = 0;
				int jj = 0;
				for (; u <= u_end; ++u, v_f += v_step, segment_idx++) {
					if (segment_idx == scale_) {
						*(++segment_probability_f) = 1.0f;
						//*(++segment_probability_b) = 1.0f;
						segment_idx = 0;
					}					
					if(flag_use_correlation_)
						current_img.at<cv::Vec3b>(jj, 0) = image.at<cv::Vec3b>(int(v_f), u);
					jj++;
					if (flag_use_histogram_edge)
					{
						MultiplyPixelColorProbability_edge(gray_roi_.at<uchar>(int(v_f) - rect_roi_.y, u - rect_roi_.x),
							segment_probability_f);
						//cout<< *segment_probability_f <<endl;
					}
					//计算每个点展开一个区域，得到描述符
#if 0
					
					if (scale_ == 1)
					{
						double dir = atan2(normal_v, normal_u);
						cv::HOGDescriptor detector = cv::HOGDescriptor(cv::Size(5, 5), cv::Size(4, 4), cv::Size(1, 1), cv::Size(4, 4), 9);
						vector<float> descriptions;
						cv::Mat roi_img;
						extractEdgeDescriptor(roi_img, Hog_img, cv::Point(u, int(v_f)), dir, 2);
						/*cv::imshow("roi_img", roi_img);
						cv::waitKey(0);*/
						detector.compute(roi_img, descriptions);
						Eigen::VectorXf real_img_desc = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(descriptions.data(), descriptions.size());

						float innerMul = 0.0;
						float amp1 = 0.0;
						float amp2 = 0.0;
						for (int i = 0; i < vimg_desc.size(); i++)
						{
							innerMul += vimg_desc(i) * real_img_desc(i);
							amp1 += pow(vimg_desc(i), 2);
							amp2 += pow(real_img_desc(i), 2);
						}
						ncc = innerMul / (sqrt(amp1) * sqrt(amp2) + 0.00001);
					}
					
#endif
					//cout<< ncc <<endl;
					/*cv::circle(show_img, cv::Point(u, int(v_f)), 2, 255, -1);;
					cv::imshow("Hog_img", show_img);
					cv::waitKey(0);*/
					//cout<< vimg_desc <<endl;

					//=================
					//高斯混合值的计算
					/*float gmm_score = 1.0f;
					if (scale_ == 1)
					{
						int value = int(fgMask.at<uchar>(col_index,dateline_index));
						
						if (value > 0)
						{
							cv::circle(show_gmm, cv::Point(u, int(v_f)), 2, cv::Scalar(0, 0, 255));
							gmm_score = 0.8f;
						}
						else
						{
							gmm_score = 0.1f;
						}
					}
					col_index++;*/
					//=================
					float mag = 1.0;
					mag = mag_.at<float>(int(v_f) - rect_roi_.y, u - rect_roi_.x);
					//cout<< mag <<endl;
					//float mag = mag_.at<float>(int(v_f) - rect_roi_.y, u - rect_roi_.x);
					//int region_sement_value = region_result_sement_img.at<uchar>(int(v_f), u);
					//cout<< region_sement_value <<endl;
					//mag = 1.0 - sqrt(powf(mag - 1, 2));
					/*show_img.at<cv::Vec3b>(int(v_f), u) = cv::Vec3b(0, 0, 255);
					cv::namedWindow("show_img", 0);
					cv::imshow("show_img", show_img);
					cv::waitKey(0);*/

					mag = powf(mag, 1);
					
					//边缘提取的响应值
					if (mag > th_mag)  //0.01
					{
						//cv::circle(show_img, cv::Point(u,int(v_f)), 30, cv::Scalar(0, 255, 255), -1);						
						float angle_real = ori_.at <float> (int(v_f) - rect_roi_.y, u - rect_roi_.x);
						//cout<< "angle_real: "<<angle_real <<endl;

						angle_real = angle_real * 180 / CV_PI;
						
						//angle_real = 270.0 - angle_real;

						if (angle_real > 180)
							angle_real = angle_real - 360.0;
						
						float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;	
						//cout << "angle_virtual: " << angle_virtual << endl;
						float distance_angle = (angle_real - angle_virtual);

						//cout << "distance_angle: "<<distance_angle << endl;
						float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
						float max_score = 1.0;
						float edge_dir = atan2(normal_u, normal_v);
						//expendOri(max_score, ori_, cv::Point(u - rect_roi_.x, int(v_f) - rect_roi_.y), edge_dir, 1, 0, 0, normal_u, normal_v);
						if (score_match >= 0.0)
						{
							//cout<< score_match <<endl;
						//cout << "score_match: " << score_match << endl;
						//*segment_probability_f = *segment_probability_f * score_match * mag * ncc;
							max_score = powf(score_match, 1.0);
							//*segment_probability_f = *segment_probability_f  * score_match* mag;
							//float score_weight = weight_response * mag + weight_orientation * max_score + weight_ncc * ncc;
							float score_weight = weight_response * weight_orientation * max_score * mag;
							//float score_weight = weight_response * weight_orientation * mag;
							//turkey
							Eigen::Vector2f diff{ (center_u - u), (center_v - int(v_f)) };
							float squared_error = diff.squaredNorm();
							float error = sqrtf(squared_error);
							// Calculate weight with Tukey norm
							float weight = 0.0001;
							if (error > std::numeric_limits<float>::min())
								weight = (TukeyNorm(error) / squared_error);

							//float score_weight = max_score;
							*segment_probability_f = *segment_probability_f * score_weight * weight;
						}
						else
							*segment_probability_f = min_probility_;
						
						//*segment_probability_f = mag;
						//int angle_thresh = 30;
						/*if (distance_angle <= angle_thresh || (distance_angle > 180 - angle_thresh && distance_angle < 180 + angle_thresh))
						{
							*segment_probability_f = mag;
						}
						else
							*segment_probability_f = 0.0f;*/

					}
					else
						*segment_probability_f = min_probility_;
				}
			}
			else {

				float *segment_probability_f = &segment_probabilities_f->back();
				//float *segment_probability_b = &segment_probabilities_b->back();
				*segment_probability_f = 1.0f;
				//*segment_probability_b = 1.0f;
				int col_index = 0;
				int segment_idx = 0;
				int jj = line_length_minus_1_;
				for (; u <= u_end; ++u, v_f += v_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(--segment_probability_f) = 1.0f;
						//*(--segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					if (flag_use_correlation_)
					    current_img.at<cv::Vec3b>(jj, 0) = image.at<cv::Vec3b>(int(v_f), u);
					jj--;
					if (flag_use_histogram_edge)
					{
						MultiplyPixelColorProbability_edge(gray_roi_.at<uchar>(int(v_f) - rect_roi_.y, u - rect_roi_.x),
							segment_probability_f);
					}
					/*MultiplyPixelColorProbability(image.at<cv::Vec3b>(int(v_f), u),
						segment_probability_f,
						segment_probability_b);*/

						//计算每个点展开一个区域，得到描述符
					/*Eigen::Vector3f real_img_desc;
					double dir = atan2(normal_v, normal_u);
					extractEdgeDescriptor(real_img_desc, ori_, cv::Point(u, int(v_f)), dir, 1);
					float ncc = fabs(real_img_desc.dot(vimg_desc));*/

#if 0
					
					if (scale_ == 1)
					{
						double dir = atan2(normal_v, normal_u);
						cv::HOGDescriptor detector = cv::HOGDescriptor(cv::Size(5, 5), cv::Size(4, 4), cv::Size(1, 1), cv::Size(4, 4), 9);
						vector<float> descriptions;
						cv::Mat roi_img;
						extractEdgeDescriptor(roi_img, Hog_img, cv::Point(u, int(v_f)), dir, 2);
						/*cv::imshow("roi_img", roi_img);
						cv::waitKey(0);*/
						detector.compute(roi_img, descriptions);
						Eigen::VectorXf real_img_desc = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(descriptions.data(), descriptions.size());

						float innerMul = 0.0;
						float amp1 = 0.0;
						float amp2 = 0.0;
						for (int i = 0; i < vimg_desc.size(); i++)
						{
							innerMul += vimg_desc(i) * real_img_desc(i);
							amp1 += pow(vimg_desc(i), 2);
							amp2 += pow(real_img_desc(i), 2);
						}
						ncc = innerMul / (sqrt(amp1) * sqrt(amp2) + 0.00001);
					}
					
#endif
					//高斯混合值的计算
					/*float gmm_score = 1.0f;
					if (scale_ == 1)
					{
						int value = int(fgMask.at<uchar>(col_index, dateline_index));
						
						if (value > 0)
						{
							cv::circle(show_gmm, cv::Point(u, int(v_f)), 2, cv::Scalar(0, 0, 255));
							gmm_score = 0.8f;
						}
						else
						{
							gmm_score = 0.1f;
						}
					}
					col_index++;*/
					float mag = 1.0;
					mag = mag_.at<float>(int(v_f) - rect_roi_.y, u - rect_roi_.x);
					//float mag = mag_.at<float>(int(v_f) - rect_roi_.y, u - rect_roi_.x);
					///int region_sement_value = region_result_sement_img.at<uchar>(int(v_f), u);
					//mag = 1.0 - sqrt(powf(mag - 1, 2));
					//show_img.at<cv::Vec3b>(int(v_f), u) = cv::Vec3b(0, 0, 255);

					mag = powf(mag, 1);
					//边缘提取的响应值
					if (mag > th_mag)
					{
						float angle_real = ori_.at<float>(int(v_f) - rect_roi_.y, u - rect_roi_.x);
						angle_real = angle_real * 180 / CV_PI;
						//angle_real = 270.0 - angle_real;
						//cout << "angle_real: " << angle_real << endl;
						if (angle_real > 180)
							angle_real = angle_real - 360.0;

						float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
						//cout << "angle_virtual: " << angle_virtual << endl;
						float distance_angle = (angle_real - angle_virtual);
						//cout << "distance_angle: "<<distance_angle << endl;
						float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
						/*cout << "score_match: " << score_match << endl;
						cv::namedWindow("show_img", 0);
						cv::imshow("show_img", show_img);
						cv::waitKey(0);*/
						float max_score = 1.0;
						float edge_dir = atan2(normal_u, normal_v);
						//expendOri(max_score, ori_, cv::Point(u - rect_roi_.x,int(v_f) - rect_roi_.y), edge_dir, 1, 0, 0, normal_u, normal_v);

						if (score_match >= 0.0)
						{
							//cout << "score_match: " << score_match << endl;
							max_score = powf(score_match, 1.0);
							//*segment_probability_f = *segment_probability_f  * score_match* mag;

							//float score_weight = weight_response * mag + weight_orientation * max_score + weight_ncc * ncc;
							float score_weight = weight_response * weight_orientation * max_score* mag;
							//float score_weight = weight_response * weight_orientation * mag;
							//turkey
							Eigen::Vector2f diff{ (center_u - u), (center_v - int(v_f)) };
							float squared_error = diff.squaredNorm();
							float error = sqrtf(squared_error);
							// Calculate weight with Tukey norm
							float weight = 0.0001;
							if (error > std::numeric_limits<float>::min())
								weight = (TukeyNorm(error) / squared_error);
							
							//float score_weight = max_score;
							*segment_probability_f = *segment_probability_f * score_weight * weight;
							//*segment_probability_f = mag;
						}
						

						/*float angle_real = ori_.at<float>(int(v_f), u);
						float angle_virtual = atan2(normal_u, normal_v);
						float distance_angle = fabs(angle_real - angle_virtual) * 180 / CV_PI;
						int angle_thresh = 30;
						if (distance_angle <= angle_thresh || (distance_angle > 180 - angle_thresh && distance_angle < 180 + angle_thresh))
						{
							*segment_probability_f = mag;
						}
						else
							*segment_probability_f = 0.0f;*/
					}
					else
						*segment_probability_f = min_probility_;
				}
			}
			// define dominant normal component and calculate delta_r
			*normal_component_to_scale = std::fabs(normal_u) / fscale_;
			*delta_r = (std::round(center_u - line_length_minus_1_half_) +
				line_length_minus_1_half_ - center_u) /
				normal_u;
		}
		else {
			// Calculate step and starting position
			float u_step = normal_u / normal_v;
			// Notice: v = int(center_v - (line_length / 2 - 0.5) + 0.5)
			int v = int(center_v - line_length_half_minus_1_);
			int v_end = v + line_length_minus_1_;
			float u_f = center_u + u_step * (float(v) - center_v) + 0.5f;
			float u_f_end = u_f + u_step * float(line_length_minus_1_);

			// Check if line is on image (margin of 1 for rounding errors of u_f_end)
			if (v < 0 || v_end > image_height_minus_1_ || int(u_f) < 0 ||
				int(u_f) > image_width_minus_1_ || int(u_f_end) < 1 ||
				int(u_f_end) > image_width_minus_2_) {
				return false;
			}

			// Iterate over all pixels of line and calculate probabilities
			if (normal_v > 0) {

				float *segment_probability_f = segment_probabilities_f->data();
				//float *segment_probability_b = segment_probabilities_b->data();
				*segment_probability_f = 1.0f;
				//*segment_probability_b = 1.0f;
				int col_index = 0;
				int segment_idx = 0;
				int jj = 0;
				for (; v <= v_end; ++v, u_f += u_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(++segment_probability_f) = 1.0f;
						//*(++segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					if (flag_use_correlation_)
					    current_img.at<cv::Vec3b>(jj, 0) = image.at<cv::Vec3b>(v, int(u_f));
					jj++;
					if (flag_use_histogram_edge)
					{
						MultiplyPixelColorProbability_edge(gray_roi_.at<uchar>(v - rect_roi_.y, int(u_f) - rect_roi_.x),
							segment_probability_f);
					}

					/*MultiplyPixelColorProbability(image.at<cv::Vec3b>(v, int(u_f)),
						segment_probability_f,
						segment_probability_b);*/
				    //计算每个点展开一个区域，得到描述符
					/*Eigen::Vector3f real_img_desc;
					double dir = atan2(normal_v, normal_u);
					extractEdgeDescriptor(real_img_desc, ori_, cv::Point(u_f, int(v)), dir, 1);
					float ncc = fabs(real_img_desc.dot(vimg_desc));*/
#if 0
					
					if (scale_ == 1 )
					{
						double dir = atan2(normal_v, normal_u);
						cv::HOGDescriptor detector = cv::HOGDescriptor(cv::Size(5, 5), cv::Size(4, 4), cv::Size(1, 1), cv::Size(4, 4), 9);
						vector<float> descriptions;
						cv::Mat roi_img;
						extractEdgeDescriptor(roi_img, Hog_img, cv::Point(int(u_f), v), dir, 2);
						/*cv::imshow("roi_img", roi_img);
						cv::waitKey(0);*/
						detector.compute(roi_img, descriptions);
						Eigen::VectorXf real_img_desc = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(descriptions.data(), descriptions.size());

						float innerMul = 0.0;
						float amp1 = 0.0;
						float amp2 = 0.0;
						for (int i = 0; i < vimg_desc.size(); i++)
						{
							innerMul += vimg_desc(i) * real_img_desc(i);
							amp1 += pow(vimg_desc(i), 2);
							amp2 += pow(real_img_desc(i), 2);
						}
						ncc = innerMul / (sqrt(amp1) * sqrt(amp2) + 0.00001);
					}
					
#endif
					/*float gmm_score = 1.0f;
					if (scale_ == 1)
					{
						int value = int(fgMask.at<uchar>(col_index, dateline_index));
						
						if (value > 0)
						{
							cv::circle(show_gmm, cv::Point(int(u_f), v),2, cv::Scalar(0, 0, 255));
							gmm_score = 0.8f;
						}
						else
						{
							gmm_score = 0.1f;
						}
					}
					col_index++;*/
					float mag = 1.0;
					mag = mag_.at<float>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
					//float mag = mag_.at<float>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
					//int region_sement_value = region_result_sement_img.at<uchar>(v, int(u_f));
					//mag = 1.0 - sqrt(powf(mag - 1, 2));

					mag = powf(mag, 1);
					//边缘提取的响应值
					if (mag > th_mag)
					{
						float angle_real = ori_.at<float>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
						angle_real = angle_real * 180 / CV_PI;
						//angle_real = 270.0 - angle_real;
						if (angle_real > 180)
							angle_real = angle_real - 360.0;
						float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
						float distance_angle = (angle_real - angle_virtual);
						//cout << "distance_angle: "<<distance_angle << endl;
						float score_match = fabs(cos(distance_angle * CV_PI / 180.0));

						float max_score = 1.0;
						float edge_dir = atan2(normal_u, normal_v);
						//expendOri(max_score, ori_, cv::Point(int(u_f) - rect_roi_.x, v - rect_roi_.y), edge_dir, 1, 0, 0, normal_u, normal_v);

						if (score_match >= 0.0)
						{
							max_score = powf(score_match, 1.0);
							
							//cout << "score_match: " << score_match << endl;
							//*segment_probability_f = *segment_probability_f  * score_match* mag ;
							//float score_weight = weight_response * mag + weight_orientation * max_score + weight_ncc * ncc;
							float score_weight = weight_response * weight_orientation * max_score * mag;
							//float score_weight = weight_response * weight_orientation * mag;

							//turkey
							Eigen::Vector2f diff{ (center_u - int(u_f)), (center_v - v) };
							float squared_error = diff.squaredNorm();
							float error = sqrtf(squared_error);
							// Calculate weight with Tukey norm
							float weight = 0.0001;
							if (error > std::numeric_limits<float>::min())
								weight = (TukeyNorm(error) / squared_error);

							//float score_weight = max_score;
							*segment_probability_f = *segment_probability_f * score_weight * weight;
						}
						

						//*segment_probability_f = mag;
						/*float angle_real = ori_.at<float>(v, int(u_f));
						float angle_virtual = atan2(normal_u, normal_v);
						float distance_angle = fabs(angle_real - angle_virtual) * 180 / CV_PI;
						int angle_thresh = 30;
						if (distance_angle <= angle_thresh || (distance_angle > 180 - angle_thresh && distance_angle < 180 + angle_thresh))
						{
							*segment_probability_f = mag;
						}
						else
							*segment_probability_f = 0.0f;*/
					}
					else
						*segment_probability_f = min_probility_;
				}
			}
			else {
				float *segment_probability_f = &segment_probabilities_f->back();
				//float *segment_probability_b = &segment_probabilities_b->back();
				*segment_probability_f = 1.0f;
				//*segment_probability_b = 1.0f;
				int col_index = 0;
				int segment_idx = 0;
				int jj = line_length_minus_1_;
				for (; v <= v_end; ++v, u_f += u_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(--segment_probability_f) = 1.0f;
						//*(--segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					if (flag_use_correlation_)
					    current_img.at<cv::Vec3b>(jj, 0) = image.at<cv::Vec3b>(v, int(u_f));
					jj--;
					if (flag_use_histogram_edge)
					{
						MultiplyPixelColorProbability_edge(gray_roi_.at<uchar>(v - rect_roi_.y, int(u_f) - rect_roi_.x),
							segment_probability_f);
					}
					/*MultiplyPixelColorProbability(image.at<cv::Vec3b>(v, int(u_f)),
						segment_probability_f,
						segment_probability_b);*/

						//计算每个点展开一个区域，得到描述符
					/*Eigen::Vector3f real_img_desc;
					double dir = atan2(normal_v, normal_u);
					extractEdgeDescriptor(real_img_desc, ori_, cv::Point(u_f, int(v)), dir, 1);
					float ncc = fabs(real_img_desc.dot(vimg_desc));*/
#if 0
					
					if (scale_ == 1)
					{
						double dir = atan2(normal_v, normal_u);
						cv::HOGDescriptor detector = cv::HOGDescriptor(cv::Size(5, 5), cv::Size(4, 4), cv::Size(1, 1), cv::Size(4, 4), 9);
						vector<float> descriptions;
						cv::Mat roi_img;
						extractEdgeDescriptor(roi_img, Hog_img, cv::Point(int(u_f), v), dir, 2);
						/*cv::imshow("roi_img", roi_img);
						cv::waitKey(0);*/
						detector.compute(roi_img, descriptions);
						Eigen::VectorXf real_img_desc = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(descriptions.data(), descriptions.size());

						float innerMul = 0.0;
						float amp1 = 0.0;
						float amp2 = 0.0;
						for (int i = 0; i < vimg_desc.size(); i++)
						{
							innerMul += vimg_desc(i) * real_img_desc(i);
							amp1 += pow(vimg_desc(i), 2);
							amp2 += pow(real_img_desc(i), 2);
						}
						ncc = innerMul / (sqrt(amp1) * sqrt(amp2) + 0.00001);
					}
					
#endif
					/*float gmm_score = 1.0f;
					if (scale_ == 1)
					{
						int value = int(fgMask.at<uchar>(col_index, dateline_index));
						
						if (value > 0)
						{
							cv::circle(show_gmm, cv::Point(int(u_f), v), 2, cv::Scalar(0, 0, 255));
							gmm_score = 0.8f;
						}
						else
						{
							gmm_score = 0.1f;
						}
					}
					col_index++;*/
					float mag = 1.0;
					mag = mag_.at<float>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
					//cout<< mag <<endl;
					//float mag = mag_.at<float>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
					//int region_sement_value = region_result_sement_img.at<uchar>(v, int(u_f));
					//mag = 1.0 - sqrt(powf(mag - 1, 2));
					//cout<<"qian: "<<mag<<endl;
					mag = powf(mag, 1);
					//cout << "hou: " << mag << endl;
					//边缘提取的响应值
					if (mag > th_mag)
					{
						float angle_real = ori_.at<float>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
						angle_real = angle_real * 180 / CV_PI;
						//angle_real = 270.0 - angle_real;
						if (angle_real > 180)
							angle_real = angle_real - 360.0;
						float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
						float distance_angle = (angle_real - angle_virtual);
						float score_match = fabs(cos(distance_angle * CV_PI / 180.0));

						float max_score = 1.0;
						float edge_dir = atan2(normal_u, normal_v);
						//expendOri(max_score, ori_,cv::Point(int(u_f) - rect_roi_.x, v - rect_roi_.y), edge_dir,1,0,0, normal_u, normal_v);
						/*cout << score_match << endl;
						cout<< max_score <<endl;*/
						if (score_match >= 0.0)
						{
							//cout << "qian："<<max_score << endl;
							max_score = powf(score_match, 1.0);  //0.5

							//cout << "hou：" << max_score << endl;
							//cout << "score_match: " << score_match << endl;
							//*segment_probability_f = *segment_probability_f  * score_match * mag;
							//float score_weight = weight_response * mag + weight_orientation * max_score + weight_ncc * ncc;
							float score_weight = weight_response * weight_orientation * max_score* mag;
							//float score_weight = weight_response * weight_orientation * mag;

							//turkey
							Eigen::Vector2f diff{ (center_u - int(u_f)), (center_v - v) };
							float squared_error = diff.squaredNorm();
							float error = sqrtf(squared_error);
							// Calculate weight with Tukey norm
							float weight = 0.0001;
							if (error > std::numeric_limits<float>::min())
								weight = (TukeyNorm(error) / squared_error);
							//cout << weight << endl;
							//float score_weight = max_score;
							*segment_probability_f = *segment_probability_f * score_weight * weight;
						}
						//*segment_probability_f = mag;
						/*float angle_real = ori_.at<float>(v, int(u_f));
						float angle_virtual = atan2(normal_u, normal_v);
						float distance_angle = fabs(angle_real - angle_virtual) * 180 / CV_PI;
						int angle_thresh = 30;
						if (distance_angle <= angle_thresh || (distance_angle > 180 - angle_thresh && distance_angle < 180 + angle_thresh))
						{
							*segment_probability_f = mag;
						}
						else
							*segment_probability_f = 0.0f;*/
					}
					else
						*segment_probability_f = min_probility_;

				}
			}

			// define normal component and calculate delta_r
			*normal_component_to_scale = std::fabs(normal_v) / fscale_;
			*delta_r = (std::round(center_v - line_length_minus_1_half_) +
				line_length_minus_1_half_ - center_v) /
				normal_v;
		}
		
		// Normalize segment probabilities
		if (scale_ >= 1) {

			/*auto segment_probability_f = begin(*segment_probabilities_f);
			auto segment_probability_b = begin(*segment_probabilities_b);
			for (; segment_probability_f != end(*segment_probabilities_f);
				++segment_probability_f, ++segment_probability_b) {
				if (*segment_probability_f || *segment_probability_b) {
					float sum = *segment_probability_f;
					sum += *segment_probability_b;
					*segment_probability_f /= sum;
					*segment_probability_b /= sum;
				}
				else {
					*segment_probability_f = 0.5f;
					*segment_probability_b = 0.5f;
				}
			}*/

			auto segment_probability_f = begin(*segment_probabilities_f);
			for (; segment_probability_f != end(*segment_probabilities_f);
				++segment_probability_f) {
				//std::cout << "*segment_probability_f edge: "<< *segment_probability_f << std::endl;
				if (*segment_probability_f) {
					if (*segment_probability_f <= 0.0)
					{
						*segment_probability_f = min_probility_;
					}
					else
					{
						/*float sum = *segment_probability_f;
						*segment_probability_f = sum;*/
						continue;
					}
					
				}
				else {
					*segment_probability_f = min_probility_;
				}
			}
		}

		std::vector<float> temp_match_;
		if (flag_use_correlation_)
		{
			cv::Mat result;
			/*较为耗时*/
			cv::matchTemplate(current_img, Last_Frame_one_line, result, cv::TM_CCORR_NORMED);
			current_mat_vector.push_back(current_img);
			
			for (int i = 0; i < result.rows; i++)
			{
				temp_match_.push_back(result.at<float>(i, 0));
				
				//cout << result.at<float>(i, 0) << endl;
			}
			double minVal, maxVal;
			cv::Point minLoc, maxLoc;
			cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
			// 最大匹配值即为归一化 NCC		
			match_result_vector.push_back(maxLoc);
		}
		
		//cout<< temp_match_ .size()<<endl;
		//计算分布

		std::vector<float>::const_iterator segment_probabilities_f_it;
		std::vector<float>::const_iterator function_lookup_f_it;
		std::vector<float>::const_iterator function_lookup_f_distribution_it;

		distribution->resize(distribution_length_);
		float distribution_area = 0.0f;

		// Loop over entire distribution and start values of segment probabilities
		auto segment_probabilities_f_it_start = begin(*segment_probabilities_f);

		//std::vector<float> match_ratio_of_lastFrame = match_ratio_of_lastFrame_[index_for_match_lastFrame_];
		int index_temp_ = int(float(current_img.rows - temp_match_.size()) / 2);
		//cout<< index_temp_ <<endl;
		int count_num = 0;
		int count_num_in = 0;
		function_lookup_f_distribution_it = begin(function_lookup_edge_distrbution_all);

		for (auto distribution_it = begin(*distribution);
			distribution_it != end(*distribution);
			++distribution_it, ++segment_probabilities_f_it_start, ++function_lookup_f_distribution_it) {
			*distribution_it = 1.0f;
			// Loop over values of segment probabilities and corresponding lookup values
			segment_probabilities_f_it = segment_probabilities_f_it_start;
			function_lookup_f_it = begin(function_lookup_edge_);	

			int ii = 0;
			for (; function_lookup_f_it != end(function_lookup_edge_);
				++function_lookup_f_it, 
				++segment_probabilities_f_it) {			
				float match_value_last_frame = 1.0f;
				if (count_num > index_temp_ && (count_num - index_temp_ - 1 + ii) < temp_match_.size())
				{				
					if (flag_tracking_success_ && flag_num_iter_for_use)
					{
						//cout << count_num - index_temp_ - 1 + ii << endl;
						if (flag_use_correlation_)
						{
							float k = 2;
							float a = 0.5;
							match_value_last_frame = temp_match_[count_num - index_temp_ - 1 + ii];
							match_value_last_frame = match_value_last_frame * match_value_last_frame;
							//match_value_last_frame =  1 - std::exp(-k * (match_value_last_frame - a));
							//match_value_last_frame = match_value_last_frame * last_ratio;
							//match_value_last_frame = match_value_last_frame * match_value_last_frame * match_value_last_frame;
							//cout<< match_value_last_frame <<endl;
						}						
					}
				}	
				if (flag_use_correlation_)
				{
					*distribution_it *= *segment_probabilities_f_it * *function_lookup_f_it * match_value_last_frame;
				}
				else
					*distribution_it *= *segment_probabilities_f_it * *function_lookup_f_it;
				//cout<< count_num + ii <<endl;
				//*distribution_it *= *segment_probabilities_f_it * *function_lookup_f_it * (1 - last_ratio) +  match_value_last_frame * last_ratio;				
				ii++;
				//std::cout << "distribution_it edge: " << *distribution_it << std::endl;
			}

			//整个分布的高斯
		    //*distribution_it *= *function_lookup_f_distribution_it;
			/*上一帧匹配成功后计算匹配*/
			//if (count_num >= index_temp_)
			//{
			//	if (count_num - index_temp_ < match_ratio_of_lastFrame.size())
			//	{
			//		if (flag_tracking_success_)
			//		{
			//			//cout << count_num - index_temp_ << endl;
			//			//最后两次迭代执行才更加有效？
			//			//*distribution_it *= match_ratio_of_lastFrame[count_num - index_temp_];
			//			//cout << match_ratio_of_lastFrame[count_num - index_temp_] << endl;
			//		}	
			//	}
			//}	
			count_num++;
			/*结束*/

			if (*distribution_it == 0)
			{
				*distribution_it = min_probility_;
			}
			distribution_area += *distribution_it;
			//std::cout << "distribution_it edge: " << *distribution_it << std::endl;
		}

		// Normalize distribution
		for (auto &probability_distribution : *distribution) {
			if (distribution_area == 0)
			{
				probability_distribution = probability_distribution;
			}
			else
			    probability_distribution /= (distribution_area);
			//std::cout << "distribution edge: "<< probability_distribution << std::endl;
		}


		float mean_from_begin = 0.0f;
		auto distribution_it_1 = begin(*distribution);

		for (int i = 0; i < distribution_length_; ++i,++distribution_it_1) {
			mean_from_begin += float(i) * *distribution_it_1;
			//cout <<"*distribution_it_1 "<< *distribution_it_1 << endl;
		}

		// Calculate variance
		float distribution_variance = 0.0f;
		auto distribution_it_2 = begin(*distribution);
		for (int i = 0; i < distribution_length_; ++i, ++distribution_it_2) {
			distribution_variance +=
				powf(float(i) - mean_from_begin, 2.0f) * *distribution_it_2;
		}

		// Calculate moments
		*mean = mean_from_begin - distribution_length_minus_1_half_;		
		//cout<< fabs(mean_before_- *mean) <<endl;
		//mean_vector_.push_back(fabs(mean_before_ - *mean));
		//mean_before_ = *mean;
		*variance = std::max(distribution_variance, min_variance_);
		*standard_deviation = std::sqrt(*variance);

		//cout << "mean_from_begin edge " << mean_from_begin << endl;
		//cout << "mean edge " << *mean << endl;
		return true;
	}

	bool EdgeModality::CalculateSegmentProbabilities(
		float center_u, float center_v, float normal_u, float normal_v,
		std::vector<float> *segment_probabilities_f,
		std::vector<float> *segment_probabilities_b,
		float *normal_component_to_scale, float *delta_r) const {
		const cv::Mat &image{ camera_ptr_->image() };

		// Select case if line is more horizontal or vertical
		if (std::fabs(normal_v) < std::fabs(normal_u)) {
			// Calculate step and starting position
			float v_step = normal_v / normal_u;
			// Notice: u = int(center_u - (line_length / 2 - 0.5) + 0.5)
			int u = int(center_u - line_length_half_minus_1_);
			int u_end = u + line_length_minus_1_;
			float v_f = center_v + v_step * (float(u) - center_u) + 0.5f;
			float v_f_end = v_f + v_step * float(line_length_minus_1_);

			// Check if line is on image (margin of 1 for rounding errors of v_f_end)
			if (u < 0 || u_end > image_width_minus_1_ || int(v_f) < 0 ||
				int(v_f) > image_height_minus_1_ || int(v_f_end) < 1 ||
				int(v_f_end) > image_height_minus_2_) {
				return false;
			}

			// Iterate over all pixels of line and calculate probabilities
			if (normal_u > 0) {
				float *segment_probability_f = segment_probabilities_f->data();
				float *segment_probability_b = segment_probabilities_b->data();
				*segment_probability_f = 1.0f;
				*segment_probability_b = 1.0f;
				int segment_idx = 0;
				for (; u <= u_end; ++u, v_f += v_step, segment_idx++) {
					if (segment_idx == scale_) {
						*(++segment_probability_f) = 1.0f;
						*(++segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					MultiplyPixelColorProbability(image.at<cv::Vec3b>(int(v_f), u),
						segment_probability_f,
						segment_probability_b);
				}
			}
			else {
				float *segment_probability_f = &segment_probabilities_f->back();
				float *segment_probability_b = &segment_probabilities_b->back();
				*segment_probability_f = 1.0f;
				*segment_probability_b = 1.0f;
				int segment_idx = 0;
				for (; u <= u_end; ++u, v_f += v_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(--segment_probability_f) = 1.0f;
						*(--segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					MultiplyPixelColorProbability(image.at<cv::Vec3b>(int(v_f), u),
						segment_probability_f,
						segment_probability_b);
				}
			}

			// define dominant normal component and calculate delta_r
			*normal_component_to_scale = std::fabs(normal_u) / fscale_;
			*delta_r = (std::round(center_u - line_length_minus_1_half_) +
				line_length_minus_1_half_ - center_u) /
				normal_u;
		}
		else {
			// Calculate step and starting position
			float u_step = normal_u / normal_v;
			// Notice: v = int(center_v - (line_length / 2 - 0.5) + 0.5)
			int v = int(center_v - line_length_half_minus_1_);
			int v_end = v + line_length_minus_1_;
			float u_f = center_u + u_step * (float(v) - center_v) + 0.5f;
			float u_f_end = u_f + u_step * float(line_length_minus_1_);

			// Check if line is on image (margin of 1 for rounding errors of u_f_end)
			if (v < 0 || v_end > image_height_minus_1_ || int(u_f) < 0 ||
				int(u_f) > image_width_minus_1_ || int(u_f_end) < 1 ||
				int(u_f_end) > image_width_minus_2_) {
				return false;
			}

			// Iterate over all pixels of line and calculate probabilities
			if (normal_v > 0) {
				float *segment_probability_f = segment_probabilities_f->data();
				float *segment_probability_b = segment_probabilities_b->data();
				*segment_probability_f = 1.0f;
				*segment_probability_b = 1.0f;
				int segment_idx = 0;
				for (; v <= v_end; ++v, u_f += u_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(++segment_probability_f) = 1.0f;
						*(++segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					MultiplyPixelColorProbability(image.at<cv::Vec3b>(v, int(u_f)),
						segment_probability_f,
						segment_probability_b);
				}
			}
			else {
				float *segment_probability_f = &segment_probabilities_f->back();
				float *segment_probability_b = &segment_probabilities_b->back();
				*segment_probability_f = 1.0f;
				*segment_probability_b = 1.0f;
				int segment_idx = 0;
				for (; v <= v_end; ++v, u_f += u_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(--segment_probability_f) = 1.0f;
						*(--segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					MultiplyPixelColorProbability(image.at<cv::Vec3b>(v, int(u_f)),
						segment_probability_f,
						segment_probability_b);
				}
			}

			// define normal component and calculate delta_r
			*normal_component_to_scale = std::fabs(normal_v) / fscale_;
			*delta_r = (std::round(center_v - line_length_minus_1_half_) +
				line_length_minus_1_half_ - center_v) /
				normal_v;
		}

		// Normalize segment probabilities
		if (scale_ > 1) {
			auto segment_probability_f = begin(*segment_probabilities_f);
			auto segment_probability_b = begin(*segment_probabilities_b);
			for (; segment_probability_f != end(*segment_probabilities_f);
				++segment_probability_f, ++segment_probability_b) {
				if (*segment_probability_f || *segment_probability_b) {
					float sum = *segment_probability_f;
					sum += *segment_probability_b;
					*segment_probability_f /= sum;
					*segment_probability_b /= sum;
				}
				else {
					*segment_probability_f = 0.5f;
					*segment_probability_b = 0.5f;
				}
			}
		}
		return true;
	}

	void EdgeModality::MultiplyPixelColorProbability(const cv::Vec3b &pixel_color,
		float *probability_f,
		float *probability_b) const {
		// Retrive pixel color probability values
		int idx = (pixel_color[0] >> histogram_bitshift_) * n_histogram_bins_squared_;
		idx += (pixel_color[1] >> histogram_bitshift_) * n_histogram_bins_;
		idx += pixel_color[2] >> histogram_bitshift_;
		float pixel_color_probability_f = histogram_f_[idx];
		float pixel_color_probability_b = histogram_b_[idx];

		// Normalize pixel color probabilitiy values
		if (pixel_color_probability_f || pixel_color_probability_b) {
			float sum = pixel_color_probability_f;
			sum += pixel_color_probability_b;
			pixel_color_probability_f /= sum;
			pixel_color_probability_b /= sum;
		}
		else {
			pixel_color_probability_f = 0.5f;
			pixel_color_probability_b = 0.5f;
		}

		// Multiply pixel color probability values
		*probability_f *= pixel_color_probability_f;
		*probability_b *= pixel_color_probability_b;
		// std::cout << *probability_f << std::endl;
	}
	void EdgeModality::MultiplyPixelColorProbability_edge(const uchar &pixel_color,
		float *probability_f) const {
		// Retrive pixel color probability values
		int idx = (pixel_color >> histogram_bitshift_);
		float pixel_color_probability_f = histogram_f_[idx];

		// Normalize pixel color probabilitiy values
		if (pixel_color_probability_f) {
			pixel_color_probability_f = pixel_color_probability_f;
		}
		else {
			pixel_color_probability_f = 0.00001f;
		}
		// Multiply pixel color probability values
		*probability_f *= pixel_color_probability_f;
	}
	void EdgeModality::CalculateDistribution(
		const std::vector<float> &segment_probabilities_f,
		const std::vector<float> &segment_probabilities_b,
		std::vector<float> *distribution) const {
		std::vector<float>::const_iterator segment_probabilities_f_it;
		std::vector<float>::const_iterator segment_probabilities_b_it;
		std::vector<float>::const_iterator function_lookup_f_it;
		std::vector<float>::const_iterator function_lookup_b_it;
		distribution->resize(distribution_length_);
		float distribution_area = 0.0f;

		// Loop over entire distribution and start values of segment probabilities
		auto segment_probabilities_f_it_start = begin(segment_probabilities_f);
		auto segment_probabilities_b_it_start = begin(segment_probabilities_b);
		for (auto distribution_it = begin(*distribution);
			distribution_it != end(*distribution);
			++distribution_it, ++segment_probabilities_f_it_start,
			++segment_probabilities_b_it_start) {
			*distribution_it = 1.0f;
			// Loop over values of segment probabilities and corresponding lookup values
			segment_probabilities_f_it = segment_probabilities_f_it_start;
			segment_probabilities_b_it = segment_probabilities_b_it_start;
			function_lookup_f_it = begin(function_lookup_f_);
			function_lookup_b_it = begin(function_lookup_b_);
			for (; function_lookup_f_it != end(function_lookup_f_);
				++function_lookup_f_it, ++function_lookup_b_it,
				++segment_probabilities_f_it, ++segment_probabilities_b_it) {
				*distribution_it *= *segment_probabilities_f_it * *function_lookup_f_it +
					*segment_probabilities_b_it * *function_lookup_b_it;
			}
			distribution_area += *distribution_it;
		}

		// Normalize distribution
		for (auto &probability_distribution : *distribution) {
			probability_distribution /= distribution_area;
		}
	}

	void EdgeModality::CalculateDistributionMoments(
		const std::vector<float> &distribution, float *mean,
		float *standard_deviation, float *variance) const {
#if 0
		// Calculate mean from the beginning of the distribution
		float mean_from_begin = 0.0f;
		for (int i = 0; i < distribution_length_; ++i) {
			mean_from_begin += float(i) * distribution[i];
		}

		// Calculate variance
		float distribution_variance = 0.0f;
		for (int i = 0; i < distribution_length_; ++i) {
			distribution_variance +=
				powf(float(i) - mean_from_begin, 2.0f) * distribution[i];
		}

		// Calculate moments
		*mean = mean_from_begin - distribution_length_minus_1_half_;
		*variance = std::max(distribution_variance, min_variance_);
		*standard_deviation = std::sqrt(*variance);
#endif	
		//求均值
		float distribution_area = 0;
		for (int i = 0; i < distribution_length_ * scale_; ++i) {
			distribution_area += distribution[i];
		}
		// Calculate variance
		float distribution_variance = 0.0f;
		for (int i = 0; i < distribution_length_ * scale_; ++i) {
			distribution_variance +=
				powf(distribution[i] - distribution_area, 2.0f);
		}
		*mean = distribution_area/(distribution_length_ * scale_);
		*variance = distribution_variance / (distribution_length_ * scale_);
		*standard_deviation = std::sqrt(*variance);
		/*cout << *mean << endl;
		cout << *variance << endl;
		cout << *standard_deviation << endl;*/
	}

	void EdgeModality::ShowAndSaveImage(const std::string &title, int save_index,
		const cv::Mat &image) const {
		if (display_visualization_)
		{
			cv::namedWindow(title,0);
			cv::imshow(title, image);
		}
		if (save_visualizations_) {
			std::experimental::filesystem::path path{
				save_directory_ / (title + "_" + std::to_string(save_index) + ".png") };
			cv::imwrite(path.string(), image);
		}
	}

	void EdgeModality::VisualizePointsCameraImage(const std::string &title,
		int save_index) const {
		cv::Mat visualization_image;
		camera_ptr_->image().copyTo(visualization_image);
		DrawPoints(cv::Vec3b{ 24, 184, 234 }, &visualization_image);
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
	}

	void EdgeModality::VisualizePointsHistogramImage(const std::string &title,
		int save_index) const {
		cv::Mat visualization_image(camera_ptr_->image().size(), CV_8UC3);
		DrawProbabilityImage(cv::Vec3b{ 255, 255, 255 }, &visualization_image);
		DrawPoints(cv::Vec3b{ 24, 184, 234 }, &visualization_image);
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
	}

	void EdgeModality::VisualizePointsOcclusionMask(const std::string &title,
		int save_index) const {
		cv::Mat visualization_image;
		occlusion_renderer_ptr_->FetchOcclusionMask();
		cv::cvtColor(occlusion_renderer_ptr_->occlusion_mask(), visualization_image,
			cv::COLOR_GRAY2BGR);
		cv::resize(visualization_image, visualization_image,
			cv::Size{ camera_ptr_->intrinsics().width,
			camera_ptr_->intrinsics().height },
			occlusion_renderer_ptr_->mask_resolution(),
			occlusion_renderer_ptr_->mask_resolution(),
			cv::InterpolationFlags::INTER_NEAREST);
		cv::addWeighted(visualization_image, 0.4, camera_ptr_->image(), 0.6, 10,
			visualization_image);
		DrawPoints(cv::Vec3b{ 24, 184, 234 }, &visualization_image);
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
	}

	void EdgeModality::VisualizeLines(const std::string &title,
		int save_index) const {
		/*cv::Mat visualization_image_for_edge = camera_ptr_->image().clone();
		DrawLines_fb(cv::Vec3b{ 24, 184, 234 }, cv::Vec3b{ 61, 63, 179 },&visualization_image_for_edge);
		DrawPoints(cv::Vec3b{ 0, 255, 0 }, &visualization_image_for_edge);
		cv::imshow("visualization_image_for_edge", visualization_image_for_edge);
		cv::resize(visualization_image_for_edge, visualization_image_for_edge,
			cv::Size(visualization_image_for_edge.cols * 2, visualization_image_for_edge.rows * 2));
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image_for_edge);*/
		//cv::waitKey(0);

		//cv::Mat visualization_image(camera_ptr_->image().size(), CV_8UC3);
		cv::Mat visualization_image = camera_ptr_->image().clone();
		//边缘提取
		cv::Mat sobel_img;
		cv::cvtColor(visualization_image, sobel_img, CV_BGR2GRAY);
		cv::Mat gradX, gradY;
		cv::Sobel(sobel_img, gradX, CV_16S, 1, 0, 3);
		cv::Sobel(sobel_img, gradY, CV_16S, 0, 1, 3);
		// Scharr(srcGray, gradX, CV_16S, 1, 0);
		// Scharr(srcGray, gradY, CV_16S, 0, 1);
		convertScaleAbs(gradX, gradX);  // calculates absolute values, and converts the result to 8-bit.
		convertScaleAbs(gradY, gradY);
		cv::Mat dst;
		cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, dst);

		//bitwise_not(dst, dst);
		//CalculateEdgeClutter(cv::Vec3b{ 24, 184, 234 }, cv::Vec3b{ 61, 63, 179 }, &dst);
		cv::cvtColor(dst, dst, CV_GRAY2BGR);
		
		cv::Mat image_edge = camera_ptr_->image_edge().clone();
		cv::Mat result;
		cv::addWeighted(image_edge, 0.6, dst, 0.4, 0, result);
		//cv::Mat visualization_image;
		//cv::cvtColor(Hog_img, visualization_image,cv::COLOR_GRAY2BGR);
		//cv::Mat visualization_image = Canny_Mat_.clone();
		//cv::cvtColor(visualization_image, visualization_image,CV_GRAY2BGR);
		//DrawProbabilityImage(cv::Vec3b{ 255, 255, 255 }, &visualization_image);
	    DrawLines(cv::Vec3b{ 24, 184, 234 }, cv::Vec3b{ 61, 63, 179 },&result);
		ShowAndSaveImage(name_ + "_" + title, save_index, result);
	}

	void EdgeModality::DrawPoints(const cv::Vec3b &color_point,
		cv::Mat *image) const {
		for (const auto &data_line : data_lines_) {
			DrawPointInImage(data_line.center_f_camera, color_point,
				camera_ptr_->intrinsics(), image);
		}
	}

	void EdgeModality::DrawLines(const cv::Vec3b &color_line,
		const cv::Vec3b &color_high_probability,
		cv::Mat *image) const {
		cv::Mat bundle_img = cv::Mat::zeros(cv::Size(data_lines_.size(), distribution_length_ * scale_),CV_8UC3);
		int line_index = 0;
		
		float scale_minus_1_half_ = (fscale_ - 1.0f) / 2.0f;
		int u, v;
		for (const auto &data_line : data_lines_) {
			float x = 0;
			float y = 0;
			float max_turkey = 0;

			int distru_index = 0;
			for (int i = 0; i < distribution_length_; ++i) {
				
				for (int j = 0; j < scale_; ++j) {
					//首先绘制的是那个点？
					if (std::fabs(data_line.normal_u) > std::fabs(data_line.normal_v)) {
						u = int(
							data_line.center_u +
							float(sgn(data_line.normal_u)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						v = int(data_line.center_v +
							(float(u) - data_line.center_u) *
							(data_line.normal_v / data_line.normal_u) +
							0.5f);

                    }
					else {
						v = int(
							data_line.center_v +
							float(sgn(data_line.normal_v)) *
							(fscale_ * (float(i) - distribution_length_minus_1_half_) +
								float(j) - scale_minus_1_half_) +
							0.5f);
						u = int(data_line.center_u +
							(float(v) - data_line.center_v) *
							(data_line.normal_u / data_line.normal_v) +
							0.5f);		
					}		
					float color_ratio = std::min(3 * data_line.distribution[i], 1.0f);
					image->at<cv::Vec3b>(v, u) = color_ratio * color_high_probability +
						(1.0f - color_ratio) * color_line;					
				}			
			}
		}
		
	}
	void EdgeModality::DrawLines_fb(const cv::Vec3b &color_line,
		const cv::Vec3b &color_high_probability,
		cv::Mat *image) const {
		cv::Mat image_for_bunble = image->clone();
		int line_length = 10;
		cv::Mat img_bundle = cv::Mat::zeros(cv::Size(int(line_length) * 2 - 1, data_lines_.size()), CV_8UC3);
		bitwise_not(img_bundle, img_bundle);
		int count_lines = 0;

		for (const auto &data_line : data_lines_) {
			int length = 0;
			// Iterate over foreground pixels
			Eigen::Vector2f center{
					data_line.center_u,
					data_line.center_v };
			//将其转换到图像
			Eigen::Vector2f normal{
				data_line.normal_u ,data_line.normal_v };

			// Iterate over background pixels
			float u = center(0) + normal(0) * unconsidered_line_length_;
			float v = center(1) + normal(1) * unconsidered_line_length_;

			//线长定义
			int n_iteration = int(line_length + 0.5);
			//cout<< n_iteration <<endl;
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				image->at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);
				img_bundle.at<cv::Vec3b>(count_lines, int(line_length) - length - 1) = image_for_bunble.at<cv::Vec3b>(int(v), int(u));
				//cout << int(line_length) - length - 1 << endl;
				length++;
				u += normal(0);
				v += normal(1);
			}
			
			u = center(0) - normal(0) * unconsidered_line_length_;
			v = center(1) - normal(1) * unconsidered_line_length_;
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				image->at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);  //24, 184, 234 yellow
				img_bundle.at<cv::Vec3b>(count_lines, length - 1) = image_for_bunble.at<cv::Vec3b>(int(v), int(u));
				//cout << length - 1 << endl;
				length++;
				u -= normal(0);
				v -= normal(1);

			}
			count_lines++;
		}
		cv::Mat draw_match_points_img = img_bundle.clone();
		cv::cvtColor(img_bundle, img_bundle, CV_BGR2GRAY);
		cv::Mat Sobel1D_dst = cv::Mat::zeros(img_bundle.size(), CV_8UC1);
		for (int r = 0; r < img_bundle.rows; ++r) {
			for (int c = 1; c < img_bundle.cols - 1; c++)
			{
				Sobel1D_dst.at<uchar>(r, c) = (uchar)abs(img_bundle.at<uchar>(r, c + 1) - img_bundle.at<uchar>(r, c - 1));
			}
			Sobel1D_dst.at<uchar>(r, 0) = Sobel1D_dst.at<uchar>(r, img_bundle.cols - 1) = 0;
		}
		bitwise_not(Sobel1D_dst, Sobel1D_dst);
		int valid_row = Sobel1D_dst.rows;
		int valid_row_all = 0;
		int match_sucess_num = 0;
		int match_num = 0;
		for (int i = 0;i < Sobel1D_dst.rows;i++)
		{	
			for (int j = 0;j < Sobel1D_dst.cols;j++)
			{
				//cout<<int(Sobel1D_dst.at<uchar>(i, j))<<endl;			
				float distance = fabs((float(Sobel1D_dst.cols/2) - j));
				if (int(Sobel1D_dst.at<uchar>(i,j)) <= 230 && distance <= 3)
				{
					draw_match_points_img.at<cv::Vec3b>(i, j) = cv::Vec3b(0,255,0);
					match_num++;
					valid_row_all++;
					match_sucess_num++;
				}
				if (int(Sobel1D_dst.at<uchar>(i, j)) <= 230 && distance > 3)
				{
					draw_match_points_img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
					match_num++;
					valid_row_all++;
				}
			}
			if (match_num == 0)
			{
				valid_row--;
			}
		}
		cout<<"background cluuter ratio: "<<float(match_sucess_num) / match_num <<endl;
		cv::imshow("img_bundle", img_bundle);
		imshow("Sobel1D_dst", Sobel1D_dst);
		cv::imshow("draw_match_points_img", draw_match_points_img);
	     //cv::waitKey(0);
	}
	void EdgeModality::CalculateEdgeClutter(const cv::Vec3b &color_line,
		const cv::Vec3b &color_high_probability,
		cv::Mat *image) const {
		int line_length = 15;
		cv::Mat img_bundle = cv::Mat::zeros(cv::Size(int(line_length + 0.5) * 2, data_lines_.size()), CV_8UC3);
		bitwise_not(img_bundle, img_bundle);
		//cv::imshow("img_bundle_ttt", img_bundle);
		//cv::waitKey(0);

		//全局图像
		cv::Mat img_for_generate_bundle = image->clone();
		cv::cvtColor(img_for_generate_bundle, img_for_generate_bundle, CV_GRAY2BGR);

		int count_lines = 0;
		int match_size = 0;
		for (const auto &data_line : data_lines_) {
			int length = 0;
			// Iterate over foreground pixels
			Eigen::Vector2f center{
					data_line.center_u,
					data_line.center_v };
			//将其转换到图像
			Eigen::Vector2f normal{
				data_line.normal_u ,data_line.normal_v };

			// Iterate over background pixels
			float u = center(0) + normal(0) * unconsidered_line_length_;
			float v = center(1) + normal(1) * unconsidered_line_length_;
			//线长定义
			int n_iteration = int(line_length) + 0.5f;

			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				//image->at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);
				img_bundle.at<cv::Vec3b>(count_lines, int(line_length + 0.5) - length) = img_for_generate_bundle.at<cv::Vec3b>(int(v), int(u));
				
				length++;
				u += normal(0);
				v += normal(1);		
			}

			u = center(0) - normal(0) * unconsidered_line_length_;
			v = center(1) - normal(1) * unconsidered_line_length_;		
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				//image->at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);  //24, 184, 234 yellow
				img_bundle.at<cv::Vec3b>(count_lines, length) = img_for_generate_bundle.at<cv::Vec3b>(int(v), int(u));
				
				length++;
				u -= normal(0);
				v -= normal(1);
				
			}			
			count_lines++;
			
		}
		cv::imshow("img_bundle-1", img_bundle);
		cv::waitKey(0);
		
		/*绘制匹配点*/
		count_lines = 0;
		match_size = 0;
		for (const auto &data_line : data_lines_) {
			int length = 0;
			// Iterate over foreground pixels
			Eigen::Vector2f center{
					data_line.center_u,
					data_line.center_v };
			//将其转换到图像
			Eigen::Vector2f normal{
				data_line.normal_u ,data_line.normal_v };

			float u = center(0) + normal(0) * unconsidered_line_length_;
			float v = center(1) + normal(1) * unconsidered_line_length_;
			//线长定义
			int n_iteration = int(line_length) + 0.5f;

			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				//image->at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);  //24, 184, 234 yellow
				if (image->at<uchar>(int(v), int(u)) < 200)
				{
					img_bundle.at<cv::Vec3b>(count_lines, int(line_length + 0.5) - length) = cv::Vec3b(0, 255, 0);
					match_size++;
				}
				
				length++;
				u += normal(0);
				v += normal(1);
			}

			// Iterate over background pixels
			u = center(0) - normal(0) * unconsidered_line_length_;
			v = center(1) - normal(1) * unconsidered_line_length_;

			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				//image->at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);
				if (image->at<uchar>(int(v), int(u)) < 200)
				{
					img_bundle.at<cv::Vec3b>(count_lines, length) = cv::Vec3b(0, 255, 0);
					match_size++;
				}
				length++;
				u -= normal(0);
				v -= normal(1);
			}
			count_lines++;
		}
		cout<<"backgroundClutter: "<< float(data_lines_.size()) / match_size <<endl;
		cv::imshow("img_bundle_clutter", img_bundle);
		cv::waitKey(0);
	}

	void EdgeModality::DrawProbabilityImage(const cv::Vec3b &color_b,
		cv::Mat *probability_image) const {
		const cv::Mat &color_image{ camera_ptr_->image() };
		float pixel_probability_f, pixel_probability_b;
		const cv::Vec3b *color_image_value;
		cv::Vec3b *probability_image_value;
		for (int v = 0; v < color_image.rows; ++v) {
			color_image_value = color_image.ptr<cv::Vec3b>(v);
			probability_image_value = probability_image->ptr<cv::Vec3b>(v);
			for (int u = 0; u < color_image.cols; ++u) {
				pixel_probability_f = 1.0f;
				pixel_probability_b = 1.0f;
				MultiplyPixelColorProbability(color_image_value[u], &pixel_probability_f,
					&pixel_probability_b);
				probability_image_value[u] = color_b * pixel_probability_b;
			}
		}
	}

	void EdgeModality::UpdateLineCentersWithCurrentPose() {
		Transform3fA body2camera_pose{ camera_ptr_->world2camera_pose() *
			body_ptr_->body2world_pose() };
		for (auto &data_line : data_lines_) {
			data_line.center_f_camera = body2camera_pose * data_line.center_f_body;
		}
	}

	float EdgeModality::MinAbsValueWithSignOfValue1(float value_1,
		float abs_value_2) {
		if (std::abs(value_1) < abs_value_2)
			return value_1;
		else
			return sgnf(value_1) * abs_value_2;
	}

	bool EdgeModality::IsSetup() const {
		if (!set_up_) {
			std::cerr << "Set up region modality " << name_ << " first" << std::endl;
			return false;
		}
		return true;
	}

	float EdgeModality::tukey_cost(float x, float c)
	{
		if (fabs(x) <= c) {
			return (c*c / 6)*((1 - (x / c)*(x / c))*(1 - (x / c)*(x / c))*(1 - (x / c)*(x / c)));
		}
		else {
			return 0;
		}
	}

	void EdgeModality::checkLineExtremes(cv::Vec4f& extremes, cv::Size imageSize)
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
	bool EdgeModality::extractEdgeDescriptor(cv::Mat& desc, const cv::Mat& im, cv::Point pt,
		double edge_dir, unsigned int window_size) const
	{
		double dx = cos(edge_dir);
		double dy = sin(edge_dir);

		double dx_k = cos(CV_PI / 2 + edge_dir);
		double dy_k = sin(CV_PI / 2 + edge_dir);

		desc.create(2 * window_size + 1, 2 * window_size + 1, CV_8UC1);
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
				desc.at<uchar>(window_size + i, (window_size - j)) = im.at<uchar>(y1, x1);

				//cv::circle(im,cv::Point(x1,y1), 1, 255);
				if (i > 0)
				{
					int x2 = round(x_center1 - i * dx);
					int y2 = round(y_center1 - i * dy);
					if (x2 < 0 || x2 >= im.cols || y2 < 0 || y2 >= im.rows)
						return false;

					desc.at<uchar>(window_size - i, (window_size - j)) = im.at<uchar>(y2, x2);
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

	bool EdgeModality::expendOri(float &max_score, const cv::Mat& im, cv::Point pt,
		double edge_dir, unsigned int window_size,float u,float v,float normal_u,float normal_v) const
	{
		double dx = cos(edge_dir);
		double dy = sin(edge_dir);

		double dx_k = cos(CV_PI / 2 + edge_dir);
		double dy_k = sin(CV_PI / 2 + edge_dir);
		cv::Mat desc;
		desc.create(2 * window_size + 1, 2 * window_size + 1, CV_32FC1); //CV_32FC1
		desc.setTo(0.0);
		/*cv::Mat im_test = im.clone();
		cv::Mat ii = im.clone();
		cv::cvtColor(im_test, im_test,CV_BGR2GRAY);*/
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
				desc.at<float>(window_size + i, (window_size - j)) = im.at<float>(y1, x1);

				//cv::circle(ii,cv::Point(x1,y1), 1, 255);
				if (i > 0)
				{
					int x2 = round(x_center1 - i * dx);
					int y2 = round(y_center1 - i * dy);
					if (x2 < 0 || x2 >= im.cols || y2 < 0 || y2 >= im.rows)
						return false;

					desc.at<float>(window_size - i, (window_size - j)) = im.at<float>(y2, x2);
					//cv::circle(ii, cv::Point(x2, y2), 1, 255);
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
					desc.at<float>((window_size + i), (window_size + j)) = im.at<float>(y1, x1);
					//cv::circle(ii, cv::Point(x1, y1), 1, 255);
					if (i > 0)
					{
						int x2 = round(x_center2 - i * dx);
						int y2 = round(y_center2 - i * dy);
						if (x2 < 0 || x2 >= im.cols || y2 < 0 || y2 >= im.rows)
							return false;

						desc.at<float>((window_size - i), (window_size + j)) = im.at<float>(y2, x2);
						//cv::circle(ii, cv::Point(x2, y2), 1, 255);
					}
				}
				/*cv::imshow("im", im);
				cv::imshow("desc", desc);
				cv::imwrite("desc.jpg", desc);
				cv::waitKey(0);*/
			}
		}
		/*cv::imshow("im", ii);
		cv::imshow("desc", desc);
		cv::imwrite("desc.jpg", desc);
		cv::waitKey(0);*/
		//计算得分最大值
		float max_sore_temp = 0;
		std::vector<float> score_list;
		std::vector<float> w_kernel = {0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.1};
		for (int i = 0;i < desc.rows;i++)
		{
			for (int j = 0;j < desc.cols;j++)
			{
				float angle_real = desc.at<float>(i, j);
				//cout<< angle_real <<endl;
				angle_real = angle_real * 180 / CV_PI;
				if (angle_real > 180)
					angle_real = angle_real - 360.0;
				float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
				float distance_angle = (angle_real - angle_virtual);
				float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
				/*if (score_match > max_sore_temp)
				{
					max_sore_temp = score_match;
				}*/
				max_sore_temp += score_match * w_kernel[i+j];
				//score_list.push_back(score_match);
			}
		}
		max_score = max_sore_temp;
		return true;
	}

	//建立高斯混合模型
	void EdgeModality::GMMCal(cv::Mat frame)
	{		
		fgMask.create(cv::Size{ frame.rows, frame.cols }, CV_8UC1);
		fgMask.setTo(cv::Scalar{ 0 });
		cv::Mat fgBackImg;
		try {
			mog2->apply(frame, fgMask);
			mog2->getBackgroundImage(fgBackImg);
		}
		catch (cv::Exception& e)
		{
			const char* msg_e = e.what();

			cout << msg_e << endl;
		}
	    cv::imshow("fgMask", fgMask);
		cv::imshow("fgBackImg", fgBackImg);
		cv::imwrite("frame.jpg", frame);
	    cv::imwrite("fgMask.jpg", fgMask);
		cv::waitKey(0);
	}
	float EdgeModality::TukeyNorm(float error) {
		if (std::abs(error) <= tukey_norm_constant_)
			return powf(tukey_norm_constant_, 2.0f) / 6.0f *
			(1.0f - powf(1.0f - powf(error / tukey_norm_constant_, 2.0f), 3.0f));
		else
			return powf(tukey_norm_constant_, 2.0f) / 6.0f;
	}
}  // namespace srt3d
