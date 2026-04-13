// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#include <srt3d/region_modality.h>
#include <opencv2/imgproc/types_c.h>
#include <srt3d/lap.h>
#include <opencv2/shape/shape_distance.hpp>
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
#include <opencv2/shape/shape_distance.hpp>、
namespace srt3d {

	RegionModality::RegionModality(const std::string &name,
		std::shared_ptr<Body> body_ptr,
		std::shared_ptr<Model> model_ptr,
		std::shared_ptr<Camera> camera_ptr)
		: name_{ name },
		body_ptr_{ std::move(body_ptr) },
		model_ptr_{ std::move(model_ptr) },
		camera_ptr_{ std::move(camera_ptr) } {
		tikhonov_matrix_.setZero();
		tikhonov_matrix_.diagonal().head<3>().array() = tikhonov_parameter_rotation_;
		tikhonov_matrix_.diagonal().tail<3>().array() =
			tikhonov_parameter_translation_;
	}

	bool RegionModality::SetUp() {
		set_up_ = false;

		// Check if all required objects are set up
		if (!model_ptr_->set_up()) {
			std::cerr << "Model " << model_ptr_->name() << " was not set up"
				<< std::endl;
			return false;
		}
		if (!camera_ptr_->set_up()) {
			std::cerr << "Camera " << camera_ptr_->name() << " was not set up"
				<< std::endl;
			return false;
		}
		if (use_occlusion_handling_ && !occlusion_renderer_ptr_->set_up()) {
			std::cerr << "Occlusion renderer " << occlusion_renderer_ptr_->name()
				<< " was not set up" << std::endl;
			return false;
		}

		PrecalculateFunctionLookup();
		PrecalculateDistributionVariables();
		PrecalculateHistogramBinVariables();
		PrecalculateBodyVariables();
		PrecalculateCameraVariables();
		SetImshowVariables();

		set_up_ = true;
		return true;
	}

	void RegionModality::set_n_lines(int n_lines) { n_lines_ = n_lines; }

	void RegionModality::set_function_amplitude(float function_amplitude) {
		function_amplitude_ = function_amplitude;
		set_up_ = false;
	}

	void RegionModality::set_function_slope(float function_slope) {
		function_slope_ = function_slope;
		set_up_ = false;
	}

	void RegionModality::set_learning_rate(float learning_rate) {
		learning_rate_ = learning_rate;
	}

	void RegionModality::set_function_length(int function_length) {
		function_length_ = function_length;
		set_up_ = false;
	}

	void RegionModality::set_distribution_length(int distribution_length) {
		distribution_length_ = distribution_length;
		set_up_ = false;
	}

	void RegionModality::set_scales(const std::vector<int> &scales) {
		scales_ = scales;
	}

	void RegionModality::set_n_newton_iterations(int n_newton_iterations) {
		n_newton_iterations_ = n_newton_iterations;
	}

	void RegionModality::set_min_continuous_distance(
		float min_continuous_distance) {
		min_continuous_distance_ = min_continuous_distance;
	}

	bool RegionModality::set_n_histogram_bins(int n_histogram_bins) {
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

	void RegionModality::set_learning_rate_f(float learning_rate_f) {
		learning_rate_f_ = learning_rate_f;
	}

	void RegionModality::set_learning_rate_b(float learning_rate_b) {
		learning_rate_b_ = learning_rate_b;
	}

	void RegionModality::set_unconsidered_line_length(
		float unconsidered_line_length) {
		unconsidered_line_length_ = unconsidered_line_length;
	}

	void RegionModality::set_considered_line_length(float considered_line_length) {
		considered_line_length_ = considered_line_length;
	}

	void RegionModality::set_tikhonov_parameter_rotation(
		float tikhonov_parameter_rotation) {
		tikhonov_parameter_rotation_ = tikhonov_parameter_rotation;
		tikhonov_matrix_.diagonal().head<3>().array() = tikhonov_parameter_rotation_;
	}

	void RegionModality::set_tikhonov_parameter_translation(
		float tikhonov_parameter_translation) {
		tikhonov_parameter_translation_ = tikhonov_parameter_translation;
		tikhonov_matrix_.diagonal().tail<3>().array() =
			tikhonov_parameter_translation_;
	}

	void RegionModality::UseOcclusionHandling(
		std::shared_ptr<OcclusionRenderer> occlusion_renderer_ptr) {
		occlusion_renderer_ptr_ = std::move(occlusion_renderer_ptr);
		use_occlusion_handling_ = true;
		set_up_ = false;
	}

	void RegionModality::DoNotUseOcclusionHandling() {
		occlusion_renderer_ptr_ = nullptr;
		use_occlusion_handling_ = false;
		set_up_ = false;
	}

	void RegionModality::set_display_visualization(bool display_visualization) {
		display_visualization_ = display_visualization;
	}

	void RegionModality::StartSavingVisualizations(
		const std::experimental::filesystem::path &save_directory) {
		save_visualizations_ = true;
		save_directory_ = save_directory;
	}

	void RegionModality::StopSavingVisualizations() {
		save_visualizations_ = false;
	}

	void RegionModality::set_visualize_lines_correspondence(
		bool visualize_lines_correspondence) {
		visualize_lines_correspondence_ = visualize_lines_correspondence;
		SetImshowVariables();
	}

	void RegionModality::set_visualize_points_occlusion_mask_correspondence(
		bool visualize_points_occlusion_mask_correspondence) {
		visualize_points_occlusion_mask_correspondence_ =
			visualize_points_occlusion_mask_correspondence;
		SetImshowVariables();
	}

	void RegionModality::set_visualize_points_pose_update(
		bool visualize_points_pose_update) {
		visualize_points_pose_update_ = visualize_points_pose_update;
		SetImshowVariables();
	}

	void RegionModality::set_visualize_points_histogram_image_pose_update(
		bool visualize_points_histogram_image_pose_update) {
		visualize_points_histogram_image_pose_update_ =
			visualize_points_histogram_image_pose_update;
		SetImshowVariables();
	}

	void RegionModality::set_visualize_points_result(bool visualize_points_result) {
		visualize_points_result_ = visualize_points_result;
		SetImshowVariables();
	}

	void RegionModality::set_visualize_points_histogram_image_result(
		bool visualize_points_histogram_image_result) {
		visualize_points_histogram_image_result_ =
			visualize_points_histogram_image_result;
		SetImshowVariables();
	}

	bool RegionModality::StartModality() {
		if (!IsSetup()) return false;

		// Initialize histogramsn_newton_iterations_
		PrecalculatePoseVariables();
		AddLinePixelColorsToTempHistograms();
		int flag_1 = 0;
		int flag_2 = 0;
		if (CalculateHistogram(1.0f, temp_histogram_f_, &histogram_f_) &&
			CalculateHistogram(1.0f, temp_histogram_b_, &histogram_b_)) {
			flag_1 = 1;
			//return true;
		}
		else {
			std::cerr << "Histograms could not be initialised for modality " << name_
				<< std::endl;
			return false;
		}
		//计算局部的颜色直方图
		if (local_flag_)
		{
			AddLinePixelColorsToTempHistograms_local();
			/*for (int i = 0; i < num_local_hist; i++)
			{
				std::fill(begin(histogram_f_local_[i]), end(histogram_f_local_[i]), 0.0f);
				std::fill(begin(histogram_b_local_[i]), end(histogram_b_local_[i]), 0.0f);
			}*/
			for (int i = 0; i < num_local_hist; i++)
			{
				if (CalculateHistogram(1.0, temp_histogram_f_local_[i], &histogram_f_local_[i]) &&
					CalculateHistogram(1.0, temp_histogram_b_local_[i], &histogram_b_local_[i]))
					//
					int flag_2 = 1;
				//return true;
				else
					std::cerr << "Local Histograms could not be initialised for modality " << name_
					<< std::endl;
			}
		}
		if (flag_1 && flag_2)
		{
			return true;
		}
	}

	bool RegionModality::CalculateBeforeCameraUpdate() {
		if (!IsSetup()) return false;
		//cout<<"CalculateBeforeCameraUpdate"<<endl;
		PrecalculatePoseVariables();

		/*应该跟踪成功才执行这一步??*/

		AddLinePixelColorsToTempHistograms();
		CalculateHistogram(learning_rate_f_, temp_histogram_f_, &histogram_f_);
		CalculateHistogram(learning_rate_b_, temp_histogram_b_, &histogram_b_);

		if (local_flag_)
		{
			/*局部多区域进行重置*/
			//cout<< local_flag_ <<endl;
			AddLinePixelColorsToTempHistograms_local();
			/*for (int i = 0; i < num_local_hist; i++)
			{
				std::fill(begin(histogram_f_local_[i]), end(histogram_f_local_[i]), 0.0f);
				std::fill(begin(histogram_b_local_[i]), end(histogram_b_local_[i]), 0.0f);
			} */
			for (int i = 0; i < num_local_hist; i++)
			{
				CalculateHistogram(learning_rate_f_local_, temp_histogram_f_local_[i], &histogram_f_local_[i]);
				CalculateHistogram(learning_rate_b_local_, temp_histogram_b_local_[i], &histogram_b_local_[i]);
			}
		}

		return true;
	}

	bool RegionModality::CalculateCorrespondences(int corr_iteration) {
		if (!IsSetup()) return false;

		float clamped_x = (corr_iteration < 1) ? 1 : (corr_iteration > 6) ? 6 : corr_iteration;
		//float weight = 0.55 * std::exp(-1.5*(clamped_x - 1));
		float weight = 0.55 * std::exp(-1.5*(clamped_x - 1)) + 0.35;

		g_l_ratio = weight;  //0.55

		//cout << "CalculateCorrespondences" << endl;
		/*float length_ = LastValidValue(distribution_length_vector_, corr_iteration);
		distribution_length_ = length_;
		PrecalculateDistributionVariables();*/

		float turkey_noram_real_ = LastValidValue(tukey_norm_constant_vector_, corr_iteration);
		tukey_norm_constant_ = turkey_noram_real_;

		//PrecalculateExtractEdge();
		PrecalculatePoseVariables();
		PrecalculateScaleDependentVariables(corr_iteration);
		//=========计算并更新颜色直方图=============
		/*if (0)
		{
			AddLinePixelColorsToTempHistograms_local();
			for (int i = 0; i < num_local_hist; i++)
			{
				CalculateHistogram(learning_rate_f_local_, temp_histogram_f_local_[i], &histogram_f_local_[i]);
				CalculateHistogram(learning_rate_b_local_, temp_histogram_b_local_[i], &histogram_b_local_[i]);
			}
		}*/

		//=======================
		function_lookup_edge_distrbution_all.resize(distribution_length_);
		float scale_function = LastValidValue(sigma_global_, corr_iteration);
		//cout<< scale_function <<endl;
		for (int i = 0; i < distribution_length_; ++i) {
			float x = float(i) - float(distribution_length_ - 1) / 2.0f;
			function_lookup_edge_distrbution_all[i] = 1 / (2.506628 * scale_function)  * std::exp(-(pow(x - mu_, 2) / (2 * scale_function * scale_function)));
			//cout << function_lookup_edge_distrbution_all[i] << endl;
		}

		//function_lookup_f_.resize(distribution_length_);
		//function_lookup_b_.resize(distribution_length_);
		//for (int i = 0; i < distribution_length_; ++i) {
		//	float x = float(i) - float(distribution_length_ - 1) / 2.0f;
		//	if (function_slope_ == 0.0f)
		//		function_lookup_f_[i] =
		//		0.5f - function_amplitude_ * ((0.0f < x) - (x < 0.0f));
		//	else
		//	{
		//		function_lookup_f_[i] =
		//			0.5f - function_amplitude_ * std::tanh(x / (2.0f * function_slope_));

		//		//cout << function_lookup_f_[i] << endl;
		//	}
		//		
		//	function_lookup_b_[i] = 1.0f - function_lookup_f_[i];
		//}


		if (use_occlusion_handling_) occlusion_renderer_ptr_->FetchOcclusionMask();

		// Search closest template view
		const Model::TemplateView *template_view;
		//cout<< body2camera_pose_.matrix() <<endl;
		model_ptr_->GetClosestTemplateView(body2camera_pose_, &template_view);

		/*for 前后帧3D点的匹配*/
		cv::Mat1f searchLines_Center(cv::Size(3, n_lines_));
		cv::Mat1f searchLines_Center_LastFrame(cv::Size(3, n_lines_));

		cv::flann::Index  knn;
		cv::Mat1i indices;
		cv::Mat1f dists;
		if (local_flag_)
		{
			int iii = 0;
			for (auto data_point = begin(template_view->data_points);
				data_point != begin(template_view->data_points) + n_lines_;
				++data_point)
			{
				memcpy(searchLines_Center.ptr(iii), &data_point->center_f_body, sizeof(float) * 3);
				memcpy(searchLines_Center_LastFrame.ptr(iii), &lastFrame_CenterPoints_[iii], sizeof(float) * 3);
				iii++;
			}

			//建立knn索引
			//knn.build(searchLines_Center_LastFrame, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_L2);
			//如何加速
			//cv::flann::KDTreeIndexParams(1);
			//knn.knnSearch(searchLines_Center, indices, dists, 1, cv::flann::SearchParams(32));
		}

		// Iterate over n_lines
		std::vector<float> segment_probabilities_f(line_length_in_segments_);
		std::vector<float> segment_probabilities_b(line_length_in_segments_);
		data_lines_.clear();
		int datePoint_index = 0;
//#pragma omp parallel for
		for (auto data_point = begin(template_view->data_points);
			data_point != begin(template_view->data_points) + n_lines_;
			++data_point) {
			/*for local region*/
			int his_index = 0;
			if (local_flag_)
			{
				//int index_for_match_lastFrame_ = indices.at<int>(datePoint_index);
				//his_index = index_for_match_lastFrame_ / (n_lines_ / num_local_hist);
				his_index = datePoint_index / (n_lines_ / num_local_hist);
				flag_match_3d_point_ = 1;
				//float dis = dists.at<float>(datePoint_index);
				/*if (dis * 1000 > 0.3)
				{
					flag_match_3d_point_ = 0;
				}
				else
				{
					flag_match_3d_point_ = 1;
				}*/
			}


			DataLine data_line;
			CalculateBasicLineData(*data_point, &data_line);
			if (!IsLineValid(data_line.center_u, data_line.center_v,
				data_line.continuous_distance))
			{
				datePoint_index++;
				continue;
			}
			if (!CalculateSegmentProbabilities(
				data_line.center_u, data_line.center_v, data_line.normal_u,
				data_line.normal_v, &segment_probabilities_f,
				&segment_probabilities_b, &data_line.normal_component_to_scale,
				&data_line.delta_r, his_index))
			{
				datePoint_index++;
				continue;
			}
			CalculateDistribution(segment_probabilities_f, segment_probabilities_b,
				&data_line.distribution);

			CalculateDistributionMoments(data_line.distribution, &data_line.mean,
				&data_line.standard_deviation,
				&data_line.variance);
			data_lines_.push_back(std::move(data_line));
			datePoint_index++;
		}

		return true;
	}

	bool RegionModality::VisualizeCorrespondences(int save_idx) {
		if (!IsSetup()) return false;

		if (visualize_lines_correspondence_)
			VisualizeLines("lines_correspondence", save_idx);
		if (visualize_points_occlusion_mask_correspondence_ &&
			use_occlusion_handling_)
			VisualizePointsOcclusionMask("occlusion_mask_correspondence", save_idx);
		return true;
	}

	bool RegionModality::CalculatePoseUpdate(int corr_iteration,
		int update_iteration) {
		if (!IsSetup()) return false;
		//cout<< "CalculatePoseUpdate" <<endl;
		PrecalculatePoseVariables();

		////=========计算并更新颜色直方图=============
		//AddLinePixelColorsToTempHistograms_local();
		//for (int i = 0; i < num_local_hist; i++)
		//{
		//	CalculateHistogram(learning_rate_f_local_, temp_histogram_f_local_[i], &histogram_f_local_[i]);
		//	CalculateHistogram(learning_rate_b_local_, temp_histogram_b_local_[i], &histogram_b_local_[i]);
		//}
		//=======================

		gradient.setZero();
		hessian.setZero();

		// Iterate over correspondence lines
		float error_count = 0.0f;
		error_count_ = 0.0f;
		float nerr = 0.0f;
		int debug = 0;
		cv::Mat img;
		if (debug)
		{
			img = camera_ptr_->image().clone();
		}

		vector <cv::Point> contourPts1;
		vector <cv::Point> contourPts2;
		int match_num = 0;
		for (auto &data_line : data_lines_) {
			// Calculate point coordinates in camera frame
			data_line.center_f_camera = body2camera_pose_ * data_line.center_f_body;
			//估计的轮廓？？？
			float x = data_line.center_f_camera(0);
			float y = data_line.center_f_camera(1);
			float z = data_line.center_f_camera(2);

			float z2 = z * z;

			// Calculate delta_cs
			float fu_z = fu_ / z;
			float fv_z = fv_ / z;
			//
			float xfu_z = x * fu_z;
			float yfv_z = y * fv_z;
			//意义：dsi
			// normal_component_to_scale   ni/s
			float delta_cs = (data_line.normal_u * (xfu_z + ppu_ - data_line.center_u) +
				data_line.normal_v * (yfv_z + ppv_ - data_line.center_v) -
				data_line.delta_r) *
				data_line.normal_component_to_scale;

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
					//cout << prob << endl;
					if (prob > max_prob)
					{
						max_prob = prob;
						x_ = u;
						y_ = v;
					}
				}
			}
			//cout << "===end===" << endl;
			//cout << max_prob << endl;
			//匹配成功
			if (max_prob >= 0.35) //0.4 0.15 * scale_ 0.5   0.35
			{
				match_num++;
			}
			/*if (max_prob <= 0.35)
			{
				continue;
			}*/
			contourPts1.push_back(cv::Point(x_, y_));
			contourPts2.push_back(cv::Point(data_line.center_u, data_line.center_v));

			if (debug)
			{
				/*cv::circle(img, cv::Point(x_, y_), 2, cv::Scalar(0, 255, 255), -1);
				cv::circle(img, cv::Point(data_line.center_u, data_line.center_v), 2, cv::Scalar(0, 0, 255), -1);*/
				img.at<cv::Vec3b>(data_line.center_v, data_line.center_u) = cv::Vec3b(0, 0, 255);
				img.at<cv::Vec3b>(y_, x_) = cv::Vec3b(0, 255, 255);
			}
			/*turkey计算残差*/
#if 0
			Eigen::Vector2f diff{ (data_line.center_u - x_), (data_line.center_v - y_) };
			float squared_error = diff.squaredNorm();
			float error = sqrtf(squared_error);
			float weight = 1.0f / variance_;
			if (error > std::numeric_limits<float>::min())
				weight = (TukeyNorm(error) / squared_error) / variance_;

			error_count += fabs(error) * weight;
			nerr += weight;
#endif
			/*基于最大概率点的计算*/
#if 0
			// Calculate derivatives
			Eigen::MatrixXf  dx_dX(2, 3);
			dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z, -y * fv_ / z2;
			Eigen::MatrixXf dx_dtranslation{ dx_dX * body2camera_rotation_ };
			Eigen::MatrixXf dx_dtheta(2, 6);
			dx_dtheta << -dx_dtranslation * Vector2Skewsymmetric(data_line.center_f_body),
				dx_dtranslation;

			// Calculate gradient and hessian
			gradient -= (weight  * diff.transpose()) * dx_dtheta;
			hessian.triangularView<Eigen::Lower>() -=
				(weight * dx_dtheta.transpose()) * dx_dtheta;
#endif

			/*基于多峰统计数据分布的高斯牛顿优化雅可比矩阵计算*/
#if 1
			//计算误差结束===============================
			// Calculate first derivative of loglikelihood with respect to delta_cs
			error_count = error_count + fabs((data_line.mean) * scale_);
			//cout << data_line.mean << endl;
			//cout << error_count << endl;
			float dloglikelihood_ddelta_cs;
			if (update_iteration < n_newton_iterations_) {
				dloglikelihood_ddelta_cs =
					(data_line.mean - delta_cs) / (data_line.variance);

				//cout<<"global:"<< dloglikelihood_ddelta_cs <<endl;
			}
			else {
				//更加精确的计算？？？
			  // Calculate distribution indexes
			  // Note: (distribution_length - 1) / 2 + 1 = (distribution_length + 1) / 2
				int dist_idx_upper = int(delta_cs + distribution_length_plus_1_half_);
				int dist_idx_lower = dist_idx_upper - 1;
				if (dist_idx_lower < 0 || dist_idx_upper >= distribution_length_)
					continue;

				//加入学习率
				dloglikelihood_ddelta_cs =
					(std::log(data_line.distribution[dist_idx_upper]) -
						std::log(data_line.distribution[dist_idx_lower])) *
					learning_rate_ / (data_line.variance);
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

			// 权重设计
			/*float weight = min_variance_ /
				(data_line.normal_component_to_scale * data_line.normal_component_to_scale * variance_);*/
			float weight = min_variance_ / (variance_);

			//cout<< weight <<endl;

			gradient += dloglikelihood_ddelta_cs * ddelta_cs_dtheta.transpose();

			//cout << "gradient" << gradient << endl;
			//除以标准差

			ddelta_cs_dtheta /= data_line.standard_deviation;
			//ddelta_cs_dtheta /= data_line.variance;
			//计算hessian矩阵
			//triangularView<Eigen::Lower>()下三角矩阵，其他部分为0
			hessian.triangularView<Eigen::Lower>() -= 
				ddelta_cs_dtheta.transpose() * ddelta_cs_dtheta;
#endif
		}
		//特征误差计算
		hessian = hessian.selfadjointView<Eigen::Lower>();
		error_count_ = error_count / data_lines_.size();  // / (scale_ * distribution_length_)
		//cv::Ptr<cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();
		//float dis = mysc->computeDistance(contourPts1, contourPts2);
		shape_cost_ = 0;
		match_ratio_ = float(match_num) / data_lines_.size();
		//cout<< error_count_ <<endl;
		//cout << "region error_count_: " << error_count_ << endl;
		//error_count_ = error_count /(nerr * match_num);
		//cout<<"================"<<endl;
		//cout<<"region error_count_: "<< error_count_ <<endl;
		//cout<< float(match_num)/data_lines_.size() <<endl;
		//CalShape(contourPts1, contourPts2);
		//cout << "================" << endl;		
		//cout<<"scale_region"<<scale_<<endl;
		//cout << "error_count region: " << error_count / data_lines_.size() << endl;	
		if (debug)
		{
			cv::namedWindow("img", 0);
			cv::imshow("img", img);
			cv::waitKey(0);
		}

		// Optimize and update pose
		if (0)
		{
			Eigen::FullPivLU<Eigen::Matrix<float, 6, 6>> lu{ tikhonov_matrix_ - hessian };
			if (lu.isInvertible()) {
				//theta为六维位姿变化量
				Eigen::Matrix<float, 6, 1> theta{ lu.solve(gradient) };
				Transform3fA pose_variation{ Transform3fA::Identity() };
				//Vector2Skewsymmetric对称矩阵
				pose_variation.rotate(Vector2Skewsymmetric(theta.head<3>()).exp());
				pose_variation.translate(theta.tail<3>());
				body_ptr_->set_body2world_pose(body_ptr_->body2world_pose() *
					pose_variation);
			}
		}
		return true;
	}
	void RegionModality::CalShape(std::vector<cv::Point> contourPts1, std::vector<cv::Point> contourPts2)
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
		float cost_ = cost / size1;
		shape_cost_ = cost_;
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
				cv::circle(resimg, p1, 1, cv::Scalar(0, 0, 255), 1, 8);
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
		///基于形状的匹配率
		match_ratio = float(link1.size()) / size;
		//cout <<"match_ratio："<< match_ratio << endl;
		/*cv::imshow("mapping", resimg);
		cv::imshow("map_clear", resimg2);
		cv::waitKey(0);*/
	}

	bool RegionModality::VisualizePoseUpdate(int save_idx) {
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

	bool RegionModality::VisualizeResults(int save_idx) {
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

	const std::string &RegionModality::name() const { return name_; }

	std::shared_ptr<Body> RegionModality::body_ptr() const { return body_ptr_; }

	std::shared_ptr<Model> RegionModality::model_ptr() const { return model_ptr_; }

	std::shared_ptr<Camera> RegionModality::camera_ptr() const {
		return camera_ptr_;
	}

	std::shared_ptr<OcclusionRenderer> RegionModality::occlusion_renderer_ptr()
		const {
		return occlusion_renderer_ptr_;
	}

	bool RegionModality::imshow_correspondence() const {
		return imshow_correspondence_;
	}

	bool RegionModality::imshow_pose_update() const { return imshow_pose_update_; }

	bool RegionModality::imshow_result() const { return imshow_result_; }

	bool RegionModality::set_up() const { return set_up_; }

	void RegionModality::PrecalculateFunctionLookup() {
		function_lookup_f_.resize(function_length_);
		function_lookup_b_.resize(function_length_);
		for (int i = 0; i < function_length_; ++i) {
			float x = float(i) - float(function_length_ - 1) / 2.0f;
			if (function_slope_ == 0.0f)
			{
				function_lookup_f_[i] =
					0.5f - function_amplitude_ * ((0.0f < x) - (x < 0.0f));
			}
			else
			{
				function_lookup_f_[i] =
					0.5f - function_amplitude_ * std::tanh(x / (2.0f * function_slope_));
			}

			function_lookup_b_[i] = 1.0f - function_lookup_f_[i];
		}
	}

	void RegionModality::PrecalculateDistributionVariables() {
		line_length_in_segments_ = function_length_ + distribution_length_ - 1;
		distribution_length_minus_1_half_ =
			(float(distribution_length_) - 1.0f) / 2.0f;
		distribution_length_plus_1_half_ =
			(float(distribution_length_) + 1.0f) / 2.0f;
		float min_variance_laplace =
			1.0f / (2.0f * powf(std::atanhf(2.0f * function_amplitude_), 2.0f));
		float min_variance_gaussian = function_slope_;
		min_variance_ = std::max(min_variance_laplace, min_variance_gaussian);
	}

	void RegionModality::PrecalculateHistogramBinVariables() {
		n_histogram_bins_squared_ = pow_int(n_histogram_bins_, 2);
		//std::cout << "n_histogram_bins_squared_:" << n_histogram_bins_squared_ << std::endl;
		n_histogram_bins_cubed_ = pow_int(n_histogram_bins_, 3);
		//std::cout << "n_histogram_bins_cubed_:" << n_histogram_bins_cubed_ << std::endl;
		temp_histogram_f_.resize(n_histogram_bins_cubed_);
		temp_histogram_b_.resize(n_histogram_bins_cubed_);
		histogram_f_.resize(n_histogram_bins_cubed_);
		histogram_b_.resize(n_histogram_bins_cubed_);

		temp_histogram_f_local_.resize(num_local_hist);
		temp_histogram_b_local_.resize(num_local_hist);
		histogram_f_local_.resize(num_local_hist);
		histogram_b_local_.resize(num_local_hist);
		for (int i = 0; i < num_local_hist; i++)
		{
			temp_histogram_f_local_[i].resize(n_histogram_bins_cubed_);
			temp_histogram_b_local_[i].resize(n_histogram_bins_cubed_);
			histogram_f_local_[i].resize(n_histogram_bins_cubed_);
			histogram_b_local_[i].resize(n_histogram_bins_cubed_);
		}
		//cout << "PrecalculateHistogramBinVariables over" << endl;
	}

	void RegionModality::SetImshowVariables() {
		imshow_correspondence_ = visualize_lines_correspondence_ ||
			(visualize_points_occlusion_mask_correspondence_ &&
				use_occlusion_handling_);
		imshow_pose_update_ = visualize_points_pose_update_ ||
			visualize_points_histogram_image_pose_update_;
		imshow_result_ =
			visualize_points_result_ || visualize_points_histogram_image_result_;
	}

	void RegionModality::PrecalculateBodyVariables() {
		if (use_occlusion_handling_)
			encoded_occlusion_id_ = (uchar(1) << unsigned(body_ptr_->occlusion_id()));
	}

	void RegionModality::PrecalculateCameraVariables() {
		fu_ = camera_ptr_->intrinsics().fu;
		fv_ = camera_ptr_->intrinsics().fv;
		ppu_ = camera_ptr_->intrinsics().ppu;
		ppv_ = camera_ptr_->intrinsics().ppv;
		image_width_minus_1_ = camera_ptr_->image().cols - 1;
		image_height_minus_1_ = camera_ptr_->image().rows - 1;
		image_width_minus_2_ = camera_ptr_->image().cols - 2;
		image_height_minus_2_ = camera_ptr_->image().rows - 2;
	}

	void RegionModality::PrecalculatePoseVariables() {
		body2camera_pose_ =
			camera_ptr_->world2camera_pose() * body_ptr_->body2world_pose();
		//cout<< "PrecalculatePoseVariables"<< body_ptr_->body2world_pose().matrix()<< endl;
		body2camera_rotation_ = body2camera_pose_.rotation().matrix();
		body2camera_rotation_xy_ = body2camera_rotation_.topRows<2>();
	}

	void RegionModality::PrecalculateScaleDependentVariables(int corr_iteration) {
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

	void RegionModality::AddLinePixelColorsToTempHistograms() {
		//cv::Mat show_use = camera_ptr_->image().clone();
		const cv::Mat &image{ camera_ptr_->image() };
		const Model::TemplateView *template_view;
		model_ptr_->GetClosestTemplateView(body2camera_pose_, &template_view);

		// Iterate over all points
		std::fill(begin(temp_histogram_f_), end(temp_histogram_f_), 0.0f);
		std::fill(begin(temp_histogram_b_), end(temp_histogram_b_), 0.0f);

		//cv::Mat temp_img_search_lines_f = cv::Mat::zeros(cv::Size(int(considered_line_length_ + 0.5), n_lines_),  CV_8UC3);
		//cv::Mat temp_img_search_lines_b = cv::Mat::zeros(cv::Size(int(considered_line_length_ + 0.5),n_lines_), CV_8UC3);
		//cv::Mat temp_img_search_lines_fb = cv::Mat::zeros(cv::Size(int(considered_line_length_ + 0.5) * 2, n_lines_), CV_8UC3);
		int count_lines = 0;	

		for (auto data_point = begin(template_view->data_points);
			data_point != begin(template_view->data_points) + n_lines_;
			++data_point) {
			// Project point data in camera frame
			//center_f_body为模型上的点===
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

			//变换法线方向一定夹角
			/*float theta = atan2(normal(0), normal(1));
			float theta1 = theta + 0.07;
			normal(0) = sin(theta1);
			normal(1) = cos(theta1);*/
			
			for (int i = 0; i < 1; i++)
			{
				int length = 0;
				// Iterate over foreground pixels
				float u = center(0) - normal(0) * unconsidered_line_length_ + 0.5f;
				float v = center(1) - normal(1) * unconsidered_line_length_ + 0.5f;
				//show_use.at<cv::Vec3b>(int(center(1)), int(center(0))) = cv::Vec3b(0, 255, 0);
				int n_iteration =
					int(std::fmin(foreground_distance - 2.0f * unconsidered_line_length_,
						considered_line_length_) +
						0.5f);
				for (int i = 0; i < n_iteration; ++i) {
					if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
						int(v) > image_height_minus_1_)
						break;

					//show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);
					//temp_img_search_lines_f.at<cv::Vec3b>(count_lines, i) = image.at<cv::Vec3b>(int(v), int(u));
					//temp_img_search_lines_fb.at<cv::Vec3b>(count_lines, length) = image.at<cv::Vec3b>(int(v), int(u));
					length++;
					AddPixelColorToHistogram(image.at<cv::Vec3b>(int(v), int(u)),
						&temp_histogram_f_);
					u -= normal(0);
					v -= normal(1);
				}

				// Iterate over background pixels
				u = center(0) + normal(0) * unconsidered_line_length_ + 0.5f;
				v = center(1) + normal(1) * unconsidered_line_length_ + 0.5f;
				//show_use.at<cv::Vec3b>(int(center(1)), int(center(0))) = cv::Vec3b(0, 255, 0);
				n_iteration =
					int(std::fmin(background_distance - 2.0f * unconsidered_line_length_,
						considered_line_length_) +
						0.5f);
				for (int i = 0; i < n_iteration; ++i) {
					if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
						int(v) > image_height_minus_1_)
						break;

					//show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);
					//temp_img_search_lines_b.at<cv::Vec3b>(count_lines, i) = image.at<cv::Vec3b>(int(v), int(u));
					//temp_img_search_lines_fb.at<cv::Vec3b>(count_lines, length) = image.at<cv::Vec3b>(int(v), int(u));
					length++;
					AddPixelColorToHistogram(image.at<cv::Vec3b>(int(v), int(u)),
						&temp_histogram_b_);
					u += normal(0);
					v += normal(1);
				}			
				/*theta1 = theta1 - 0.035;
				normal(0) = sin(theta1);
				normal(1) = cos(theta1);*/
			}
			count_lines++;
		}
		

        #if 0 
		// 定义直方图参数
		int histSize[] = { 256, 256, 256 };
		float range[] = { 0, 256 };
		const float* histRange[] = { range, range, range };
		bool uniform = true, accumulate = false;

		vector<cv::Mat> rgb_channel_f;
		cv::split(temp_img_search_lines_f, rgb_channel_f);
		vector<cv::Mat> rgb_channel_b;
		cv::split(temp_img_search_lines_b, rgb_channel_b);
		// 计算直方图
		cv::Mat hist1, hist2;
		cv::calcHist(&rgb_channel_f[0], 1, 0, cv::Mat(), hist1, 1, histSize, histRange, uniform, accumulate);
		cv::calcHist(&rgb_channel_b[0], 1, 0, cv::Mat(), hist2, 1, histSize, histRange, uniform, accumulate);
		// 归一化直方图
		normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		// 计算Bhattacharyya系数
		double bhatta = compareHist(hist1, hist2, CV_COMP_BHATTACHARYYA);
        #endif
		// 输出结果
        #if 0
		cv::imshow("temp_img_search_lines_f", temp_img_search_lines_f);
		cv::imshow("temp_img_search_lines_b", temp_img_search_lines_b);
		cv::imshow("temp_img_search_lines_fb", temp_img_search_lines_fb);
		cv::Mat hist1 = cv::Mat::zeros(cv::Size(temp_histogram_f_.size(), int(1)), CV_32F);
		cv::Mat hist2 = cv::Mat::zeros(cv::Size(temp_histogram_b_.size(), int(1)), CV_32F);
		for (int i = 0;i < temp_histogram_f_.size();i++)
		{
			hist1.at<float>(0, i) = temp_histogram_f_[i];
			hist2.at<float>(0, i) = temp_histogram_b_[i];
			//cout<< temp_histogram_f_[i] <<endl;
		}
		// 归一化直方图
		normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

		double bhatta = compareHist(hist1, hist2, CV_COMP_BHATTACHARYYA);
		cout << "Bhattacharyya coefficient: " << 1 - bhatta << endl;

		cv::imshow("show_use", show_use);
		cv::namedWindow("show_use", 0);
		cv::waitKey(0);
        #endif
	}

	void RegionModality::AddLinePixelColorsToTempHistograms_local() {

		//cv::Mat show_use = camera_ptr_->image().clone();
		const cv::Mat &image{ camera_ptr_->image() };
		const Model::TemplateView *template_view;
		model_ptr_->GetClosestTemplateView(body2camera_pose_, &template_view);
		//===局部颜色直方图建立
		for (int i = 0; i < num_local_hist; i++)
		{
			std::fill(begin(temp_histogram_f_local_[i]), end(temp_histogram_f_local_[i]), 0.0f);
			std::fill(begin(temp_histogram_b_local_[i]), end(temp_histogram_b_local_[i]), 0.0f);
		}
		//调整遍历方式
		lastFrame_CenterPoints_.clear();
		for (int i = 0; i < n_lines_; i++)
		{
			//cout<<i<<endl;
			int index = i / (n_lines_ / num_local_hist);

			//cout<<"index: "<<index<<endl;
			const Model::PointData *data_point = &template_view->data_points[i];
			// Project point data in camera frame
			//center_f_body为模型上的点===
			Eigen::Vector3f center_f_camera{ body2camera_pose_ *
											data_point->center_f_body };

			lastFrame_CenterPoints_.push_back(data_point->center_f_body);

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
			float u = center(0) - normal(0) * unconsidered_line_length_local_ + 0.5f;
			float v = center(1) - normal(1) * unconsidered_line_length_local_ + 0.5f;
			int n_iteration =
				int(std::fmin(foreground_distance - 2.0f * unconsidered_line_length_local_,
					considered_line_length_local_) +
					0.5f);
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;

				/*if(index == 0)
					show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(129, 132, 225);
				if (index == 1)
					show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(194, 154, 246);
				if (index == 2)
					show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(119, 147, 119);
				if (index == 3)
					show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(157, 202, 131);*/

				AddPixelColorToHistogram(image.at<cv::Vec3b>(int(v), int(u)),
					&temp_histogram_f_local_[index]);

				u -= normal(0);
				v -= normal(1);
			}

			// Iterate over background pixels
			u = center(0) + normal(0) * unconsidered_line_length_local_ + 0.5f;
			v = center(1) + normal(1) * unconsidered_line_length_local_ + 0.5f;
			n_iteration =
				int(std::fmin(background_distance - 2.0f * unconsidered_line_length_local_,
					considered_line_length_local_) +
					0.5f);
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;

				/*if (index == 0)
					show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(129, 132, 225);
				if (index == 1)
					show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(194, 154, 246);
				if (index == 2)
					show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(119, 147, 119);
				if (index == 3)
					show_use.at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(157, 202, 131);*/

				AddPixelColorToHistogram(image.at<cv::Vec3b>(int(v), int(u)),
					&temp_histogram_b_local_[index]);
				u += normal(0);
				v += normal(1);
			}

		}
		//cv::imshow("show_use", show_use);
		//cv::imwrite("show_local.jpg", show_use);
		//cv::namedWindow("show_use", 0);
		//cv::waitKey(0);
	}

	void RegionModality::AddPixelColorToHistogram(
		const cv::Vec3b &pixel_color,
		std::vector<float> *enlarged_histogram) const {
		(*enlarged_histogram)[(pixel_color[0] >> histogram_bitshift_) *
			n_histogram_bins_squared_ +
			(pixel_color[1] >> histogram_bitshift_) *
			n_histogram_bins_ +
			(pixel_color[2] >> histogram_bitshift_)] += 1.0f;
	}

	bool RegionModality::CalculateHistogram(
		float learning_rate, const std::vector<float> &temp_histogram,
		std::vector<float> *histogram) {
		// Calculate sum for normalization
		float sum = 0.0f;
#ifndef _DEBUG
#pragma omp simd
#endif
		//对颜色直方图求和
		for (int i = 0; i < n_histogram_bins_cubed_; i++) {
			sum += temp_histogram[i];
			//cout << (temp_histogram)[i] << endl;
		}
		//cout<<"==============================="<<endl;
		if (!sum) return false;

		// Calculate histogram
		float complement_learning_rate = 1.0f - learning_rate;
		float learning_rate_divide_sum = learning_rate / sum;
#ifndef _DEBUG
#pragma omp simd
#endif
		for (int i = 0; i < n_histogram_bins_cubed_; i++) {
			//每个直方图乘学习率
			(*histogram)[i] *= complement_learning_rate;
			//见论文
			(*histogram)[i] += temp_histogram[i] * learning_rate_divide_sum;
		}
		/*for (int i = 0; i < n_histogram_bins_cubed_; i++) {
			cout<< (*histogram)[i] <<endl;
		}*/
		return true;
	}

	void RegionModality::CalculateBasicLineData(const Model::PointData &data_point,
		DataLine *data_line) const {
		Eigen::Vector3f center_f_camera{ body2camera_pose_ * data_point.center_f_body };
		Eigen::Vector2f normal_f_camera{
			(body2camera_rotation_xy_ * data_point.normal_f_body).normalized() };

		data_line->center_f_body = data_point.center_f_body;
		data_line->center_f_camera = center_f_camera;
		data_line->center_u = center_f_camera(0) * fu_ / center_f_camera(2) + ppu_;
		data_line->center_v = center_f_camera(1) * fv_ / center_f_camera(2) + ppv_;
		data_line->normal_u = normal_f_camera(0);
		data_line->normal_v = normal_f_camera(1);
		data_line->continuous_distance =
			std::min(data_point.background_distance, data_point.foreground_distance) *
			fu_ / (center_f_camera(2) * fscale_);
	}

	bool RegionModality::IsLineValid(float u, float v,
		float continuous_distance) const {
		// Check if continuous distance is long enough
		if (continuous_distance < min_continuous_distance_) return false;

		// Check if image coordinate is on image
		int i_u = int(u + 0.5f);
		int i_v = int(v + 0.5f);
		if (i_u < 0 || i_u > image_width_minus_1_ || i_v < 0 ||
			i_v > image_height_minus_1_)
			return false;

		// Check if line center is on mask
		if (use_occlusion_handling_) {
			const cv::Mat &image{ camera_ptr_->image_mask_dilate() };
			float mask_prob = image.at<float>(i_v, i_u);
			bool flag_occ = false;
			if (mask_prob > 0.60)
				flag_occ = true;
			//return occlusion_renderer_ptr_->GetValue(i_v, i_u) & encoded_occlusion_id_;
			return (occlusion_renderer_ptr_->GetValue(i_v, i_u) & encoded_occlusion_id_) && flag_occ;

			//return occlusion_renderer_ptr_->GetValue(i_v, i_u) & encoded_occlusion_id_;
		}
		return true;
	}

	bool RegionModality::CalculateSegmentProbabilities(
		float center_u, float center_v, float normal_u, float normal_v,
		std::vector<float> *segment_probabilities_f,
		std::vector<float> *segment_probabilities_b,
		float *normal_component_to_scale, float *delta_r, int date_point_index) const {
		const cv::Mat &image{ camera_ptr_->image() };

		const cv::Mat &image_mask{ camera_ptr_->image_mask() };

		/*cv::cvtColor(image_mask, image_mask, CV_BGR2GRAY);
		std::cout << image_mask.channels() << std::endl;*/
		/*cv::imshow("image_mask", image_mask);
		cv::waitKey(0);*/
		

		/*cv::Mat img_show = camera_ptr_->image().clone();
		cout<< "index_datepoint: "<<date_point_index <<endl;
		if (date_point_index == 0)
		{
			img_show.at<cv::Vec3b>(center_v, center_u) = cv::Vec3b(255, 0, 0);
			cv::namedWindow("img_show", 0);
			cv::imshow("img_show", img_show);
			cv::waitKey(0);
		}*/

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
					if (global_flag)
					{
						MultiplyPixelColorProbability(image.at<cv::Vec3b>(int(v_f), u),
							segment_probability_f,
							segment_probability_b);

						
						float f_prob = image_mask.at<float>(int(v_f), u);
						float b_prob = 1 - image_mask.at<float>(int(v_f), u);
						//*segment_probability_f = *segment_probability_f * (1 - g_l_ratio) + g_l_ratio * f_prob;
						//*segment_probability_b = *segment_probability_b* (1 - g_l_ratio) + g_l_ratio * b_prob;
						*segment_probability_f = *segment_probability_f * (1 - g_l_ratio) + g_l_ratio * f_prob;
						*segment_probability_b = *segment_probability_b* (1 - g_l_ratio) + g_l_ratio * b_prob;
					}


					if (local_flag_ && flag_match_3d_point_)
					{
						float f = 1.0f;
						float b = 1.0f;
						float *f_ = &f;
						float *b_ = &b;
						MultiplyPixelColorProbability_local(image.at<cv::Vec3b>(int(v_f), u),
							f_,
							b_, date_point_index);
						*segment_probability_f = *segment_probability_f * g_l_ratio + (1 - g_l_ratio) * *f_;
						*segment_probability_b = *segment_probability_b * g_l_ratio + (1 - g_l_ratio) * *b_;
						//cout << *segment_probability_f << endl;
						//*segment_probability_f = *segment_probability_f * *f_;
						//*segment_probability_b = *segment_probability_b * *b_;
					}

					//加入梯度响应
					/*if (scale_ <= 2)
					{
						float mag = mag_.at<float>(int(v_f), u);
						float mag1 = mag * mag;
						float mag2 = mag * mag * mag;
						if (mag > 0)
						{
							*segment_probability_f = *segment_probability_f * mag1;
							*segment_probability_b = *segment_probability_b * mag2;
						}
					}*/

					//float angle_real = ori_.at<float>(int(v_f), u);
					//angle_real = angle_real * 180 / CV_PI;
					//if (angle_real > 180)
					//	angle_real = angle_real - 360.0;
					//float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
					//float distance_angle = (angle_real - angle_virtual);
					////cout << "distance_angle: "<<distance_angle << endl;
					//float score_match = fabs(cos(distance_angle * CV_PI / 180.0));

					//if (score_match > 0)
					//{
					//	*segment_probability_f = *segment_probability_f * score_match;
					//	*segment_probability_b = *segment_probability_b * (1.0 - score_match);
					//}

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
					if (global_flag)
					{
						MultiplyPixelColorProbability(image.at<cv::Vec3b>(int(v_f), u),
							segment_probability_f,
							segment_probability_b);

						float f_prob = image_mask.at<float>(int(v_f), u);
						float b_prob = 1 - image_mask.at<float>(int(v_f), u);
						*segment_probability_f = *segment_probability_f * (1 - g_l_ratio) + g_l_ratio * f_prob;
						*segment_probability_b = *segment_probability_b* (1 - g_l_ratio) + g_l_ratio * b_prob;

					}

					if (local_flag_ && flag_match_3d_point_)
					{
						float f = 1.0f;
						float b = 1.0f;
						float *f_ = &f;
						float *b_ = &b;
						MultiplyPixelColorProbability_local(image.at<cv::Vec3b>(int(v_f), u),
							f_,
							b_, date_point_index);
						*segment_probability_f = *segment_probability_f * g_l_ratio + (1 - g_l_ratio) * *f_;
						*segment_probability_b = *segment_probability_b * g_l_ratio + (1 - g_l_ratio) * *b_;

						//*segment_probability_f = *segment_probability_f * *f_;
						//*segment_probability_b = *segment_probability_b * *b_;
					}


					//加入梯度响应
					/*if (scale_ <= 2)
					{
						float mag = mag_.at<float>(int(v_f), u);
						float mag1 = mag * mag;
						float mag2 = mag * mag * mag;
						if (mag > 0)
						{
							*segment_probability_f = *segment_probability_f * mag1;
							*segment_probability_b = *segment_probability_b * mag2;
						}
					}*/

					//float angle_real = ori_.at<float>(int(v_f), u);
					//angle_real = angle_real * 180 / CV_PI;
					//if (angle_real > 180)
					//	angle_real = angle_real - 360.0;
					//float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
					//float distance_angle = (angle_real - angle_virtual);
					////cout << "distance_angle: "<<distance_angle << endl;
					//float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
					//score_match = score_match * score_match * score_match * score_match * score_match;
					//if (score_match > 0)
					//{
					//	*segment_probability_f = *segment_probability_f * score_match;
					//	*segment_probability_b = *segment_probability_b * (1.0 - score_match);
					//}


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
					if (global_flag)
					{
						MultiplyPixelColorProbability(image.at<cv::Vec3b>(v, int(u_f)),
							segment_probability_f,
							segment_probability_b);

						float f_prob = image_mask.at<float>(v, int(u_f));
						float b_prob = 1 - image_mask.at<float>(v, int(u_f));
						*segment_probability_f = *segment_probability_f * (1 - g_l_ratio) + g_l_ratio * f_prob;
						*segment_probability_b = *segment_probability_b* (1 - g_l_ratio) + g_l_ratio * b_prob;
					}
					if (local_flag_ && flag_match_3d_point_)
					{
						float f = 1.0f;
						float b = 1.0f;
						float *f_ = &f;
						float *b_ = &b;
						MultiplyPixelColorProbability_local(image.at<cv::Vec3b>(v, int(u_f)),
							f_,
							b_, date_point_index);
						*segment_probability_f = *segment_probability_f * g_l_ratio + (1 - g_l_ratio) * *f_;
						*segment_probability_b = *segment_probability_b * g_l_ratio + (1 - g_l_ratio) * *b_;

						//*segment_probability_f = *segment_probability_f * *f_;
						//*segment_probability_b = *segment_probability_b * *b_;
					}


					//加入梯度响应
					/*if (scale_ <= 2)
					{
						float mag = mag_.at<float>(v, int(u_f));
						float mag1 = mag * mag;
						float mag2 = mag * mag * mag;
						if (mag > 0)
						{
							*segment_probability_f = *segment_probability_f * mag1;
							*segment_probability_b = *segment_probability_b * mag2;
						}
					}*/

					//float angle_real = ori_.at<float>(v, int(u_f));
					//angle_real = angle_real * 180 / CV_PI;
					//if (angle_real > 180)
					//	angle_real = angle_real - 360.0;
					//float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
					//float distance_angle = (angle_real - angle_virtual);
					////cout << "distance_angle: "<<distance_angle << endl;
					//float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
					//score_match = score_match * score_match * score_match * score_match * score_match;
					//if (score_match > 0)
					//{
					//	*segment_probability_f = *segment_probability_f * score_match;
					//	*segment_probability_b = *segment_probability_b * (1.0 - score_match);
					//}

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
					if (global_flag)
					{
						MultiplyPixelColorProbability(image.at<cv::Vec3b>(v, int(u_f)),
							segment_probability_f,
							segment_probability_b);

						float f_prob = image_mask.at<float>(v, int(u_f));
						float b_prob = 1 - image_mask.at<float>(v, int(u_f));
						*segment_probability_f = *segment_probability_f * (1 - g_l_ratio) + g_l_ratio * f_prob;
						*segment_probability_b = *segment_probability_b* (1 - g_l_ratio) + g_l_ratio * b_prob;
					}

					if (local_flag_ && flag_match_3d_point_)
					{
						float f = 1.0f;
						float b = 1.0f;
						float *f_ = &f;
						float *b_ = &b;
						MultiplyPixelColorProbability_local(image.at<cv::Vec3b>(v, int(u_f)),
							f_,
							b_, date_point_index);
						*segment_probability_f = *segment_probability_f * g_l_ratio + (1 - g_l_ratio) * *f_;
						*segment_probability_b = *segment_probability_b * g_l_ratio + (1 - g_l_ratio) * *b_;

						//*segment_probability_f = *segment_probability_f * *f_;
						//*segment_probability_b = *segment_probability_b * *b_;
					}

					//加入梯度响应
					/*if(scale_ <= 2)
					{
						float mag = mag_.at<float>(v, int(u_f));
						float mag1 = mag * mag;
						float mag2 = mag * mag * mag;
						if (mag > 0)
						{
							*segment_probability_f = *segment_probability_f * mag1;
							*segment_probability_b = *segment_probability_b * mag2;
						}
					}*/

					//float angle_real = ori_.at<float>(v, int(u_f));
					//angle_real = angle_real * 180 / CV_PI;
					//if (angle_real > 180)
					//	angle_real = angle_real - 360.0;
					//float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
					//float distance_angle = (angle_real - angle_virtual);
					////cout << "distance_angle: "<<distance_angle << endl;
					//float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
					//score_match = score_match * score_match * score_match * score_match * score_match;
					//if (score_match > 0)
					//{
					//	*segment_probability_f = *segment_probability_f * score_match;
					//	*segment_probability_b = *segment_probability_b * (1.0 - score_match);
					//}

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

				//std::cout << "*segment_probability_f region: " << *segment_probability_f << std::endl;
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

	void RegionModality::MultiplyPixelColorProbability(const cv::Vec3b &pixel_color,
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

	void RegionModality::MultiplyPixelColorProbability_local(const cv::Vec3b &pixel_color,
		float *probability_f,
		float *probability_b, int date_point_index) const {
		// Retrive pixel color probability values
		int idx = (pixel_color[0] >> histogram_bitshift_) * n_histogram_bins_squared_;
		idx += (pixel_color[1] >> histogram_bitshift_) * n_histogram_bins_;
		idx += pixel_color[2] >> histogram_bitshift_;
		float pixel_color_probability_f = histogram_f_local_[date_point_index][idx];
		float pixel_color_probability_b = histogram_b_local_[date_point_index][idx];

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

	void RegionModality::CalculateDistribution(
		const std::vector<float> &segment_probabilities_f,
		const std::vector<float> &segment_probabilities_b,
		std::vector<float> *distribution) const {
		std::vector<float>::const_iterator segment_probabilities_f_it;
		std::vector<float>::const_iterator segment_probabilities_b_it;
		std::vector<float>::const_iterator function_lookup_f_it;
		std::vector<float>::const_iterator function_lookup_b_it;
		distribution->resize(distribution_length_);
		float distribution_area = 0.0f;

		//加入整体正态分布
		std::vector<float>::const_iterator function_lookup_f_distribution_it;
		function_lookup_f_distribution_it = begin(function_lookup_edge_distrbution_all);

		// Loop over entire distribution and start values of segment probabilities
		auto segment_probabilities_f_it_start = begin(segment_probabilities_f);
		auto segment_probabilities_b_it_start = begin(segment_probabilities_b);

		for (auto distribution_it = begin(*distribution);
			distribution_it != end(*distribution);
			++distribution_it, ++segment_probabilities_f_it_start,
			++segment_probabilities_b_it_start, ++function_lookup_f_distribution_it) {
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
				if (*distribution_it == 0)
				{
					*distribution_it = 0.0001;
				}
			}

			//整个分布的高斯
			//这句话在精修时有用，但在实际过程中，大位移时效果不好
			//*distribution_it *= *function_lookup_f_distribution_it;

			distribution_area += *distribution_it;
		}
		//cout<<"====================="<<endl;
		// Normalize distribution
		for (auto &probability_distribution : *distribution) {
			probability_distribution /= distribution_area;

			//std::cout << "distribution region: " << probability_distribution << std::endl;
		}
	}

	void RegionModality::CalculateDistributionMoments(
		const std::vector<float> &distribution, float *mean,
		float *standard_deviation, float *variance) const {
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
		//均值、方差、标准差
		*mean = mean_from_begin - distribution_length_minus_1_half_;
		*variance = std::max(distribution_variance, min_variance_);
		*standard_deviation = std::sqrt(*variance);

		//cout << "mean region "<<*mean << endl;
		/*cout << *variance << endl;
		cout << *standard_deviation << endl;*/
	}

	void RegionModality::ShowAndSaveImage(const std::string &title, int save_index,
		const cv::Mat &image) const {
		if (display_visualization_)
		{
			cv::namedWindow(title, 0);
			cv::imshow(title, image);
		}
		if (save_visualizations_) {
			std::experimental::filesystem::path path{
				save_directory_ / (title + "_" + std::to_string(save_index) + ".png") };
			cv::imwrite(path.string(), image);
		}
	}

	void RegionModality::VisualizePointsCameraImage(const std::string &title,
		int save_index) const {
		cv::Mat visualization_image;
		camera_ptr_->image().copyTo(visualization_image);
		DrawPoints(cv::Vec3b{ 24, 184, 234 }, &visualization_image);
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
	}

	void RegionModality::VisualizePointsHistogramImage(const std::string &title,
		int save_index) const {
		cv::Mat visualization_image(camera_ptr_->image().size(), CV_8UC3);
		DrawProbabilityImage(cv::Vec3b{ 255, 255, 255 }, &visualization_image);
		DrawPoints(cv::Vec3b{ 24, 184, 234 }, &visualization_image);
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
	}

	void RegionModality::VisualizePointsOcclusionMask(const std::string &title,
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

	void RegionModality::VisualizeLines(const std::string &title,
		int save_index) const {
		/*cv::Mat visualization_image_for_region = camera_ptr_->image().clone();
		DrawLines_fb(cv::Vec3b{ 24, 184, 234 }, cv::Vec3b{ 61, 63, 179 },&visualization_image_for_region);
		DrawPoints(cv::Vec3b{ 0, 255, 0 }, &visualization_image_for_region);
		cv::imshow("visualization_image_for_region", visualization_image_for_region);
		cv::resize(visualization_image_for_region, visualization_image_for_region,
			cv::Size(visualization_image_for_region.cols * 2, visualization_image_for_region.rows * 2));
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image_for_region);*/
		
		//cv::waitKey(0);

		cv::Mat visualization_image(camera_ptr_->image().size(), CV_8UC3);
		DrawProbabilityImage(cv::Vec3b{ 255, 255, 255 }, &visualization_image);
		//cv::imwrite("visualization_image.jpg", visualization_image);
		//normalize(visualization_image, visualization_image,0,255, cv::NORM_MINMAX);

		const cv::Mat &image_mask{ camera_ptr_->image_mask() };
		cv::Mat dst;
		image_mask.copyTo(dst);
		dst.convertTo(dst, CV_8U, 255.0, 0);
		bitwise_not(dst, dst);

		//cv::imshow("image_mask", dst);
		//cv::imshow("visualization_image", visualization_image);
		cv::Mat result;
		cv::Mat vis_image;
		cv::cvtColor(visualization_image, vis_image, CV_BGR2GRAY);

		cv::addWeighted(dst, 0.5, vis_image, 0.5, 0, result);
		cv::cvtColor(result, result, CV_GRAY2BGR);
		//cv::imshow("result_mask", result);
		//cv::waitKey(1);
		/*红色*/
		//cv::Vec3b{ 24, 184, 234 }, cv::Vec3b{ 61, 63, 179 }
		DrawLines(cv::Vec3b{ 24, 184, 234 }, cv::Vec3b{ 61, 63, 179 },
			&result);
		ShowAndSaveImage(name_ + "_" + title, save_index, result);
	}

	void RegionModality::DrawPoints(const cv::Vec3b &color_point,
		cv::Mat *image) const {
		for (const auto &data_line : data_lines_) {
			DrawPointInImage(data_line.center_f_camera, color_point,
				camera_ptr_->intrinsics(), image);
		}
	}

	void RegionModality::DrawLines(const cv::Vec3b &color_line,
		const cv::Vec3b &color_high_probability,
		cv::Mat *image) const {
		cv::Mat heatmap(image->size(), CV_8UC1, cv::Scalar(0));
		/*只保留这些像素*/
		cv::Mat map_for_deal(image->size(), CV_8UC1, cv::Scalar(0));

		float scale_minus_1_half_ = (fscale_ - 1.0f) / 2.0f;
		int u, v;
		for (const auto &data_line : data_lines_) {
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
					float color_ratio = std::min(3 * data_line.distribution[i], 1.0f);
					image->at<cv::Vec3b>(v, u) = color_ratio * color_high_probability +
						(1.0f - color_ratio) * color_line;
					
					heatmap.at<uchar>(v, u) = uchar(color_ratio * 255);
					map_for_deal.at< uchar>(v, u) = uchar(255);
				}
			}
		}
#if 0
		// 创建自定义颜色表
		cv::Vec3b Start = { 24, 184, 234 };
		cv::Vec3b end = { 61, 63, 179 };
		cv::Vec3b diff = {37, 121, 55};
		float b = 0.145;
		float g = 0.475;
		float r = 0.216;
		cout<< diff <<endl;
		cv::Mat customColorMap(256, 1, CV_8UC3);
		for (int i = 0; i < 256; i++)
		{
			// 在这里根据你的需求定义颜色映射

			// 你可以使用任何颜色空间，这里使用RGB作为示例
			// 以灰度值线性映射到RGB颜色空间
			customColorMap.at<cv::Vec3b>(i) = cv::Vec3b((24 + i * b), (184 - i * g), (234 - i * r));
		}
		cv::imshow("heatmap", heatmap);
		cv::waitKey(0);

		cv::Mat coloredHeatmap;
		cv::applyColorMap(heatmap, coloredHeatmap, customColorMap);  //COLORMAP_SUMMER  COLORMAP_AUTUMN
		for (int y = 0; y < coloredHeatmap.rows; y++)
		{
			for (int x = 0; x < coloredHeatmap.cols; x++)
			{
				if (map_for_deal.at<uchar>(y, x) == 0)
				{
					coloredHeatmap.at<cv::Vec3b>(y, x) = cv::Vec3b(0,0,0);  //220,220,220
				}
			}
		}

		//cv::Mat colorBar;
		//// 创建图例
		//cv::Mat legend(256, 10, CV_8UC1);
		//for (int y = 0; y < legend.rows; y++)
		//{
		//	for (int x = 0; x < legend.cols; x++)
		//	{
		//		legend.at<uchar>(y, x) = 256 - uchar(y);
		//	}
		//}
		//cv::applyColorMap(legend, colorBar, customColorMap);
		//resize(colorBar, colorBar,cv::Size(10,40));
		//colorBar.copyTo(coloredHeatmap(cv::Rect(0, 0, 10, 40)));
		// 添加刻度到图例
		/*for (int i = 0; i < 7; i++)
		{
			cv::putText(coloredHeatmap, std::to_string(i * 0.2),
			cv::Point(12, i * 42), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
		}*/

		//cv::namedWindow("colorBar", 0);
		//cv::imshow("colorBar", colorBar);
		cv::namedWindow("heatmap", 0);		
		cv::imshow("heatmap", coloredHeatmap);	
		cv::waitKey(0);
#endif
	}

	void RegionModality::DrawLines_fb(const cv::Vec3b &color_line,
		const cv::Vec3b &color_high_probability,
		cv::Mat *image) const {

		for (const auto &data_line : data_lines_) {
			// Iterate over foreground pixels
			Eigen::Vector2f center{
					data_line.center_u,
					data_line.center_v};
			//将其转换到图像
			Eigen::Vector2f normal{
				data_line.normal_u ,data_line.normal_v };

			float u = center(0) - normal(0) * unconsidered_line_length_ + 0.5f;
			float v = center(1) - normal(1) * unconsidered_line_length_ + 0.5f;

			int n_iteration =int(12) +	0.5f;

			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				image->at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(255, 127, 80);

				u -= normal(0);
				v -= normal(1);
			}

			// Iterate over background pixels
			u = center(0) + normal(0) * unconsidered_line_length_ + 0.5f;
			v = center(1) + normal(1) * unconsidered_line_length_ + 0.5f;
			
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				image->at<cv::Vec3b>(int(v), int(u)) = cv::Vec3b(30, 144, 255);
				u += normal(0);
				v += normal(1);
			}
		}
	}

	void RegionModality::DrawProbabilityImage(const cv::Vec3b &color_b,
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
				//lnb
				/*for (int i = 0;i < num_local_hist;i++)
				{
					MultiplyPixelColorProbability_local(color_image_value[u], &pixel_probability_f,
						&pixel_probability_b, i);
				}*/
				/**/
				probability_image_value[u] = color_b * pixel_probability_b;
			}
		}
	}

	void RegionModality::UpdateLineCentersWithCurrentPose() {
		Transform3fA body2camera_pose{ camera_ptr_->world2camera_pose() *
									  body_ptr_->body2world_pose() };
		for (auto &data_line : data_lines_) {
			data_line.center_f_camera = body2camera_pose * data_line.center_f_body;
		}
	}

	float RegionModality::MinAbsValueWithSignOfValue1(float value_1,
		float abs_value_2) {
		if (std::abs(value_1) < abs_value_2)
			return value_1;
		else
			return sgnf(value_1) * abs_value_2;
	}

	bool RegionModality::IsSetup() const {
		if (!set_up_) {
			std::cerr << "Set up region modality " << name_ << " first" << std::endl;
			return false;
		}
		return true;
	}

	void RegionModality::PrecalculateExtractEdge()
	{
		const cv::Mat &image{ camera_ptr_->image() };
		cv::Mat canny_mat;
		cv::cvtColor(image, canny_mat, CV_BGR2GRAY);
		//进行sobel计算
		cv::Mat sobel_dx;
		cv::Mat sobel_dy;
		cv::Sobel(canny_mat, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
		cv::Sobel(canny_mat, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_DEFAULT);

		cv::Mat mag = cv::Mat::zeros(canny_mat.size(), CV_32FC1);
		cv::Mat ori = cv::Mat::zeros(canny_mat.size(), CV_32FC1);

		//==为true时表示以弧度表示====
		cv::cartToPolar(sobel_dx, sobel_dy, mag, ori);//true
		cv::Mat ori_temp;
		//弧度制
		phase(sobel_dx, sobel_dy, ori_temp);
		mag_ = mag;
		ori_ = ori_temp;

		//normalize(ori_, ori_, 0, 1, cv::NORM_MINMAX);
		normalize(mag_, mag_, 0, 1, cv::NORM_MINMAX);

		//cv::namedWindow("ori_", 0);
		//cv::imshow("ori_", ori_);

		/*cv::namedWindow("mag_",0);
		cv::imshow("mag_", mag_);
		cv::waitKey(0);*/

	}
	float RegionModality::TukeyNorm(float error) {
		if (std::abs(error) <= tukey_norm_constant_)
			return powf(tukey_norm_constant_, 2.0f) / 6.0f *
			(1.0f - powf(1.0f - powf(error / tukey_norm_constant_, 2.0f), 3.0f));
		else
			return powf(tukey_norm_constant_, 2.0f) / 6.0f;
	}

}  // namespace srt3d

