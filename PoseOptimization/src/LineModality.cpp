// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#include <srt3d/LineModality.h>
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

#include <assert.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <stdlib.h>

#include <time.h>
#include <math.h>
#include <srt3d/lap.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
namespace srt3d {

	LineModality::LineModality(const std::string &name,
		std::shared_ptr<Body> body_ptr,
		std::shared_ptr<LineModel> LineModel_ptr,
		std::shared_ptr<Camera> camera_ptr)
		: name_{ name },
		body_ptr_{ std::move(body_ptr) },
		LineModel_ptr_{ std::move(LineModel_ptr) },
		camera_ptr_{ std::move(camera_ptr) } {
		tikhonov_matrix_.setZero();
		tikhonov_matrix_.diagonal().head<3>().array() = tikhonov_parameter_rotation_;
		tikhonov_matrix_.diagonal().tail<3>().array() =
			tikhonov_parameter_translation_;
	}

	bool LineModality::SetUp() {
		set_up_ = false;

		// Check if all required objects are set up
		if (!LineModel_ptr_->set_up()) {
			std::cout << "EdgeModel " << LineModel_ptr_->name() << " was not set up"
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
		//PrecalculateHistogramBinVariables();
		PrecalculateBodyVariables();
		PrecalculateCameraVariables();
		SetImshowVariables();

		set_up_ = true;
		return true;
	}

	void LineModality::set_n_lines(int n_lines) { n_lines_ = n_lines; }

	void LineModality::set_function_amplitude(float function_amplitude) {
		function_amplitude_ = function_amplitude;
		set_up_ = false;
	}

	void LineModality::set_function_slope(float function_slope) {
		function_slope_ = function_slope;
		set_up_ = false;
	}

	void LineModality::set_learning_rate(float learning_rate) {
		learning_rate_ = learning_rate;
	}

	void LineModality::set_function_length(int function_length) {
		function_length_ = function_length;
		set_up_ = false;
	}

	void LineModality::set_distribution_length(int distribution_length) {
		distribution_length_ = distribution_length;
		set_up_ = false;
	}

	void LineModality::set_scales(const std::vector<int> &scales) {
		scales_ = scales;
	}

	void LineModality::set_n_newton_iterations(int n_newton_iterations) {
		n_newton_iterations_ = n_newton_iterations;
	}

	void LineModality::set_min_continuous_distance(
		float min_continuous_distance) {
		min_continuous_distance_ = min_continuous_distance;
	}

	bool LineModality::set_n_histogram_bins(int n_histogram_bins) {
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

	void LineModality::set_learning_rate_f(float learning_rate_f) {
		learning_rate_f_ = learning_rate_f;
	}

	void LineModality::set_learning_rate_b(float learning_rate_b) {
		learning_rate_b_ = learning_rate_b;
	}

	void LineModality::set_unconsidered_line_length(
		float unconsidered_line_length) {
		unconsidered_line_length_ = unconsidered_line_length;
	}

	void LineModality::set_considered_line_length(float considered_line_length) {
		considered_line_length_ = considered_line_length;
	}

	void LineModality::set_tikhonov_parameter_rotation(
		float tikhonov_parameter_rotation) {
		tikhonov_parameter_rotation_ = tikhonov_parameter_rotation;
		tikhonov_matrix_.diagonal().head<3>().array() = tikhonov_parameter_rotation_;
	}

	void LineModality::set_tikhonov_parameter_translation(
		float tikhonov_parameter_translation) {
		tikhonov_parameter_translation_ = tikhonov_parameter_translation;
		tikhonov_matrix_.diagonal().tail<3>().array() =
			tikhonov_parameter_translation_;
	}

	void LineModality::UseOcclusionHandling(
		std::shared_ptr<OcclusionRenderer> occlusion_renderer_ptr) {
		occlusion_renderer_ptr_ = std::move(occlusion_renderer_ptr);
		use_occlusion_handling_ = true;
		set_up_ = false;
	}

	void LineModality::DoNotUseOcclusionHandling() {
		occlusion_renderer_ptr_ = nullptr;
		use_occlusion_handling_ = false;
		set_up_ = false;
	}

	void LineModality::set_display_visualization(bool display_visualization) {
		display_visualization_ = display_visualization;
	}

	void LineModality::StartSavingVisualizations(
		const std::experimental::filesystem::path &save_directory) {
		save_visualizations_ = true;
		save_directory_ = save_directory;
	}

	void LineModality::StopSavingVisualizations() {
		save_visualizations_ = false;
	}

	void LineModality::set_visualize_lines_correspondence(
		bool visualize_lines_correspondence) {
		visualize_lines_correspondence_ = visualize_lines_correspondence;
		SetImshowVariables();
	}

	void LineModality::set_visualize_points_occlusion_mask_correspondence(
		bool visualize_points_occlusion_mask_correspondence) {
		visualize_points_occlusion_mask_correspondence_ =
			visualize_points_occlusion_mask_correspondence;
		SetImshowVariables();
	}

	void LineModality::set_visualize_points_pose_update(
		bool visualize_points_pose_update) {
		visualize_points_pose_update_ = visualize_points_pose_update;
		SetImshowVariables();
	}

	void LineModality::set_visualize_points_histogram_image_pose_update(
		bool visualize_points_histogram_image_pose_update) {
		visualize_points_histogram_image_pose_update_ =
			visualize_points_histogram_image_pose_update;
		SetImshowVariables();
	}

	void LineModality::set_visualize_points_result(bool visualize_points_result) {
		visualize_points_result_ = visualize_points_result;
		SetImshowVariables();
	}

	void LineModality::set_visualize_points_histogram_image_result(
		bool visualize_points_histogram_image_result) {
		visualize_points_histogram_image_result_ =
			visualize_points_histogram_image_result;
		SetImshowVariables();
	}

	bool LineModality::StartModality() {
		if (!IsSetup()) return false;

		// Initialize histograms
		//这一步计算模型与相机的转换关系，是需要的
		PrecalculatePoseVariables();

		//mog2->setHistory(10);
		//mog2->setVarThreshold(16);
		//mog2->setDetectShadows(false);
		
		//构建颜色直方图
		//AddLinePixelColorsToTempHistograms();
		//直方图归一化
		/*if (CalculateHistogram(1.0f, temp_histogram_f_, &histogram_f_) &&
			CalculateHistogram(1.0f, temp_histogram_b_, &histogram_b_)) {
			return true;
		}*/
		/*else {
			std::cout << "Histograms could not be initialised for modality " << name_
				<< std::endl;
			return false;
		}*/
	}

	bool LineModality::CalculateBeforeCameraUpdate() {
		if (!IsSetup()) return false;

		PrecalculatePoseVariables();
		//AddLinePixelColorsToTempHistograms();
		//CalculateHistogram(learning_rate_f_, temp_histogram_f_, &histogram_f_);
		//CalculateHistogram(learning_rate_b_, temp_histogram_b_, &histogram_b_);

		return true;
	}

	bool LineModality::CalculateCorrespondences(int corr_iteration) {
		if (!IsSetup()) return false;
		float length_ = LastValidValue(distribution_length_vector_, corr_iteration);
		//cout<< length_ <<endl;
		distribution_length_ = length_;

		float turkey_noram_real_ = LastValidValue(tukey_norm_constant_vector_, corr_iteration);
		tukey_norm_constant_ = turkey_noram_real_;

		PrecalculateDistributionVariables();
		PrecalculatePoseVariables();
		//PrecalculateExtractEdge();
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
		const LineModel::TemplateView *template_view;
		//cout << body2camera_pose_.matrix() << endl;
		LineModel_ptr_->GetClosestTemplateView(body2camera_pose_, &template_view);
				
		// Iterate over n_lines
		std::vector<float> segment_probabilities_f(line_length_in_segments_);
		std::vector<float> segment_probabilities_b(line_length_in_segments_);
		data_lines_.clear();
#if 1
		std::vector<cv::Point2f> points_edge;
		std::vector<DataLine> dataLines;
		for (auto data_point = begin(template_view->data_points);
			data_point != begin(template_view->data_points) + n_lines_;
			++data_point) {
			DataLine data_line;
			//建立搜索线
			CalculateBasicLineData(*data_point, &data_line);
			//计算包围区域
			points_edge.push_back(cv::Point2f(data_line.center_u, data_line.center_v));
			if (!IsLineValid(data_line.center_u, data_line.center_v,
				data_line.continuous_distance))
				continue;
			dataLines.push_back(data_line);

		}
		rect_roi_ = boundingRect(points_edge);

		PrecalculateExtractEdge();
#endif
		//cv::Mat show_img_test_ = camera_ptr_->image().clone();

		

		/*for (auto data_point = begin(template_view->data_points);
			data_point != begin(template_view->data_points) + n_lines_;
			++data_point)*/
		for (int i = 0; i < dataLines.size(); i++)
		{
			DataLine data_line = dataLines[i];
			//DataLine data_line;
			//建立搜索线
			/*CalculateBasicLineData(*data_point, &data_line);

			if (!IsLineValid(data_line.center_u, data_line.center_v,
				data_line.continuous_distance))
				continue;*/
			//计算得到normal_component_to_scale 和 delta_r
#if 1
			if (!CalculateSegmentProbabilities_edge(
				data_line.center_u, data_line.center_v, data_line.normal_u,
				data_line.normal_v, &segment_probabilities_f,
				&segment_probabilities_b, &data_line.normal_component_to_scale,
				&data_line.delta_r, &data_line.distribution, &data_line.mean, &data_line.standard_deviation, &data_line.variance, 
				data_line.vimg_desc))
				continue;
			data_lines_.push_back(std::move(data_line));
#endif			
			/*CalculateEdgeSearchLineDistribution(data_line.center_u, data_line.center_v, data_line.normal_u, data_line.normal_v,
				&data_line.distribution);

			CalculateDistributionMoments(data_line.distribution, &data_line.mean,
				&data_line.standard_deviation,
				&data_line.variance);*/
			//show_img_test_.at<cv::Vec3b>(int(data_line.center_v), int(data_line.center_u)) = cv::Vec3b(255,0,0);
		}


		//cv::imshow("show_gmm", show_img_test_);
		//cv::waitKey(0);
		//cout<< data_lines_ .size()<<endl;
		//计算高斯混合模型
		//preGMM();

		return true;
	}
	bool LineModality::preGMM(float center_u, float center_v, float normal_u, float normal_v,int index_line, std::vector<cv::Vec3b> *value) const
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
	bool LineModality::CalculateEdgeSearchLineDistribution(float center_u, float center_v, float normal_u, float normal_v, 
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

	bool LineModality::VisualizeCorrespondences(int save_idx) {
		if (!IsSetup()) return false;

		if (visualize_lines_correspondence_)
			VisualizeLines("lines_correspondence", save_idx);
		if (visualize_points_occlusion_mask_correspondence_ &&
			use_occlusion_handling_)
			VisualizePointsOcclusionMask("occlusion_mask_correspondence", save_idx);
		return true;
	}

	//计算位姿
	bool LineModality::CalculatePoseUpdate(int corr_iteration,
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
#if 1
		int angle_left_num = 0;
		int angle_right_num = 0;
		int angle_up_num = 0;
		int angle_down_num = 0;
		//cv::Mat img = camera_ptr_->image().clone();
		//cv::Mat img = camera_ptr_->image().clone();
		cv::Mat img;
		int debug = 1;
		if(debug)
			img = camera_ptr_->image().clone();
		vector <cv::Point> contourPts1;
		vector <cv::Point> contourPts2;
		float error_count = 0.0f;
		float nerr = 0.0f;
		float match_num = 0;
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
			/*if (max_prob <= 0.45)
			{
				continue;
			}*/

			match_num++;
			contourPts1.push_back(cv::Point(x_, y_));
			contourPts2.push_back(cv::Point(data_line.center_u, data_line.center_v));

			if (debug)
			{
				cv::circle(img, cv::Point(data_line.center_u, data_line.center_v), 1, cv::Scalar(0, 0, 255), -1);
			    cv::circle(img, cv::Point(x_, y_), 1, cv::Scalar(0, 255, 255), -1);
			}
			

			//cv::imshow("img", img);
			//cv::waitKey(0);
			Eigen::Vector2f diff{ (data_line.center_u - x_), (data_line.center_v - y_)};
			float squared_error = diff.squaredNorm();
			float error = sqrtf(squared_error);

			// Calculate weight with Tukey norm
			float weight = 1.0f / variance_;
			if (error > std::numeric_limits<float>::min())
				weight = (TukeyNorm(error) / squared_error) / variance_;

			error_count += fabs(error) * weight;
			nerr += weight;
			// Calculate derivatives
			Eigen::MatrixXf  dx_dX(2,3);
			dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z, -y * fv_ / z2;
			Eigen::MatrixXf dx_dtranslation{ dx_dX * body2camera_rotation_ };
			Eigen::MatrixXf dx_dtheta(2,6);
			dx_dtheta << -dx_dtranslation * Vector2Skewsymmetric(data_line.center_f_body),
				dx_dtranslation;

			// Calculate gradient and hessian
			gradient_edge -= (weight * diff.transpose()) * dx_dtheta;
			hessian_edge.triangularView<Eigen::Lower>() -=
				(weight * dx_dtheta.transpose()) * dx_dtheta;
		}
		//特征误差计算
		hessian_edge = hessian_edge.selfadjointView<Eigen::Lower>();
		
		error_count_ = error_count /(nerr * match_num);
		//cout << "================" << endl;
		//cout << "line error_count_: " << error_count_ << endl;
		//CalShape(contourPts1, contourPts2);

		//cout << "================" << endl;
		//cv::imshow("img", img);
		//cv::waitKey(0);

		if (debug)
		{
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
	void LineModality::CalShape(std::vector<cv::Point> contourPts1, std::vector<cv::Point> contourPts2)
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
		//cout << "line match_ratio： " << match_ratio << endl;
		/*cv::imshow("line_mapping", resimg);
	    cv::imshow("line_map_clear", resimg2);
		cv::waitKey(0);*/
	}

	cv::Matx44f  LineModality::exp(cv::Matx61f xi)
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
	cv::Matx33f LineModality::axiator(cv::Vec3f a)
	{
		float a1 = a[0];
		float a2 = a[1];
		float a3 = a[2];

		return cv::Matx33f(0, -a3, a2,
			a3, 0, -a1,
			-a2, a1, 0);
	}
	bool LineModality::VisualizePoseUpdate(int save_idx) {
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

	bool LineModality::VisualizeResults(int save_idx) {
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

	const std::string &LineModality::name() const { return name_; }

	std::shared_ptr<Body> LineModality::body_ptr() const { return body_ptr_; }

	std::shared_ptr<LineModel> LineModality::LineModel_ptr() const { return LineModel_ptr_; }

	std::shared_ptr<Camera> LineModality::camera_ptr() const {
		return camera_ptr_;
	}

	std::shared_ptr<OcclusionRenderer> LineModality::occlusion_renderer_ptr()
		const {
		return occlusion_renderer_ptr_;
	}

	bool LineModality::imshow_correspondence() const {
		return imshow_correspondence_;
	}

	bool LineModality::imshow_pose_update() const { return imshow_pose_update_; }

	bool LineModality::imshow_result() const { return imshow_result_; }

	bool LineModality::set_up() const { return set_up_; }

	void LineModality::PrecalculateFunctionLookup() {
		
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
		}
		
	}

	void LineModality::PrecalculateDistributionVariables() {
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

	void LineModality::PrecalculateHistogramBinVariables() {
		n_histogram_bins_squared_ = pow_int(n_histogram_bins_, 2);
		std::cout << "n_histogram_bins_squared_:" << n_histogram_bins_squared_ << std::endl;
		n_histogram_bins_cubed_ = pow_int(n_histogram_bins_, 3);
		std::cout << "n_histogram_bins_cubed_:" << n_histogram_bins_cubed_ << std::endl;
		temp_histogram_f_.resize(n_histogram_bins_cubed_);
		temp_histogram_b_.resize(n_histogram_bins_cubed_);
		histogram_f_.resize(n_histogram_bins_cubed_);
		histogram_b_.resize(n_histogram_bins_cubed_);
	}

	void LineModality::SetImshowVariables() {
		imshow_correspondence_ = visualize_lines_correspondence_ ||
			(visualize_points_occlusion_mask_correspondence_ &&
				use_occlusion_handling_);
		imshow_pose_update_ = visualize_points_pose_update_ ||
			visualize_points_histogram_image_pose_update_;
		imshow_result_ =
			visualize_points_result_ || visualize_points_histogram_image_result_;
	}

	void LineModality::PrecalculateBodyVariables() {
		if (use_occlusion_handling_)
			encoded_occlusion_id_ = (uchar(1) << unsigned(body_ptr_->occlusion_id()));
	}

	void LineModality::PrecalculateCameraVariables() {
		fu_ = camera_ptr_->intrinsics().fu;
		fv_ = camera_ptr_->intrinsics().fv;
		ppu_ = camera_ptr_->intrinsics().ppu;
		ppv_ = camera_ptr_->intrinsics().ppv;
		image_width_minus_1_ = camera_ptr_->image().cols - 1;
		image_height_minus_1_ = camera_ptr_->image().rows - 1;
		image_width_minus_2_ = camera_ptr_->image().cols - 2;
		image_height_minus_2_ = camera_ptr_->image().rows - 2;
		std::cout << "fu_,fv_:"<< fu_ <<" "<< fv_ << std::endl;
	}

	void LineModality::PrecalculateExtractEdge()
	{
		const cv::Mat &image{ camera_ptr_->image()};
		//计算截取区域
		int expend_length = 40;
		cv::Rect roi_new = cv::Rect(rect_roi_.x - expend_length, rect_roi_.y - expend_length,
			rect_roi_.width + 2 * expend_length, rect_roi_.height + 2 * expend_length);
		if (roi_new.x + roi_new.width > image.cols)
		{
			roi_new.width = image.cols - roi_new.x;
		}
		if (roi_new.y + roi_new.height > image.rows)
		{
			roi_new.height = image.rows - roi_new.y;
		}
		if (roi_new.x < 0)
		{
			roi_new.x = 0;
		}
		if (roi_new.y < 0)
		{
			roi_new.y = 0;
		}

		//cv::rectangle(canny_mat, roi_new, cv::Scalar(255), 1);
		cv::Mat ori_img = image(roi_new);
		rect_roi_ = roi_new;

		cv::Mat canny_mat;
		cv::cvtColor(ori_img, canny_mat, CV_BGR2GRAY);
		
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

		cv::Mat mag = cv::Mat::zeros(canny_mat.size(), CV_32FC1);
		cv::Mat ori = cv::Mat::zeros(canny_mat.size(), CV_32FC1);
		
#if 1
		/*cv::Mat value;
		cv::magnitude(sobel_dx, sobel_dy, value);
		cv::imshow("value",value);
		cv::imwrite("value.jpg", value);*/
		//==为true时表示以弧度表示====
		//cv::cartToPolar(sobel_dx, sobel_dy, mag, ori);//true

		
		cv::Mat ori_temp;
		//弧度制
		phase(sobel_dx, sobel_dy, ori_temp);
		//mag_ = mag;
		ori_ = ori_temp;
		
		/*cv::Laplacian(canny_mat, canny_mat, CV_8U, 3, 1, 0);
		cv::convertScaleAbs(canny_mat, canny_mat);
		mag_ = canny_mat;*/
		//normalize(ori_, ori_, 0, 1, cv::NORM_MINMAX);
		//normalize(mag_, mag_, 0, 1, cv::NORM_MINMAX); //NORM_L2  NORM_MINMAX
		/*cv::imshow("mag", mag_);
		cv::imwrite("mag_.jpg", mag_);
		cv::waitKey(0);*/
# if 0
		cv::Mat grad_x, grad_y;
		/// Gradient X
		cv::Sobel(canny_mat, grad_x, CV_32F, 1, 0, 3);
		cv::Mat dst = cv::Mat::zeros(canny_mat.size(), CV_32FC1);
		/// Gradient Y
		cv::Sobel(canny_mat, grad_y, CV_32F, 0, 1, 3);

		for (int i = 0; i < canny_mat.rows; i++)
		{
			for (int j = 0; j < canny_mat.cols; j++)
			{
				dst.at<float>(i, j) = atan2(grad_y.at<float>(i, j), grad_x.at<float>(i, j));
				//std::cout << "(" << i << ", " << j << ") " << dst.at<float>(i,j) << " " << grad_y.at<float>(i,j) << " " << grad_x.at<float>(i,j) << std::endl;
			}
		}
		ori_ = dst;
#endif
		/*cv::namedWindow("ori_", 0);
		cv::imshow("ori_", ori_);
		cv::waitKey(0);*/

		/*cv::namedWindow("mag_",0);
		cv::imshow("mag_", mag_);
		cv::waitKey(0);*/
#endif

		/*cv::Mat save_mag;
		normalize(mag, save_mag, 0, 255, cv::NORM_MINMAX);
		save_mag.convertTo(save_mag, CV_8UC1);
		cv::imwrite("save_mag.jpg", save_mag);*/

		/*normalize(ori, ori, 0, 255, cv::NORM_MINMAX);
		ori.convertTo(ori,CV_8UC1);
		imshow("ori",ori);
		normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
		mag.convertTo(mag,CV_8UC1);
        imshow("mag",mag);
		cv::waitKey(0);*/

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
#if 1
		//cv::Mat input_img = cv::imread("normal_image_.png");
		cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
		ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
		ed->params.GradientThresholdValue = 30;    //80
		ed->params.AnchorThresholdValue = 8;//3
		vector<cv::Vec6d> ellipses;
		vector<cv::Vec4f> lines;
		lines.clear();
		//you should call this before detectLines() and detectEllipses()
		//cv::cvtColor(canny_mat, input_img, CV_BGR2GRAY);
		ed->detectEdges(canny_mat);
		cv::Mat edge_img,gray_img;
		//=======该函数很耗时？=======
		//ed->getEdgeImage(edge_img);
		//ed->getGradientImage(gray_img);
		/*cv::imshow("edge_img", edge_img);
		cv::imshow("gray_img", gray_img);
		cv::waitKey(0);*/
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

			//kl.startPointX = extremes[0];
			//kl.startPointY = extremes[1];
			//kl.endPointX = extremes[2];
			//kl.endPointY = extremes[3];
			//kl.sPointInOctaveX = extremes[0];
			//kl.sPointInOctaveY = extremes[1];
			//kl.ePointInOctaveX = extremes[2];
			//kl.ePointInOctaveY = extremes[3];
			//kl.lineLength = (float)sqrt(pow(extremes[0] - extremes[2], 2) + pow(extremes[1] - extremes[3], 2));

			///* compute number of pixels covered by line */
			//cv::LineIterator li(edge_image_ed, cv::Point2f(extremes[0], extremes[1]), cv::Point2f(extremes[2], extremes[3]));
			//kl.numOfPixels = li.count;

			//kl.angle = atan2((kl.endPointY - kl.startPointY), (kl.endPointX - kl.startPointX));
			//kl.class_id = ++class_counter;
			//kl.octave = 0;
			//kl.size = (kl.endPointX - kl.startPointX) * (kl.endPointY - kl.startPointY);
			//kl.response = kl.lineLength / max(edge_image_ed.cols, edge_image_ed.rows);
			//kl.pt = cv::Point2f((kl.endPointX + kl.startPointX) / 2, (kl.endPointY + kl.startPointY) / 2);
			//keylines.push_back(kl);

			cv::line(edge_image_ed, cv::Point(extremes[0], extremes[1]), cv::Point(extremes[2], extremes[3]), cv::Scalar(255), 1);
		}
		int lsdNFeatures = 100;
		//cout << "filter lines" << endl;
		/*if (keylines.size()>lsdNFeatures)
		{
			sort(keylines.begin(), keylines.end(), sort_lines_by_response());
			keylines.resize(lsdNFeatures);
			for (int i = 0; i<lsdNFeatures; i++)
				keylines[i].class_id = i;
		}*/
		//cv::Mat combine_img = cv::imread("edge_image_ed111.jpg",1);
		//for (int i = 0;i < keylines.size();i++)
		//{
		//	cv::line_descriptor::KeyLine line = keylines[i];
		//	cv::line(edge_image_ed, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(255), 1);
		//	//cv::line(combine_img, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(0,0,255), 1);			
		//}
		mag_ = edge_image_ed;
		//mag_ = edge_img;
		//cv::imshow("edge_image_ed", mag_);
		//cv::waitKey(0);
#endif
#if 0
		cv::Mat gray, edges;
		
		Canny(canny_mat, edges, 20, 40); // 进行边缘检测

		vector<cv::Vec4f> plines;

		//cvtColor(gray, edges, CV_GRAY2BGR);
		cv::HoughLinesP(edges, plines, 1, CV_PI / 180, 50, 20, 10); // 进行霍夫直线检测
		cv::Mat edge_hough = cv::Mat::zeros(canny_mat.size(), CV_8UC1);
		for (size_t i = 0; i < plines.size(); i++)
		{
			cv::Vec4f hline = plines[i];
			cv::line(edge_hough, cv::Point(hline[0], hline[1]), cv::Point(hline[2], hline[3]), cv::Scalar(255), 1, cv::LINE_AA);//绘制直线
		}
		mag_ = edge_hough;
		cv::imshow("Hough Lines", edge_hough);
		cv::waitKey(0);
#endif
		/*cv::namedWindow("edge_", 0);
		cv::imshow("edge_", canny_mat);
		cv::namedWindow("line_angle", 0);
		cv::imshow("line_angle", line_angle);*/

		//cv::Mat img = cv::imread("2022-08-30_14_52_32_630.bmp");
		//cv::Mat gray;
		//cv::cvtColor(img, gray, CV_BGR2GRAY);
		//lines.clear();
		////you should call this before detectLines() and detectEllipses()
		//ed->detectEdges(gray);
		//// Detect lines
		//ed->detectLines(lines);
		//cv::Mat edge_ = img.clone();
		//for (int i = 0;i < lines.size();i++)
		//{
		//	cv::Vec4f line = lines[i];
		//	cv::line(edge_, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0,255,0), 1);
		//}
		//cv::imshow("edge_", edge_);
		//cv::waitKey(0);

		//cv::imwrite("Canny_Mat_.jpg", Canny_Mat_);
	}

	void LineModality::PrecalculatePoseVariables() {
		body2camera_pose_ =
			camera_ptr_->world2camera_pose() * body_ptr_->body2world_pose();
		//cout<< "PrecalculatePoseVariables"<<body2camera_pose_ .matrix()<<endl;
		body2camera_rotation_ = body2camera_pose_.rotation().matrix();
		body2camera_rotation_xy_ = body2camera_rotation_.topRows<2>();
	}

	void LineModality::PrecalculateScaleDependentVariables(int corr_iteration) {
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

	void LineModality::AddLinePixelColorsToTempHistograms() {
		const cv::Mat &image{ camera_ptr_->image() };
		const LineModel::TemplateView *template_view;
		LineModel_ptr_->GetClosestTemplateView(body2camera_pose_, &template_view);

		// Iterate over all points
		std::fill(begin(temp_histogram_f_), end(temp_histogram_f_), 0.0f);
		std::fill(begin(temp_histogram_b_), end(temp_histogram_b_), 0.0f);
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
			float u = center(0) - normal(0) * unconsidered_line_length_ + 0.5f;
			float v = center(1) - normal(1) * unconsidered_line_length_ + 0.5f;
			int n_iteration =
				int(std::fmin(foreground_distance - 2.0f * unconsidered_line_length_,
					considered_line_length_) +
					0.5f);
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				AddPixelColorToHistogram(image.at<cv::Vec3b>(int(v), int(u)),
					&temp_histogram_f_);
				u -= normal(0);
				v -= normal(1);
			}

			// Iterate over background pixels
			u = center(0) + normal(0) * unconsidered_line_length_ + 0.5f;
			v = center(1) + normal(1) * unconsidered_line_length_ + 0.5f;
			n_iteration =
				int(std::fmin(background_distance - 2.0f * unconsidered_line_length_,
					considered_line_length_) +
					0.5f);
			for (int i = 0; i < n_iteration; ++i) {
				if (int(u) < 0 || int(u) > image_width_minus_1_ || int(v) < 0 ||
					int(v) > image_height_minus_1_)
					break;
				AddPixelColorToHistogram(image.at<cv::Vec3b>(int(v), int(u)),
					&temp_histogram_b_);
				u += normal(0);
				v += normal(1);
			}
		}
	}

	void LineModality::AddPixelColorToHistogram(
		const cv::Vec3b &pixel_color,
		std::vector<float> *enlarged_histogram) const {
		(*enlarged_histogram)[(pixel_color[0] >> histogram_bitshift_) *
			n_histogram_bins_squared_ +
			(pixel_color[1] >> histogram_bitshift_) *
			n_histogram_bins_ +
			(pixel_color[2] >> histogram_bitshift_)] += 1.0f;
	}

	bool LineModality::CalculateHistogram(
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

	void LineModality::CalculateBasicLineData(const LineModel::PointData &data_point,
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
			data_line->continuous_distance =
				std::min(data_point.background_distance, data_point.foreground_distance) *
				fu_ / (center_f_camera(2) * fscale_);
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

	bool LineModality::IsLineValid(float u, float v,
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
			return occlusion_renderer_ptr_->GetValue(i_v, i_u) & encoded_occlusion_id_;
		}
		return true;
	}
	bool LineModality::CalculateSegmentProbabilities_edge(
		float center_u, float center_v, float normal_u, float normal_v,
		std::vector<float> *segment_probabilities_f,
		std::vector<float> *segment_probabilities_b,
		float *normal_component_to_scale, float *delta_r, std::vector<float> *distribution, float *mean,
		float *standard_deviation, float *variance, Eigen::VectorXf vimg_desc){

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

				int col_index = 0;
				int segment_idx = 0;
				for (; u <= u_end; ++u, v_f += v_step, segment_idx++) {
					if (segment_idx == scale_) {
						*(++segment_probability_f) = 1.0f;
						*(++segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					//=================
					int mag = 0;
					mag = int(mag_.at<uchar>(int(v_f) - rect_roi_.y, u - rect_roi_.x));
					//cout << mag << endl;
					if (mag > 50.0)
					{
						float angle_real = ori_.at<float>(int(v_f) - rect_roi_.y, u - rect_roi_.x);
						angle_real = angle_real * 180 / CV_PI;
						//angle_real = 270.0 - angle_real;
						if (angle_real > 180)
							angle_real = angle_real - 360.0;
						float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
						float distance_angle = (angle_real - angle_virtual);
						float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
						if (score_match > 0.3)
						{
							Eigen::Vector2f diff{ (center_u - u), (center_v - int(v_f)) };
							float squared_error = diff.squaredNorm();
							float error = sqrtf(squared_error);
							// Calculate weight with Tukey norm
							float weight = 0.1;
							if (error > std::numeric_limits<float>::min())
								weight = (TukeyNorm(error) / squared_error);
							//cout << weight <<endl;
							*segment_probability_f = *segment_probability_f * weight * score_match;
						}
					}
					else
						*segment_probability_f = min_probility_;
				}
			}
			else {

				float *segment_probability_f = &segment_probabilities_f->back();
				float *segment_probability_b = &segment_probabilities_b->back();
				*segment_probability_f = 1.0f;
				*segment_probability_b = 1.0f;
				int col_index = 0;
				int segment_idx = 0;
				for (; u <= u_end; ++u, v_f += v_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(--segment_probability_f) = 1.0f;
						*(--segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					
					int mag = 0;
					mag = int(mag_.at<uchar>(int(v_f) - rect_roi_.y, u - rect_roi_.x));
					//cout << mag << endl;
					//边缘提取的响应值
					if (mag > 50.0)
					{
						float angle_real = ori_.at<float>(int(v_f) - rect_roi_.y, u - rect_roi_.x);
						angle_real = angle_real * 180 / CV_PI;
						//angle_real = 270.0 - angle_real;
						if (angle_real > 180)
							angle_real = angle_real - 360.0;
						float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
						float distance_angle = (angle_real - angle_virtual);
						float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
						if (score_match > 0.3)
						{
							Eigen::Vector2f diff{ (center_u - u), (center_v - int(v_f)) };
							float squared_error = diff.squaredNorm();
							float error = sqrtf(squared_error);
							// Calculate weight with Tukey norm
							float weight = 0.1;
							if (error > std::numeric_limits<float>::min())
								weight = (TukeyNorm(error) / squared_error);
							//cout << weight << endl;

							*segment_probability_f = *segment_probability_f * weight * score_match;
						}
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
				float *segment_probability_b = segment_probabilities_b->data();
				*segment_probability_f = 1.0f;
				*segment_probability_b = 1.0f;
				int col_index = 0;
				int segment_idx = 0;
				for (; v <= v_end; ++v, u_f += u_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(++segment_probability_f) = 1.0f;
						*(++segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					int mag = 0;
					mag = mag_.at<uchar>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
					//cout << mag << endl;
					if (mag > 50.0)
					{	
						float angle_real = ori_.at<float>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
						angle_real = angle_real * 180 / CV_PI;
						//angle_real = 270.0 - angle_real;
						if (angle_real > 180)
							angle_real = angle_real - 360.0;
						float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
						float distance_angle = (angle_real - angle_virtual);
						float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
						if (score_match > 0.3)
						{
							Eigen::Vector2f diff{ (center_u - int(u_f)), (center_v - v) };
							float squared_error = diff.squaredNorm();
							float error = sqrtf(squared_error);
							// Calculate weight with Tukey norm
							float weight = 0.1;
							if (error > std::numeric_limits<float>::min())
								weight = (TukeyNorm(error) / squared_error);
							*segment_probability_f = *segment_probability_f * weight * score_match;
						}
					   
					   
					}
					else
						*segment_probability_f = min_probility_;
				}
			}
			else {
				float *segment_probability_f = &segment_probabilities_f->back();
				float *segment_probability_b = &segment_probabilities_b->back();
				*segment_probability_f = 1.0f;
				*segment_probability_b = 1.0f;
				int col_index = 0;
				int segment_idx = 0;
				for (; v <= v_end; ++v, u_f += u_step, ++segment_idx) {
					if (segment_idx == scale_) {
						*(--segment_probability_f) = 1.0f;
						*(--segment_probability_b) = 1.0f;
						segment_idx = 0;
					}
					
					int mag = 0;
					mag = int(mag_.at<uchar>(v - rect_roi_.y, int(u_f) - rect_roi_.x));
					//cout<< mag <<endl;
					if (mag > 50.0)
					{			
						float angle_real = ori_.at<float>(v - rect_roi_.y, int(u_f) - rect_roi_.x);
						angle_real = angle_real * 180 / CV_PI;
						//angle_real = 270.0 - angle_real;
						if (angle_real > 180)
							angle_real = angle_real - 360.0;
						float angle_virtual = atan2(normal_v, normal_u) * 180 / CV_PI;
						float distance_angle = (angle_real - angle_virtual);
						float score_match = fabs(cos(distance_angle * CV_PI / 180.0));
						if (score_match > 0.3)
						{
							Eigen::Vector2f diff{ (center_u - int(u_f)), (center_v - v) };
							float squared_error = diff.squaredNorm();
							float error = sqrtf(squared_error);
							// Calculate weight with Tukey norm
							float weight = 0.1;
							if (error > std::numeric_limits<float>::min())
								weight = (TukeyNorm(error) / squared_error);
							*segment_probability_f = *segment_probability_f * weight * score_match;
						}
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
		//cout<<"000"<<endl;
		//cv::imshow("show", imshow);
		//cv::waitKey(0);
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
		//计算分布


		std::vector<float>::const_iterator segment_probabilities_f_it;
		std::vector<float>::const_iterator function_lookup_f_it;
		std::vector<float>::const_iterator function_lookup_f_distribution_it;

		distribution->resize(distribution_length_);
		float distribution_area = 0.0f;

		// Loop over entire distribution and start values of segment probabilities
		auto segment_probabilities_f_it_start = begin(*segment_probabilities_f);

		function_lookup_f_distribution_it = begin(function_lookup_edge_distrbution_all);
		for (auto distribution_it = begin(*distribution);
			distribution_it != end(*distribution);
			++distribution_it, ++segment_probabilities_f_it_start, ++function_lookup_f_distribution_it) {
			*distribution_it = 1.0f;
			// Loop over values of segment probabilities and corresponding lookup values
			segment_probabilities_f_it = segment_probabilities_f_it_start;
			function_lookup_f_it = begin(function_lookup_edge_);
			
			for (; function_lookup_f_it != end(function_lookup_edge_);
				++function_lookup_f_it, 
				++segment_probabilities_f_it) {
				
				//*distribution_it *= *segment_probabilities_f_it * *function_lookup_f_it;
				*distribution_it *= *segment_probabilities_f_it;
				//std::cout << "distribution_it edge: " << *distribution_it << std::endl;
			}
			//整个分布的高斯
			//std::cout << "distribution_it edge: " << *distribution_it << std::endl;
			//*distribution_it *= *function_lookup_f_distribution_it;
			if (*distribution_it == 0)
			{
				*distribution_it = min_probility_;
			}
			distribution_area += *distribution_it;
			
		}

		// Normalize distribution
		for (auto &probability_distribution : *distribution) {
			if (distribution_area == 0)
			{
				probability_distribution = probability_distribution;
			}
			
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
		*variance = std::max(distribution_variance, min_variance_);
		*standard_deviation = std::sqrt(*variance);

		//cout << "*variance edge " << *variance << endl;
		//cout << "mean_from_begin edge " << mean_from_begin << endl;
		//cout << "distribution_length_minus_1_half_ " << distribution_length_minus_1_half_ << endl;
		//cout << "mean edge " << *mean << endl;
		return true;
	}

	bool LineModality::CalculateSegmentProbabilities(
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

	void LineModality::MultiplyPixelColorProbability(const cv::Vec3b &pixel_color,
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

	void LineModality::CalculateDistribution(
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

	void LineModality::CalculateDistributionMoments(
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

	void LineModality::ShowAndSaveImage(const std::string &title, int save_index,
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

	void LineModality::VisualizePointsCameraImage(const std::string &title,
		int save_index) const {
		cv::Mat visualization_image;
		camera_ptr_->image().copyTo(visualization_image);
		DrawPoints(cv::Vec3b{ 24, 184, 234 }, &visualization_image);
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
	}

	void LineModality::VisualizePointsHistogramImage(const std::string &title,
		int save_index) const {
		cv::Mat visualization_image(camera_ptr_->image().size(), CV_8UC3);
		DrawProbabilityImage(cv::Vec3b{ 255, 255, 255 }, &visualization_image);
		DrawPoints(cv::Vec3b{ 24, 184, 234 }, &visualization_image);
		ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
	}

	void LineModality::VisualizePointsOcclusionMask(const std::string &title,
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

	void LineModality::VisualizeLines(const std::string &title,
		int save_index){
		//cv::Mat visualization_image(camera_ptr_->image().size(), CV_8UC3);
		cv::Mat visualization_image = camera_ptr_->image().clone();

		cv::Mat sobel_img;
		cv::cvtColor(visualization_image, sobel_img, CV_BGR2GRAY);
		cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
		ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
		ed->params.GradientThresholdValue = 80;    //80
		ed->params.AnchorThresholdValue = 3;    //3
		vector<cv::Vec6d> ellipses;
		vector<cv::Vec4f> lines;
		lines.clear();
		//you should call this before detectLines() and detectEllipses()
		//cv::cvtColor(canny_mat, input_img, CV_BGR2GRAY);
		ed->detectEdges(sobel_img);
		// Detect lines
		ed->detectLines(lines);
		cv::Mat edge_image_ed = cv::Mat::zeros(sobel_img.size(), CV_8UC1);
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
		int lsdNFeatures = 100;
		//cout << "filter lines" << endl;
		/*if (keylines.size()>lsdNFeatures)
		{
			sort(keylines.begin(), keylines.end(), sort_lines_by_response());
			keylines.resize(lsdNFeatures);
			for (int i = 0; i<lsdNFeatures; i++)
				keylines[i].class_id = i;
		}*/
		//cv::Mat combine_img = cv::imread("edge_image_ed111.jpg",1);
		for (int i = 0; i < keylines.size(); i++)
		{
			cv::line_descriptor::KeyLine line = keylines[i];
			cv::line(edge_image_ed, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(255), 1);
			//cv::line(combine_img, cv::Point(line.startPointX, line.startPointY), cv::Point(line.endPointX, line.endPointY), cv::Scalar(0,0,255), 1);			
		}
		cv::cvtColor(edge_image_ed, edge_image_ed, CV_GRAY2BGR);
		//cv::Mat visualization_image;
		//cv::cvtColor(Hog_img, visualization_image,cv::COLOR_GRAY2BGR);
		//cv::Mat visualization_image = Canny_Mat_.clone();
		//cv::cvtColor(visualization_image, visualization_image,CV_GRAY2BGR);
		//DrawProbabilityImage(cv::Vec3b{ 255, 255, 255 }, &visualization_image);
		DrawLines(cv::Vec3b{ 24, 184, 234 }, cv::Vec3b{ 61, 63, 179 },
			&edge_image_ed);
		ShowAndSaveImage(name_ + "_" + title, save_index, edge_image_ed);
	}

	void LineModality::DrawPoints(const cv::Vec3b &color_point,
		cv::Mat *image) const {
		for (const auto &data_line : data_lines_) {
			DrawPointInImage(data_line.center_f_camera, color_point,
				camera_ptr_->intrinsics(), image);
		}
	}

	void LineModality::DrawLines(const cv::Vec3b &color_line,
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

	void LineModality::DrawProbabilityImage(const cv::Vec3b &color_b,
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

	void LineModality::UpdateLineCentersWithCurrentPose() {
		Transform3fA body2camera_pose{ camera_ptr_->world2camera_pose() *
			body_ptr_->body2world_pose() };
		//cout<<"UpdateLineCentersWithCurrentPose"<< body_ptr_->body2world_pose().matrix() <<endl;
		for (auto &data_line : data_lines_) {
			data_line.center_f_camera = body2camera_pose * data_line.center_f_body;
		}
	}

	float LineModality::MinAbsValueWithSignOfValue1(float value_1,
		float abs_value_2) {
		if (std::abs(value_1) < abs_value_2)
			return value_1;
		else
			return sgnf(value_1) * abs_value_2;
	}

	bool LineModality::IsSetup() const {
		if (!set_up_) {
			std::cerr << "Set up region modality " << name_ << " first" << std::endl;
			return false;
		}
		return true;
	}

	float LineModality::tukey_cost(float x, float c)
	{
		if (fabs(x) <= c) {
			return (c*c / 6)*((1 - (x / c)*(x / c))*(1 - (x / c)*(x / c))*(1 - (x / c)*(x / c)));
		}
		else {
			return 0;
		}
	}

	void LineModality::checkLineExtremes(cv::Vec4f& extremes, cv::Size imageSize)
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
	bool LineModality::extractEdgeDescriptor(cv::Mat& desc, const cv::Mat& im, cv::Point pt,
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

	bool LineModality::expendOri(float &max_score, const cv::Mat& im, cv::Point pt,
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
	void LineModality::GMMCal(cv::Mat frame)
	{		
		//fgMask.create(cv::Size{ frame.rows, frame.cols }, CV_8UC1);
		//fgMask.setTo(cv::Scalar{ 0 });
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

	float LineModality::TukeyNorm(float error) {
		if (std::abs(error) <= tukey_norm_constant_)
			return powf(tukey_norm_constant_, 2.0f) / 6.0f *
			(1.0f - powf(1.0f - powf(error / tukey_norm_constant_, 2.0f), 3.0f));
		else
			return powf(tukey_norm_constant_, 2.0f) / 6.0f;
	}
}  // namespace srt3d
