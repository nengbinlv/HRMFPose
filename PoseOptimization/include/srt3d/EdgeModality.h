
// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef OBJECT_TRACKING_INCLUDE_SRT3D_EDGE_MODALITY_H_
#define OBJECT_TRACKING_INCLUDE_SRT3D_EDGE_MODALITY_H_

#include <srt3d/body.h>
#include <srt3d/camera.h>
#include <srt3d/common.h>
#include <srt3d/edge_model.h>
#include <srt3d/occlusion_renderer.h>
#include <srt3d/modality.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

namespace srt3d {

	// Class that implements all functionality for a region modality
	// including correspondence search, calculation of the gradient vector and
	// hessian matrix, pose optimization, calculation of color histograms, and
	// visualization. Also the modality is able to consider occlusions using an
	// occlusion mask.
	class EdgeModality{
	private:
		// Data for correspondence line calculated during CalculateCorrespondences
		using DataLine = struct DataLine {
			Eigen::Vector3f center_f_body;
			Eigen::Vector3f center_f_camera;
			float center_u = 0.0f;
			float center_v = 0.0f;
			float normal_u = 0.0f;
			float normal_v = 0.0f;
			float delta_r = 0.0f;
			float normal_component_to_scale = 0.0f;
			float continuous_distance = 0.0f;
			std::vector<float> distribution;
			float mean = 0.0f;
			float standard_deviation = 0.0f;
			float variance = 0.0f;
			Eigen::VectorXf vimg_desc;
			int index_contour;
		};

	public:
		// Constructors and setup methods
		EdgeModality(const std::string &name, std::shared_ptr<Body> body_ptr,
			std::shared_ptr<EdgeModel> EdgeModel_ptr,
			std::shared_ptr<Camera> camera_ptr);
		bool SetUp();

		// Setters for general distribution
		void set_n_lines(int n_lines);
		void set_function_amplitude(float function_amplitude);
		void set_function_slope(float function_slope);
		void set_learning_rate(float learning_rate);
		void set_function_length(int function_length);
		void set_distribution_length(int distribution_length);
		void set_scales(const std::vector<int> &scales);
		void set_n_newton_iterations(int n_newton_iterations);
		void set_min_continuous_distance(float min_continuous_distance);

		// Setters for histogram calculation
		bool set_n_histogram_bins(int n_histogram_bins);
		void set_learning_rate_f(float learning_rate_f);
		void set_learning_rate_b(float learning_rate_b);
		void set_unconsidered_line_length(float unconsidered_line_length);
		void set_considered_line_length(float considered_line_length);

		// Setters for optimization
		void set_tikhonov_parameter_rotation(float tikhonov_parameter_rotation);
		void set_tikhonov_parameter_translation(float tikhonov_parameter_translation);

		// Setters for occlusion handling
		void UseOcclusionHandling(
			std::shared_ptr<OcclusionRenderer> occlusion_renderer_ptr);
		void DoNotUseOcclusionHandling();

		// Setters for general visualization settings
		void set_display_visualization(bool display_visualization);
		void StartSavingVisualizations(const std::experimental::filesystem::path &save_directory);
		void StopSavingVisualizations();

		// Setters to turn on individual visualizations
		void set_visualize_lines_correspondence(bool visualize_lines_correspondence);
		void set_visualize_points_occlusion_mask_correspondence(
			bool visualize_points_occlusion_mask_correspondence);
		void set_visualize_points_pose_update(bool visualize_points_pose_update);
		void set_visualize_points_histogram_image_pose_update(
			bool visualize_points_histogram_image_pose_update);
		void set_visualize_points_result(bool visualize_points_result);
		void set_visualize_points_histogram_image_result(
			bool visualize_points_result);

		// Main methods
		bool StartModality();
		bool CalculateBeforeCameraUpdate();
		bool CalculateCorrespondences(int corr_iteration);
		bool VisualizeCorrespondences(int save_idx);
		bool CalculatePoseUpdate(int corr_iteration, int update_iteration);
		bool VisualizePoseUpdate(int save_idx);
		bool VisualizeResults(int save_idx);

		// Getters data
		const std::string &name() const;
		std::shared_ptr<Body> body_ptr() const;
		std::shared_ptr<EdgeModel> EdgeModel_ptr() const;
		std::shared_ptr<Camera> camera_ptr() const;
		std::shared_ptr<OcclusionRenderer> occlusion_renderer_ptr() const;

		// Getters visualization and state
		bool imshow_correspondence() const;
		bool imshow_pose_update() const;
		bool imshow_result() const;
		bool set_up() const;

		void PrecalculateExtractEdge();
		cv::Matx44f  exp(cv::Matx61f xi);
		cv::Matx33f axiator(cv::Vec3f a);
	public:
		// Helper methods for precalculation of internal data
		void PrecalculateFunctionLookup();
		void PrecalculateDistributionVariables();
		void PrecalculateHistogramBinVariables();
		void SetImshowVariables();

		// Helper methods for precalculation of referenced data and changing data
		void PrecalculateBodyVariables();
		void PrecalculateCameraVariables();
		void PrecalculatePoseVariables();
		void PrecalculateScaleDependentVariables(int corr_iteration);

		// Helper methods for histogram calculation
		void AddLinePixelColorsToTempHistograms();
		void AddPixelColorToHistogram(const uchar &pixel_color,
			std::vector<float> *enlarged_histogram) const;
		bool CalculateHistogram(float learning_rate,
			const std::vector<float> &temp_histogram,
			std::vector<float> *histogram);

		// Helper methods for CalculateCorrespondences
		void CalculateBasicLineData(const EdgeModel::PointData &data_point,
			DataLine *data_line) const;
		bool IsLineValid(float u, float v, float continuous_distance) const;
		bool CalculateSegmentProbabilities(
			float center_u, float center_v, float normal_u, float normal_v,
			std::vector<float> *segment_probabilities_f,
			std::vector<float> *segment_probabilities_b,
			float *normal_component_to_scale, float *delta_r) const;
		void MultiplyPixelColorProbability(const cv::Vec3b &pixel_color,
			float *probability_f,
			float *probability_b) const;

		void CalculateDistribution(const std::vector<float> &segment_probabilities_f,
			const std::vector<float> &segment_probabilities_b,
			std::vector<float> *distribution) const;
		void CalculateDistributionMoments(const std::vector<float> &distribution,
			float *mean, float *standard_deviation,
			float *variance) const;


		bool EdgeModality::CalculateEdgeSearchLineDistribution(float center_u, float center_v,
			float normal_u, float normal_v, std::vector<float> *distribution);

		// Helper methods for visualization
		void ShowAndSaveImage(const std::string &title, int save_index,
			const cv::Mat &image) const;
		void VisualizePointsCameraImage(const std::string &title,
			int save_index) const;
		void VisualizePointsHistogramImage(const std::string &title,
			int save_index) const;
		void VisualizePointsOcclusionMask(const std::string &title,
			int save_index) const;
		void VisualizeLines(const std::string &title, int save_index) const;
		void DrawPoints(const cv::Vec3b &color_point, cv::Mat *image) const;
		void DrawLines(const cv::Vec3b &color_line,
			const cv::Vec3b &color_high_probability, cv::Mat *image) const;
		void DrawProbabilityImage(const cv::Vec3b &color_b,
			cv::Mat *probability_image) const;
		void UpdateLineCentersWithCurrentPose();

		// Other helper methods
		static float MinAbsValueWithSignOfValue1(float value_1, float abs_value_2);
		bool IsSetup() const;

		// Internal data objects
		std::string name_;
		std::vector<float> temp_histogram_f_;
		std::vector<float> temp_histogram_b_;
		std::vector<float> histogram_f_;
		std::vector<float> histogram_b_;
		std::vector<DataLine> data_lines_;

		// Pointers to referenced objects
		std::shared_ptr<Body> body_ptr_ = nullptr;
		std::shared_ptr<EdgeModel> EdgeModel_ptr_ = nullptr;
		std::shared_ptr<Camera> camera_ptr_ = nullptr;
		std::shared_ptr<OcclusionRenderer> occlusion_renderer_ptr_ = nullptr;

		// Parameters for general distribution
		int n_lines_ = 200;                   //200
		float function_amplitude_ = 0.36f;      //0.36f    //决定了跟踪时模型注册的稳定性  越大跟踪效果越好，但容易陷入局部最优
		float function_slope_ = 0.0f;          //0.0f
		float learning_rate_ = 1.3f;             //1.3f
		int function_length_ = 3;                  //8 
		int distribution_length_ = 18;             //12  20
		//由于sobel边缘检测算子性质决定了其不能超过2
		std::vector<int> scales_ = {1, 1, 1, 1, 1};      //{5, 2, 2, 1}  1, 1, 1, 1
		int n_newton_iterations_ = 3;             //3
		float min_continuous_distance_ = 6.0f;       //6.0f

		//30.0f, 20.0f, 15.0f, 10.0f ; 50.0f, 35.0f, 25.0f, 10.0f ;40.0f, 25.0f, 18.0f, 12.0f, 8.0f
		//50.0f, 25.0f, 16.0f, 12.0f, 8.0f
		std::vector<float> distribution_length_vector_{ 20.0f, 15.0f ,10.0f, 8.0f, 6.0f };  //原为: 25.0f, 15.0f ,10.0f, 8.0f, 6.0f
		float min_probility_ = 0.00001;
		float variance_{};
		//该参数的大小对位姿变动的大小有影响
		std::vector<float> standard_deviations_{ 2.50f, 1.50f, 1.0, 0.5f, 0.2f };   //2.50f, 1.50f, 0.35f, 0.1f, 0.09f  //////  0.10f, 0.09f, 0.08f, 0.07f, 0.05f
		//15.0f, 5.0f, 3.5f, 1.5f // 5.0f, 5.0f, 3.5f, 1.5f
		//正太分布的均值和方差
		float sigma_ = 1.0f;   //1.0
		float mu_ = 0.0f;

		// 语义边缘权重
		float semantic_weight = 0.2f;  // 0.2f
		//sigma越大，曲线越扁平，越小，则越瘦高

		//最好设置可变的分布
		//当误差较大时分布扁平，当误差较小时，分布瘦高

		std::vector<float> sigma_global_{ 10.0f, 5.0f, 1.5f, 1.0f, 0.8f};          //15.0f, 5.0f, 3.5f, 1.5f, 1.4f

		//std::vector<float> sigma_global_{ 1.5f, 1.5f, 1.5f, 1.5f };
		//是否基于偏差向量过滤误匹配点
		int flag_use_filter = 0;
		// Parameters for histogram calculation
		int n_histogram_bins_ = 32;  //32
		int histogram_bitshift_ = 3; //3
		float learning_rate_f_ = 0.2f;
		float learning_rate_b_ = 0.2f;
		float unconsidered_line_length_ = 0.0f;
		
		float considered_line_length_ = 3.0f;     //18.0f
		//边缘的考虑长度///
		float considered_line_length_ncc_ = 7.0f;

		// Parameters for optimization
		float tikhonov_parameter_rotation_ = 5000.0f;
		float tikhonov_parameter_translation_ = 500000.0f;
		Eigen::Matrix<float, 6, 6> tikhonov_matrix_;

		// Parameters for occlusion handling
		bool use_occlusion_handling_ = false;

		// Parameters for general visualization settings
		bool display_visualization_ = true;
		bool save_visualizations_ = false;
		std::experimental::filesystem::path save_directory_;

		// Parameters to turn on individual visualizations
		bool visualize_lines_correspondence_ = false;
		bool visualize_points_occlusion_mask_correspondence_ = false;
		bool visualize_points_pose_update_ = false;
		bool visualize_points_histogram_image_pose_update_ = false;
		bool visualize_points_result_ = false;
		bool visualize_points_histogram_image_result_ = false;

		// State variables (internal data)
		bool imshow_correspondence_ = false;
		bool imshow_pose_update_ = false;
		bool imshow_result_ = false;
		bool set_up_ = false;

		// Precalculated variables for smoothed step function lookup (internal data)
		std::vector<float> function_lookup_f_;
		std::vector<float> function_lookup_b_;
		std::vector<float> function_lookup_edge_;
		std::vector<float> function_lookup_edge_distrbution_all;


		// Precalculated variables for distributions (internal data)
		int line_length_in_segments_{};
		float distribution_length_minus_1_half_{};
		float distribution_length_plus_1_half_{};
		float min_variance_{};

		// Precalculated variables for histogram calculation (internal data)
		int n_histogram_bins_squared_{};
		int n_histogram_bins_cubed_{};

		// Precalculated variables for body (referenced data)
		uchar encoded_occlusion_id_ = 0;

		// Precalculated variables for camera (referenced data)
		float fu_{};
		float fv_{};
		float ppu_{};
		float ppv_{};
		int image_width_minus_1_{};
		int image_height_minus_1_{};
		int image_width_minus_2_{};
		int image_height_minus_2_{};

		// Precalculated variables for poses (continuously changing)
		Transform3fA body2camera_pose_;
		Eigen::Matrix3f body2camera_rotation_;
		Eigen::Matrix<float, 2, 3> body2camera_rotation_xy_;

		// Precalculate variables depending on scale (continuously changing)
		int scale_{};
		float fscale_{};
		int line_length_{};
		int line_length_minus_1_{};
		float line_length_minus_1_half_{};
		float line_length_half_minus_1_{};
		cv::Mat Canny_Mat_;
	public:
			Eigen::Matrix<float, 6, 1> gradient_edge;
			Eigen::Matrix<float, 6, 6> hessian_edge;
			float error_count_ = 0 ;

			float tukey_cost(float x, float c);
			void checkLineExtremes(cv::Vec4f& extremes, cv::Size imageSize);

			struct sort_lines_by_response
			{
				inline bool operator()(const cv::line_descriptor::KeyLine& a, const cv::line_descriptor::KeyLine& b) {
					return (a.response > b.response);
				}
			};


			cv::Mat line_angle;

			cv::Mat mag_;
			cv::Mat pre_edge_;
			cv::Mat mag_temp;
			cv::Mat ori_;

			bool EdgeModality::CalculateSegmentProbabilities_edge(
				float center_u, float center_v, float normal_u, float normal_v,
				std::vector<float> *segment_probabilities_f,
				std::vector<float> *segment_probabilities_b,
				float *normal_component_to_scale, float *delta_r, std::vector<float> *distribution, float *mean,
				float *standard_deviation, float *variance, Eigen::VectorXf vimg_desc);

			bool EdgeModality::extractEdgeDescriptor(cv::Mat& desc, const cv::Mat& im, cv::Point pt,
				double edge_dir, unsigned int window_size) const;
			bool EdgeModality::expendOri(float &max_score, const cv::Mat& im, cv::Point pt,
				double edge_dir, unsigned int window_size, float u, float v, float normal_u, float normal_v) const;
			void GMMCal(cv::Mat frame);
			cv::Ptr<cv::BackgroundSubtractorMOG2>  mog2 = cv::createBackgroundSubtractorMOG2();

			cv::Mat fgMask;
			bool preGMM(float center_u, float center_v, float normal_u, float normal_v, int index_line, std::vector<cv::Vec3b> *value) const;

			int dateline_index;
			cv::Mat show_gmm;

			//设计两种权重的系数
			float weight_response = 1.0f;
			float weight_orientation = 1.0f;

			float tukey_norm_constant_ = 30.0f;  //30.0f
			//30.0f, 25.0f, 15.0f, 10.0f
			std::vector<float> tukey_norm_constant_vector_ = { 25.0f, 20.0f, 15.0f, 10.0f, 9.0f };   //原来：30.0f, 25.0f, 15.0f, 10.0f, 9.0f
			//基于hog特征做局部梯度方向和幅值的匹配。通过构建一个向量实现。
			float weight_ncc = 0.0f;
			cv::Mat Hog_img;

			void CalShape(std::vector<cv::Point> contourPts1, std::vector<cv::Point> contourPts2);
			float shape_cost_ = 0;
			float match_ratio_ = 0;
			float TukeyNorm(float error);

			cv::Rect rect_roi_;
			//EdgeModel::TemplateView *template_view;
			float mean_before_ = 0;
			vector<float> mean_vector_;

			void AddEdgePixelGradientForCorrelation();
			std::vector<std::vector<cv::Vec3b>> SearchLinesGradient_;
			std::vector<Eigen::Vector3f> lastFrame_CenterPoints_;
			std::vector<Eigen::Vector2f> lastFrame_CenterNormal_;
			std::vector<int> flag_lines_use_;
			std::vector<std::vector<float>> match_ratio_of_lastFrame_;
			int index_for_match_lastFrame_;
			int flag_tracking_success_;

			std::vector<cv::Point2f> points_lk_;
			cv::Mat Last_Frame;
			cv::Mat Lasr_Frame_Search_Line_img;
			std::vector<cv::Mat> last_mat_vector;
			std::vector<cv::Mat> current_mat_vector;

			cv::Mat Last_Frame_one_line;
			std::vector<cv::Point> match_result_vector;
			int flag_num_iter_for_use;
			/*小位移才有用？？*/
			int flag_use_correlation_ = 0;
			float last_ratio = 0.0;  //0.1

			int  flag_use_histogram_edge = 0;
			void MultiplyPixelColorProbability_edge(const uchar &pixel_color,
				float *probability_f) const;
			cv::Mat gray_roi_;

			float th_mag = 0.0;  //0.001

			void DrawLines_fb(const cv::Vec3b &color_line,
				const cv::Vec3b &color_high_probability,
				cv::Mat *image) const;

			void CalculateEdgeClutter(const cv::Vec3b &color_line,
				const cv::Vec3b &color_high_probability,
				cv::Mat *image) const;

			std::vector<Eigen::Vector2f> pre_heatmap_2dpoints;
			std::vector<Eigen::Vector2f> pre_edge_vector_2d;

			std::vector<Eigen::Vector3f> points3d_body;
	};

}  // namespace srt3d

#endif  // OBJECT_TRACKING_INCLUDE_SRT3D_REGION_MODALITY_H_

