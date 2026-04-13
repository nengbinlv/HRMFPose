// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#include "srt3d/rbot_evaluator.h"
//#include <srt3d/RegionTrajectory.h>
#include <srt3d/common.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
int flag_region = 1;
int flag_edge = 1;
int flag_line = 0;

RBOTEvaluator::RBOTEvaluator(const std::string &name,
                             const std::experimental::filesystem::path &dataset_directory,
                             const std::vector<std::string> &body_names,
                             const std::vector<std::string> &sequence_names,
                             const std::vector<bool> &sequence_occlusions)
    : name_{name},
      dataset_directory_{dataset_directory},
      body_names_{body_names},
      sequence_names_{sequence_names},
      sequence_occlusions_{sequence_occlusions} {}

bool RBOTEvaluator::SetUp() {
  set_up_ = false;

  // Create run configurations
  run_configurations_.clear();
  for (size_t i = 0; i < sequence_names_.size(); ++i) {
    for (size_t j = 0; j < body_names_.size(); ++j) {
      run_configurations_.push_back(RunConfiguration{
          sequence_names_[i], sequence_occlusions_[i], body_names_[j]});
    }
  }

  // Read poses
  if (!ReadPosesRBOTDataset(dataset_directory_ / "poses_first.txt",
                            &poses_gt_first_))
    return false;
  if (!ReadPosesRBOTDataset(dataset_directory_ / "poses_second.txt",
                            &poses_gt_second_))
    return false;

  // Generate models
  GenerateModels();

  set_up_ = true;
  return true;
}

void RBOTEvaluator::set_translation_error_threshold(
    float translation_error_threshold) {
  translation_error_threshold_ = translation_error_threshold;
}

void RBOTEvaluator::set_rotation_error_threshold(
    float rotation_error_threshold) {
  rotation_error_threshold_ = rotation_error_threshold;
}

void RBOTEvaluator::set_visualize_all_results(bool visualize_all_results) {
  visualize_all_results_ = visualize_all_results;
}

void RBOTEvaluator::SaveResults(std::experimental::filesystem::path save_directory) {
  save_directory_ = save_directory;
  save_results_ = true;
}

void RBOTEvaluator::DoNotSaveResults() { save_results_ = false; }

void RBOTEvaluator::set_tracker_setter(
    const std::function<void(std::shared_ptr<srt3d::Tracker>)>
        &tracker_setter) {
  tracker_setter_ = tracker_setter;
}

void RBOTEvaluator::set_region_modality_setter(
    const std::function<void(std::shared_ptr<srt3d::RegionModality>)>
        &region_modality_setter) {
  region_modality_setter_ = region_modality_setter;
}

void RBOTEvaluator::set_model_setter(
    const std::function<void(std::shared_ptr<srt3d::Model>)> &model_setter) {
  model_setter_ = model_setter;
  set_up_ = false;
}

void RBOTEvaluator::set_occlusion_renderer_setter(
    const std::function<void(std::shared_ptr<srt3d::OcclusionRenderer>)>
        &occlusion_renderer_setter) {
  occlusion_renderer_setter_ = occlusion_renderer_setter;
}

bool RBOTEvaluator::Evaluate() {
  if (!set_up_) {
    std::cerr << "Set up evaluator " << name_ << " first" << std::endl;
    return false;
  }

  // Evaluate all run configuration
  results_.resize(run_configurations_.size());
  if (visualize_all_results_) {
    auto renderer_geometry_ptr{std::make_shared<srt3d::RendererGeometry>("rg")};
    renderer_geometry_ptr->SetUp();
    for (size_t i = 0; i < int(run_configurations_.size()); ++i) {
      EvaluateRunConfiguration(run_configurations_[i], renderer_geometry_ptr,
                               &results_[i]);

      std::string title{run_configurations_[i].sequence_name + "_" +
                        (run_configurations_[i].occlusions ? "modeled_" : "") +
                        run_configurations_[i].body_name};
      if (save_results_) SaveFinalResult(results_[i], title);
      VisualizeFinalResult(results_[i], title);
    }
  } else {
    std::vector<std::shared_ptr<srt3d::RendererGeometry>>
        renderer_geometry_ptrs(omp_get_max_threads());
    for (auto &renderer_geometry_ptr : renderer_geometry_ptrs) {
      renderer_geometry_ptr = std::make_shared<srt3d::RendererGeometry>("rg");
      renderer_geometry_ptr->SetUp();
    }
#pragma omp parallel for
    for (int i = 0; i < int(run_configurations_.size()); ++i) {
      EvaluateRunConfiguration(run_configurations_[i],
                               renderer_geometry_ptrs[omp_get_thread_num()],
                               &results_[i]);

      std::string title{run_configurations_[i].sequence_name + "_" +
                        (run_configurations_[i].occlusions ? "modeled_" : "") +
                        run_configurations_[i].body_name};
      if (save_results_) SaveFinalResult(results_[i], title);
#pragma omp critical
      VisualizeFinalResult(results_[i], title);
    }
  }

  // Calculate final results
  CalculateAverageResult(results_, &final_result_);
  if (save_results_) SaveFinalResult(final_result_, "all_sequences_all_bodies");
  VisualizeFinalResult(final_result_, "all_sequences_all_bodies");
  return true;
}

float RBOTEvaluator::tracking_success() {
  return final_result_.tracking_success;
}

void RBOTEvaluator::EvaluateRunConfiguration(
    RunConfiguration run_configuration,
    std::shared_ptr<srt3d::RendererGeometry> renderer_geometry_ptr,
    DataResult *average_result) {
  auto tracker_ptr{std::make_shared<srt3d::Tracker>("tracker")};
  SetUpTracker(run_configuration, renderer_geometry_ptr, tracker_ptr);
  ResetBody(tracker_ptr, 0);
  if (run_configuration.occlusions) ResetOcclusionBody(tracker_ptr, 0);

  // Iterate over all frames
  std::vector<DataResult> results(kNFrames_);
  for (int i = 0; i < kNFrames_; ++i) {
    results[i].frame_index = i;
    ExecuteMeasuredTrackingCycle(tracker_ptr, i, &results[i].execution_times);

    // Calculate results for main body
	//cout<<"EvaluateRunConfiguration:"<<tracker_ptr->region_modality_ptrs()[0]->body_ptr()->body2world_pose().matrix()<<endl;
    CalculatePoseResults(
        tracker_ptr->region_modality_ptrs()[0]->body_ptr()->body2world_pose(),
        poses_gt_first_[i + 1], &results[i]);
    if (visualize_all_results_)
      VisualizeFrameResult(results[i], run_configuration.sequence_name + ": " +
                                           run_configuration.body_name);
    if (results[i].tracking_success == 0.0f) ResetBody(tracker_ptr, i + 1);

    // Calculate results for occluding body
    if (run_configuration.occlusions) {
      DataResult occlusion_result;
      CalculatePoseResults(
          tracker_ptr->region_modality_ptrs()[1]->body_ptr()->body2world_pose(),
          poses_gt_second_[i + 1], &occlusion_result);
      if (occlusion_result.tracking_success == 0.0f)
        ResetOcclusionBody(tracker_ptr, i + 1);
    }
  }
  CalculateAverageResult(results, average_result);
}

void RBOTEvaluator::SetUpTracker(
    const RunConfiguration &run_configuration,
    std::shared_ptr<srt3d::RendererGeometry> renderer_geometry_ptr,
    std::shared_ptr<srt3d::Tracker> tracker_ptr) {
  renderer_geometry_ptr->ClearBodies();

  // Init camera
  auto camera_ptr{std::make_shared<srt3d::LoaderCamera>(
      "camera", dataset_directory_ / run_configuration.body_name / "frames_regular",  //frames_light  frames_occlusion
      kRBOTIntrinsics, run_configuration.sequence_name, 0, 4)};  //0  2

  cv::Mat1f distortion1_(1, 5);
  //distortion1_ << -0.4159719580468175, 0.09641148132927757, -0.001169350966459926, 0.00190663251158204, 2.393780916474181;
  distortion1_ << 0, 0, 0, 0, 0;
  camera_ptr->distortion_coeff_ = distortion1_;

  // Init Viewer
  if (visualize_all_results_) {
    auto viewer_ptr{std::make_shared<srt3d::NormalViewer>(
        "viewer", camera_ptr, renderer_geometry_ptr)};
	//imwrite
	viewer_ptr->StartSavingImages(dataset_directory_/ run_configuration.body_name/"ours_result_regular","png");
    tracker_ptr->AddViewer(viewer_ptr);
  }

  // Init body
  std::experimental::filesystem::path geometry_path{dataset_directory_ /
                                      run_configuration.body_name /
                                      (run_configuration.body_name + ".obj")};

  /*与参考论文不一样 0.3f*/
  auto body_ptr{std::make_shared<srt3d::Body>(
      run_configuration.body_name, geometry_path, 0.001f, true, false, 0.3f,  //0.3   //原为0.3
      srt3d::Transform3fA::Identity(), 7)};
  renderer_geometry_ptr->AddBody(body_ptr);

  // Init model
  auto model_ptr{std::make_shared<srt3d::Model>(
      "model", body_ptr, dataset_directory_ / run_configuration.body_name,
      run_configuration.body_name + "_model.bin", 0.8, 4, 200, false, 2000)};    //原为0.8

  auto model_ptr_edge{ std::make_shared<srt3d::EdgeModel>(
	  "model", body_ptr, dataset_directory_ / run_configuration.body_name,
	  run_configuration.body_name + "_edge_model.bin", 0.8, 4, 300, false, 2000) };  //原为0.8

  auto model_ptr_line{ std::make_shared<srt3d::LineModel>(
	  "model", body_ptr, dataset_directory_ / run_configuration.body_name,
	  run_configuration.body_name + "_line_model.bin", 0.8, 4, 300, false, 2000) };


  model_setter_(model_ptr);

  // Init region modality
  auto region_modality_ptr{std::make_shared<srt3d::RegionModality>(
      "region_modality", body_ptr, model_ptr, camera_ptr)};

  auto edge_modality_ptr{ std::make_shared<srt3d::EdgeModality>(
	  "edge_modality", body_ptr, model_ptr_edge, camera_ptr) };

  auto line_modality_ptr{ std::make_shared<srt3d::LineModality>(
	  "line_modality", body_ptr, model_ptr_line, camera_ptr) };
  //********************可视化*********************
#if 1
  //region_modality_ptr->set_visualize_lines_correspondence(true);
  //edge_modality_ptr->set_visualize_lines_correspondence(true);
  //edge_modality_ptr->StartSavingVisualizations(dataset_directory_ / run_configuration.body_name/"edge_tracking_process_lines_gray");
 // region_modality_ptr->StartSavingVisualizations(dataset_directory_ / run_configuration.body_name/"region_tracking_process_lines_gray");
  //edge_modality_ptr->set_n_histogram_bins(64);

  //edge_modality_ptr->set_visualize_points_result(true);
  //region_modality_ptr->set_visualize_points_result(true);
  //line_modality_ptr->set_visualize_lines_correspondence(true);
#endif

  region_modality_setter_(region_modality_ptr);

  //********************添加类*********************
  tracker_ptr->AddRegionModality(region_modality_ptr);
  if (flag_edge)
  {
	  tracker_ptr->AddEdgeModality(edge_modality_ptr);
  }
  if (flag_line)
  {
	  tracker_ptr->AddLineModality(line_modality_ptr);
  }

  if (run_configuration.occlusions) {
    // Init occlusion body
    auto occlusion_body_ptr{std::make_shared<srt3d::Body>(
        "squirrel_small", dataset_directory_ / "14.obj", 0.001f,
        true, false, 0.3f, srt3d::Transform3fA::Identity(), 1)};   //原为0.3
    renderer_geometry_ptr->AddBody(occlusion_body_ptr);

    // Init occlusion model
    auto occlusion_model_ptr{std::make_shared<srt3d::Model>(
        "occlusion_model", occlusion_body_ptr, dataset_directory_,
        "squirrel_small_model_region.bin", 0.8, 4, 200, false, 2000)};   //原为0.8

	auto occlusion_model_ptr_edge{ std::make_shared<srt3d::EdgeModel>(
		"occlusion_model_edge", occlusion_body_ptr, dataset_directory_,
		"squirrel_small_model_edge.bin", 0.8, 4, 300, false, 2000) };    //原为0.8

	/*auto occlusion_model_ptr_line{ std::make_shared<srt3d::LineModel>(
		"occlusion_model_line", occlusion_body_ptr, dataset_directory_,
		"squirrel_small_model.bin", 0.8, 4, 200, false, 2000) };*/

    model_setter_(occlusion_model_ptr);

    // Init occlusion region modality
    auto occlusion_region_modality_ptr{std::make_shared<srt3d::RegionModality>(
        "occlusion_region_modality", occlusion_body_ptr, occlusion_model_ptr,
        camera_ptr)};

	auto occlusion_region_modality_ptr_edge{ std::make_shared<srt3d::EdgeModality>(
	   "occlusion_edge_modality", occlusion_body_ptr, occlusion_model_ptr_edge,
	   camera_ptr) };

	/*auto occlusion_region_modality_ptr_line{ std::make_shared<srt3d::LineModality>(
	   "occlusion_region_modality", occlusion_body_ptr, occlusion_model_ptr_line,
	   camera_ptr) };*/

	//occlusion_region_modality_ptr->set_visualize_lines_correspondence(true);
	//occlusion_region_modality_ptr_edge->set_visualize_lines_correspondence(true);

    region_modality_setter_(occlusion_region_modality_ptr);
    tracker_ptr->AddRegionModality(occlusion_region_modality_ptr);
	if (flag_edge)
	{
		tracker_ptr->AddEdgeModality(occlusion_region_modality_ptr_edge);
	}
	if (flag_line)
	{
		//tracker_ptr->AddLineModality(occlusion_region_modality_ptr_line);
	}

    // Init occlusion renderer
    auto occlusion_renderer_ptr{std::make_shared<srt3d::OcclusionRenderer>(
        "occlusion_renderer", renderer_geometry_ptr, camera_ptr)};
    occlusion_renderer_setter_(occlusion_renderer_ptr);
    region_modality_ptr->UseOcclusionHandling(occlusion_renderer_ptr);
    occlusion_region_modality_ptr->UseOcclusionHandling(occlusion_renderer_ptr);
	if (flag_edge)
	{
		edge_modality_ptr->UseOcclusionHandling(occlusion_renderer_ptr);
		occlusion_region_modality_ptr_edge->UseOcclusionHandling(occlusion_renderer_ptr);
	}
	if (flag_line)
	{
		//line_modality_ptr->UseOcclusionHandling(occlusion_renderer_ptr);
		//occlusion_region_modality_ptr_line->UseOcclusionHandling(occlusion_renderer_ptr);
	}
  }

  tracker_setter_(tracker_ptr);
  tracker_ptr->SetUpTracker();
}

void RBOTEvaluator::ResetBody(std::shared_ptr<srt3d::Tracker> tracker_ptr,
                              int i_frame) {
  tracker_ptr->region_modality_ptrs()[0]->body_ptr()->set_body2world_pose(poses_gt_first_[i_frame]);
  tracker_ptr->region_modality_ptrs()[0]->StartModality();
  //cout<< poses_gt_first_[i_frame].matrix()<<endl; 
 
  if (flag_edge)
  {
	  tracker_ptr->edge_modality_ptrs()[0]->body_ptr()->set_body2world_pose(poses_gt_first_[i_frame]);
	  tracker_ptr->edge_modality_ptrs()[0]->StartModality();
  }
  if (flag_line)
  {
	  tracker_ptr->line_modality_ptrs()[0]->body_ptr()->set_body2world_pose(poses_gt_first_[i_frame]);
	  tracker_ptr->line_modality_ptrs()[0]->StartModality();
  }
}

void RBOTEvaluator::ResetOcclusionBody(
    std::shared_ptr<srt3d::Tracker> tracker_ptr, int i_frame) {
  tracker_ptr->region_modality_ptrs()[1]->body_ptr()->set_body2world_pose(
      poses_gt_second_[i_frame]);
  tracker_ptr->region_modality_ptrs()[1]->StartModality();
  if (flag_edge)
  {
	  tracker_ptr->edge_modality_ptrs()[1]->body_ptr()->set_body2world_pose(poses_gt_second_[i_frame]);
	  tracker_ptr->edge_modality_ptrs()[1]->StartModality();
  }
  if (flag_line)
  {
	  tracker_ptr->line_modality_ptrs()[1]->body_ptr()->set_body2world_pose(poses_gt_second_[i_frame]);
	  tracker_ptr->line_modality_ptrs()[1]->StartModality();
  }
}

void RBOTEvaluator::ExecuteMeasuredTrackingCycle(
    std::shared_ptr<srt3d::Tracker> tracker_ptr, int iteration,
    DataExecutionTimes *execution_times) {
  // Calculate before camera update and camera update
  auto begin_time{std::chrono::high_resolution_clock::now()};

  /*获取当前位姿*/

  srt3d::Transform3fA pose_pre_ = tracker_ptr->region_modality_ptrs()[0]->body_ptr()->body2world_pose();
  cv::Matx44f pose;
  cv::eigen2cv(pose_pre_.matrix(), pose);
  cv::Matx33f cur_pose_R = cv::Matx33f(pose(0, 0), pose(0, 1), pose(0, 2),
	  pose(1, 0), pose(1, 1), pose(1, 2),
	  pose(2, 0), pose(2, 1), pose(2, 2));

  /*获取第一帧图像*/
  tracker_ptr->CalculateBeforeCameraUpdate();
  //cout<< index_frame <<endl;
  if (index_frame != 0)
  {
	  float	theta = srt3d::getRDiff(pre_pose_R, cur_pose_R);
	  //cout << tracker_ptr->err_current[0] << endl;
	/*求前5帧的误差中值*/
	  _frameInfo.push_back({ theta,tracker_ptr->err_current[0] });
	  while (_frameInfo.size() > 30)
		  _frameInfo.pop_front();
  }
  pre_pose_R = cur_pose_R;
  
  execution_times->calculate_before_camera_update = ElapsedTime(begin_time);
  /*更新下一帧图像*/
  index_frame++;
  tracker_ptr->UpdateCameras();

  execution_times->start_occlusion_rendering = 0.0f;
  execution_times->calculate_correspondences = 0.0f;
  execution_times->calculate_pose_update = 0.0f;

	for (int corr_iteration = 0;
		corr_iteration < tracker_ptr->n_corr_iterations(); ++corr_iteration) {
		// Start occlusion rendering
		begin_time = std::chrono::high_resolution_clock::now();
		tracker_ptr->StartOcclusionRendering();
		execution_times->start_occlusion_rendering += ElapsedTime(begin_time);

		// Calculate correspondences
		begin_time = std::chrono::high_resolution_clock::now();
		tracker_ptr->CalculateCorrespondences(corr_iteration);
		execution_times->calculate_correspondences += ElapsedTime(begin_time);

		// Visualize correspondences
		int corr_save_idx =
			iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
		tracker_ptr->VisualizeCorrespondences(corr_save_idx);

		for (int update_iteration = 0;
			update_iteration < tracker_ptr->n_update_iterations();
			++update_iteration) {
			// Calculate pose update
			begin_time = std::chrono::high_resolution_clock::now();
			tracker_ptr->CalculatePoseUpdate(corr_iteration, update_iteration);
			execution_times->calculate_pose_update += ElapsedTime(begin_time);

			// Visualize pose update
			int update_save_idx =
				corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
			tracker_ptr->VisualizePoseUpdate(update_save_idx);
		}
	}
  

  auto dpose = tracker_ptr->pose_current[0];

  /*TODO 多个模型的误差计算*/
#if 1
  //经过上述的优化，仍然误差很大
  //进行非局部优化搜索--视点搜索
  float errT = 2.0f;   //2.0
  float thetaT = CV_PI / 30;   //8  30

  if (!_frameInfo.empty())
  {
	  thetaT = _getMedianOfLastN(_frameInfo, 5, [](const FrameInfo& v) {return v.theta; });
	  errT = _getMedianOfLastN(_frameInfo, 15, [](const FrameInfo& v) {return v.err; });

	  thetaT = thetaT * 0.85;  //0.85 0.75
	  if (isnan(thetaT))
	  {
		  thetaT = CV_PI / 30;
	  }
	  errT = errT * 1.0; //1.0
	  //cout<<"thetaT： "<< thetaT <<endl;
	  //cout << "errT： " << errT << endl;
  }  
  //cout<<"true error:  "<< tracker_ptr->err_current[0] <<endl;
  float errMin = tracker_ptr->err_current[0];
  //cout << "errMin： " << errMin << endl;
  srt3d::Transform3fA min_error_pose_;
  min_error_pose_.matrix() << dpose.R(0, 0), dpose.R(0, 1), dpose.R(0, 2), dpose.t(0),
	  dpose.R(1, 0), dpose.R(1, 1), dpose.R(1, 2), dpose.t(1),
	  dpose.R(2, 0), dpose.R(2, 1), dpose.R(2, 2), dpose.t(2);

      /*外平面旋转*/
      /*外平面旋转*/
  int iter_num = 0;
  if (errMin > errT)
  {
	  if (errMin > errT)
	  {
		  tracker_ptr->flag_use_no_local = 1;
		  //获取当前位姿的R矩阵
		  //cout << "local opti start====" << endl;
		  const auto R0 = tracker_ptr->R_[0];
		  const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;   //12
		  //cout<< N <<endl;
		  //划分角度
		  const float dc = thetaT / N;
		  //子划分
		  const int subDiv = 3;
		  const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
		  //轨迹？
		  //cout<< subRegionSize <<endl;
		  srt3d::RegionTrajectory_ traj_(cv::Size(subRegionSize, subRegionSize), dc / subDiv);

		  cv::Mat1b label = cv::Mat1b::zeros(2 * N + 1, 2 * N + 1);  //3 x 3
		  struct DSeed
		  {
			  cv::Point coord;
			  //bool  isLocalMinima;
		  };
		  std::deque<DSeed>  seeds;
		  seeds.push_back({ cv::Point(N,N)/*,true*/ });
		  label(N, N) = 1;
		  auto checkAdd = [&seeds, &label](const DSeed& curSeed, int dx, int dy) {
			  int x = curSeed.coord.x + dx, y = curSeed.coord.y + dy;
			  if (uint(x) < uint(label.cols) && uint(y) < uint(label.rows))
			  {
				  if (label(y, x) == 0)
				  {
					  label(y, x) = 1;
					  seeds.push_back({ cv::Point(x, y)/*, false*/ });
				  }
			  }
		  };
		  /*****************不断变换初值进行位姿优化***************************/
		  while (!seeds.empty())
		  {
			  auto curSeed = seeds.front();
			  seeds.pop_front();
			  checkAdd(curSeed, 1, 0);
			  checkAdd(curSeed, 0, 1);
			  checkAdd(curSeed, 0, -1);
			  checkAdd(curSeed, -1, 0);

			  //计算外平面旋转
			  auto dR = srt3d::theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);
			  //cout<< dR <<endl;
			  /*理论上还有一个绕z轴的旋转*/

			  auto dposex = dpose;  //dpose
			  dposex.R = dR * R0;
			  
			  cv::Point2f start = srt3d::dir2Theta(srt3d::viewDirFromR(dposex.R * R0.t()));
			  /*这里实际上start等于上面的输入float(curSeed.coord.x - N) * dc，float(curSeed.coord.y - N) * dc*/

			  /*平移的扰动*/
			  iter_num++;
			  int outerItrs = 1;
			  int innerItrs = 1;//4   原来为3	  
			  for (int itr = 0; itr < outerItrs*innerItrs; ++itr)
			  {
				  //获取当前最近的视点，为了获取响应视点下的数据，从而可以进行位姿迭代计算
					  //得到当前视点的投影点数据，需要得到边缘和区域的两个数据
				  //绕z轴的旋转，该如何添加？？？
				  srt3d::Transform3fA body2camera_pose_;
				  body2camera_pose_.matrix() << dposex.R(0, 0), dposex.R(0, 1), dposex.R(0, 2), dposex.t(0),
					  dposex.R(1, 0), dposex.R(1, 1), dposex.R(1, 2), dposex.t(1),
					  dposex.R(2, 0), dposex.R(2, 1), dposex.R(2, 2), dposex.t(2);
				  //重新斟酌
				  std::shared_ptr<srt3d::RegionModality> region_modality_ptr = tracker_ptr->region_modality_ptrs_[0];
				  const auto &body_ptr{ region_modality_ptr->body_ptr() };
				  body_ptr->set_body2world_pose(body2camera_pose_);

				  /*for (auto &edge_modality_ptr : edge_modality_ptrs_)
				  {
					  const auto &body_ptr{ edge_modality_ptr->body_ptr() };
					  body_ptr->set_body2world_pose(body2camera_pose_);
				  }*/

				  //计算一次看位姿变动量
				  //tracker_ptr->CalculateBeforeCameraUpdate();
				  //tracker_ptr->UpdateCameras();
				  /*TODO 在每跑一轮就计算一次是否满足小于误差*/
				  int break_flag = 0;
				  for (int corr_iteration = 3;      //迭代四次？   -2
					  corr_iteration < tracker_ptr->n_corr_iterations() - 1; ++corr_iteration) {
					  int corr_save_idx =
						  iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
					  tracker_ptr->StartOcclusionRendering();
					  begin_time = std::chrono::high_resolution_clock::now();
					  tracker_ptr->CalculateCorrespondences(corr_iteration);
					  execution_times->calculate_correspondences += ElapsedTime(begin_time);
					  tracker_ptr->VisualizeCorrespondences(corr_save_idx);
					  int flag_break = 0;
					  int flag_break_2 = 0;
					  for (int update_iteration = 0;
						  update_iteration < tracker_ptr->n_update_iterations();  // - 1
						  ++update_iteration) {
						  int update_save_idx =
							  corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
						  begin_time = std::chrono::high_resolution_clock::now();
						  tracker_ptr->CalculatePoseUpdate(corr_iteration, update_iteration);
						  execution_times->calculate_pose_update += ElapsedTime(begin_time);
						  tracker_ptr->VisualizePoseUpdate(update_save_idx);
					  }
					  /*原来的更新判断*/
					  float eps = 1e-4f;
					  if (tracker_ptr->diff_pose[0] < eps * eps)
					  {
						  //cout << "diff_pose" << endl;
						  break_flag = 1;
						  break;
					  }
					  ///============
					  dposex = tracker_ptr->pose_current[0];
					  cv::Point2f end = srt3d::dir2Theta(srt3d::viewDirFromR(dposex.R * R0.t()));
					  //没有运动
					  if (traj_.addStep(start, end))
					  {
						  //cout<<"traj_.addStep"<<endl;
						  break_flag = 1;
						  break;
					  }
					  start = end;
				  }
				  //tracker_ptr->VisualizeResults(iteration);
				  //tracker_ptr->UpdateViewers(iteration);
				  //cv::waitKey(0);		
				  if (break_flag)
				  {
					  //break;
				  }
			  }
			  {
				  dposex = tracker_ptr->pose_current[0];
				  if (tracker_ptr->err_current[0] < errMin)
				  {
					  errMin = tracker_ptr->err_current[0];
					  /*最小误差时的位姿*/
					  dpose = dposex;
					  min_error_pose_.matrix() << dposex.R(0, 0), dposex.R(0, 1), dposex.R(0, 2), dposex.t(0),
						  dposex.R(1, 0), dposex.R(1, 1), dposex.R(1, 2), dposex.t(1),
						  dposex.R(2, 0), dposex.R(2, 1), dposex.R(2, 2), dposex.t(2);
				  }
				  //满足提出条件
				  //if (errMin < errT)
				 //cout << "当前误差：" << errMin << endl;
				  if (errMin < errT)
				  {
					  break;
				  }
			  }
		  }
	  }
	  //std::cout << "iter_num: "<<iter_num << std::endl;
	  std::shared_ptr<srt3d::RegionModality> region_modality_ptr = tracker_ptr->region_modality_ptrs_[0];
	  const auto &body_ptr{ region_modality_ptr->body_ptr() };
	  body_ptr->set_body2world_pose(min_error_pose_);

	  //cout << "local opti over====" << endl;
	 /*绕z轴旋转--通过调整上向量的方式*/
	  /*有时有正向的影响，有时是负向的*/
#if 1
	  if (errMin > errT)
	  {
		  const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;   //12
		  //cout<< N <<endl;
		  //划分角度
		  const float dc = (thetaT * 1.0) / N;
		  //子划分
		  const int subDiv = 3;
		  const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
		  //轨迹？
		  //cout<< subRegionSize <<endl;
		  srt3d::RegionTrajectory_ traj_(cv::Size(subRegionSize, 1), dc / subDiv);

		  cv::Mat1b label = cv::Mat1b::zeros(2 * N + 1, 1);  //3 x 3
		  
		  struct DSeed
		  {
			  cv::Point coord;
			  //bool  isLocalMinima;
		  };
		  std::deque<DSeed>  seeds;
		  seeds.push_back({ cv::Point(N,0)/*,true*/ });
		  label(N, 0) = 1;
		  auto checkAdd = [&seeds, &label](const DSeed& curSeed, int dx) {
			  int x = curSeed.coord.x + dx;
			  if (uint(x) < uint(label.rows))
			  {
				  if (label(x, 0) == 0)
				  {
					  label(x, 0) = 1;
					  seeds.push_back({ cv::Point(x, 0)/*, false*/ });
				  }
			  }
		  };

		  //std::vector<float> z_rot_vector;
		  //z_rot_vector.push_back(30);  //2.5
		  //z_rot_vector.push_back(-30);

		  const auto R0 = tracker_ptr->R_[0];

		  while (!seeds.empty())
		  {		  
			  auto curSeed = seeds.front();
			  seeds.pop_front();
			  checkAdd(curSeed, -1);
			  checkAdd(curSeed, 1);	  

			  //auto dz = z_rot_vector.front();
			  //z_rot_vector.erase(z_rot_vector.begin());

			  float gamma = float(curSeed.coord.x - N) * dc;
			  //计算外平面旋转
			  auto dR = srt3d::dir2InofplaneRotation(gamma);

			  auto dpose = tracker_ptr->pose_current[0];
			  auto dposex = dpose;  //dpose
			  dposex.R = dR * R0;
			  /*计算开始的点*/
			  
			  //float start = gamma;
			  cv::Matx33f R_dr = dposex.R * R0.t();
			  //cout << R_dr << endl;
			  cv::Vec3f rr = cv::normalize(cv::Vec3f(R_dr(0, 0), R_dr(0, 1), R_dr(0, 2)));
			  float start_z = -asin(rr(1));
			  
			  //cout<< gamma <<endl;
			  //cout<< start_z <<endl;
			  int outerItrs = 1;
			  int innerItrs = 1;
			  for (int itr = 0; itr < outerItrs * innerItrs; ++itr)
			  {
				  //获取当前最近的视点，为了获取响应视点下的数据，从而可以进行位姿迭代计算
					  //得到当前视点的投影点数据，需要得到边缘和区域的两个数据

				  srt3d::Transform3fA body2camera_pose_;
				  body2camera_pose_.matrix() << dposex.R(0, 0), dposex.R(0, 1), dposex.R(0, 2), dposex.t(0),
					  dposex.R(1, 0), dposex.R(1, 1), dposex.R(1, 2), dposex.t(1),
					  dposex.R(2, 0), dposex.R(2, 1), dposex.R(2, 2), dposex.t(2);
				  //重新斟酌
				  std::shared_ptr<srt3d::RegionModality> region_modality_ptr = tracker_ptr->region_modality_ptrs_[0];
				  const auto &body_ptr{ region_modality_ptr->body_ptr() };
				  body_ptr->set_body2world_pose(body2camera_pose_);

				  int break_flag = 0;
				  for (int corr_iteration = 3;      //迭代四次？    -2
					  corr_iteration < tracker_ptr->n_corr_iterations() - 1; ++corr_iteration) {
					  int corr_save_idx =
						  iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
					  tracker_ptr->StartOcclusionRendering();
					  begin_time = std::chrono::high_resolution_clock::now();
					  tracker_ptr->CalculateCorrespondences(corr_iteration);
					  execution_times->calculate_correspondences += ElapsedTime(begin_time);
					  tracker_ptr->VisualizeCorrespondences(corr_save_idx);
					  for (int update_iteration = 0;
						  update_iteration < tracker_ptr->n_update_iterations();// - 1
						  ++update_iteration) {
						  int update_save_idx =
							  corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
						  begin_time = std::chrono::high_resolution_clock::now();
						  tracker_ptr->CalculatePoseUpdate(corr_iteration, update_iteration);
						  execution_times->calculate_pose_update += ElapsedTime(begin_time);
						  tracker_ptr->VisualizePoseUpdate(update_save_idx);
					  }		
					  /*原来的更新判断*/
					  float eps = 1e-4f;
					  if (tracker_ptr->diff_pose[0] < eps * eps)
					  {
						  break_flag = 1;
						  break;
					  }
					  dposex = tracker_ptr->pose_current[0];

					  cv::Matx33f  diff_r = dposex.R * R0.t();
					  //cout << R_dr << endl;
					  cv::Vec3f diff_norm_r = cv::normalize(cv::Vec3f(diff_r(0, 0), diff_r(0, 1), diff_r(0, 2)));
					  float end_z = -asin(diff_norm_r(1));
					  /*计算两者的距离*/
					  //cout << traj_._pathMask.cols << endl;
					  if (traj_.addStep(cv::Point2f(start_z, 0), cv::Point2f(end_z, 0)))
					  {
						  //cout<<"traj_.addStep"<<endl;
						  break_flag = 1;
						  break;
					  }
					  start_z = end_z;
				  }
				  if (break_flag)
				  {
					  //break;
				  }
				  //tracker_ptr->VisualizeResults(iteration);
				  //tracker_ptr->UpdateViewers(iteration);
				  //cv::waitKey(0);				 
			  }
			  {
				  dposex = tracker_ptr->pose_current[0];
				  if (tracker_ptr->err_current[0] < errMin)
				  {
					  errMin = tracker_ptr->err_current[0];
					  /*最小误差时的位姿*/
					  dpose = dposex;
					  min_error_pose_.matrix() << dposex.R(0, 0), dposex.R(0, 1), dposex.R(0, 2), dposex.t(0),
						  dposex.R(1, 0), dposex.R(1, 1), dposex.R(1, 2), dposex.t(1),
						  dposex.R(2, 0), dposex.R(2, 1), dposex.R(2, 2), dposex.t(2);
				  }
				  if (errMin < errT)
				  {
					  break;
				  }
			  }
		  }
	  }
#endif

	 /*绕z轴旋转*/
#if 0
	  if (errMin > errT)
	  {
		  std::vector<float> z_rot_vector;
		  z_rot_vector.push_back(30);  //2.5
		  z_rot_vector.push_back(-30);
		  while (!z_rot_vector.empty())
		  {
			  auto dz = z_rot_vector.front();
			  z_rot_vector.erase(z_rot_vector.begin());

			  auto dpose = tracker_ptr->pose_current[0];
			  auto dposex_t = dpose;  //dpose

			  int outerItrs = 1;
			  int innerItrs = 3;
			  for (int itr = 0; itr < outerItrs * innerItrs; ++itr)
			  {
				  //获取当前最近的视点，为了获取响应视点下的数据，从而可以进行位姿迭代计算
					  //得到当前视点的投影点数据，需要得到边缘和区域的两个数据
				  srt3d::Transform3fA body2camera_pose_;
				  body2camera_pose_.matrix() << dposex_t.R(0, 0), dposex_t.R(0, 1), dposex_t.R(0, 2), dposex_t.t(0),
					  dposex_t.R(1, 0), dposex_t.R(1, 1), dposex_t.R(1, 2), dposex_t.t(1),
					  dposex_t.R(2, 0), dposex_t.R(2, 1), dposex_t.R(2, 2), dposex_t.t(2);

				  Eigen::Matrix4f result_pose_ = body2camera_pose_.matrix();
				  float x_ = result_pose_(0, 3);
				  float y_ = result_pose_(1, 3);
				  float z_ = result_pose_(2, 3);

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
				  gamma_result_ = gamma_result_ + dz;

				  cv::Matx44f body1_translationMatrix = cv::Matx44f(1, 0, 0, x_,
					  0, 1, 0, y_,
					  0, 0, 1, z_,
					  0, 0, 0, 1);

				  cv::Matx44f T_cm = body1_translationMatrix * rotationMatrix(alph_result_, cv::Vec3f(1, 0, 0)) * rotationMatrix(beta_result_, cv::Vec3f(0, 1, 0))*rotationMatrix(gamma_result_, cv::Vec3f(0, 0, 1))
					  *cv::Matx44f::eye();
				  body2camera_pose_.matrix() << T_cm(0, 0), T_cm(0, 1), T_cm(0, 2), T_cm(0, 3),
					  T_cm(1, 0), T_cm(1, 1), T_cm(1, 2), T_cm(1, 3),
					  T_cm(2, 0), T_cm(2, 1), T_cm(2, 2), T_cm(2, 3),
					  T_cm(3, 0), T_cm(3, 1), T_cm(3, 2), T_cm(3, 3);
				  body2camera_pose_ = body2camera_pose_;

				  //重新斟酌
				  std::shared_ptr<srt3d::RegionModality> region_modality_ptr = tracker_ptr->region_modality_ptrs_[0];
				  const auto &body_ptr{ region_modality_ptr->body_ptr() };
				  body_ptr->set_body2world_pose(body2camera_pose_);

				  for (int corr_iteration = 3;      //迭代四次？
					  corr_iteration < tracker_ptr->n_corr_iterations() - 2; ++corr_iteration) {
					  int corr_save_idx =
						  iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
					  tracker_ptr->StartOcclusionRendering();
					  begin_time = std::chrono::high_resolution_clock::now();
					  tracker_ptr->CalculateCorrespondences(corr_iteration);
					  execution_times->calculate_correspondences += ElapsedTime(begin_time);
					  tracker_ptr->VisualizeCorrespondences(corr_save_idx);
					  for (int update_iteration = 0;
						  update_iteration < tracker_ptr->n_update_iterations();// - 1
						  ++update_iteration) {
						  int update_save_idx =
							  corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
						  begin_time = std::chrono::high_resolution_clock::now();
						  //tracker_ptr->CalculatePoseUpdate(corr_iteration, update_iteration);
						  //execution_times->calculate_pose_update += ElapsedTime(begin_time);
						  //tracker_ptr->VisualizePoseUpdate(update_save_idx);
					  }
				  }
				  tracker_ptr->VisualizeResults(iteration);
				  tracker_ptr->UpdateViewers(iteration);
				  //cv::waitKey(0);
				  /*原来的更新判断*/
				  float eps = 1e-4f;
				  if (tracker_ptr->diff_pose[0] < eps * eps)
				  {
					  break;
				  }
				  dposex_t = tracker_ptr->pose_current[0];

			  }
			  {
				  dposex_t = tracker_ptr->pose_current[0];

				  if (tracker_ptr->err_current[0] < errMin)
				  {
					  errMin = tracker_ptr->err_current[0];
					  /*最小误差时的位姿*/
					  dpose = dposex_t;
					  min_error_pose_.matrix() << dposex_t.R(0, 0), dposex_t.R(0, 1), dposex_t.R(0, 2), dposex_t.t(0),
						  dposex_t.R(1, 0), dposex_t.R(1, 1), dposex_t.R(1, 2), dposex_t.t(1),
						  dposex_t.R(2, 0), dposex_t.R(2, 1), dposex_t.R(2, 2), dposex_t.t(2);
				  }
				  if (errMin < errT)
				  {
					  break;
				  }
			  }
		  }
	  }
#endif
	  /*基于平移扰动的非局部优化*/
#if 0
	   /*仍然没有小于设定的误差值，则进行平移扰动*/
	  std::vector<cv::Vec3f> dt_vector;
	  dt_vector.push_back(cv::Vec3f(0, 0.003, 0));  //0.002
	  dt_vector.push_back(cv::Vec3f(0, -0.003, 0));
	  dt_vector.push_back(cv::Vec3f(0.003, 0, 0));
	  dt_vector.push_back(cv::Vec3f(-0.003, 0, 0));
	  dt_vector.push_back(cv::Vec3f(0, 0.0, 0.003));
	  dt_vector.push_back(cv::Vec3f(0, 0.0, -0.003));

	  //dt_vector.push_back(cv::Vec3f(0, 0.005, 0));  //0.002
	  //dt_vector.push_back(cv::Vec3f(0, -0.005, 0));
	  //dt_vector.push_back(cv::Vec3f(0.005, 0, 0));
	  //dt_vector.push_back(cv::Vec3f(-0.005, 0, 0));
	  //dt_vector.push_back(cv::Vec3f(0, 0.0, 0.005));
	  //dt_vector.push_back(cv::Vec3f(0, 0.0, -0.005));

	  if (errMin > errT)
	  {
		  while (!dt_vector.empty())
		  {
			  auto dt = dt_vector.front();
			  //cout << dt << endl;
			  dt_vector.erase(dt_vector.begin());

			  auto dpose = tracker_ptr->pose_current[0];
			  auto dposex_t = dpose;  //dpose
			  dposex_t.t = dposex_t.t + dt;

			  int outerItrs = 1;
			  int innerItrs = 4;
			  for (int itr = 0; itr < outerItrs * innerItrs; ++itr)
			  {
				  //获取当前最近的视点，为了获取响应视点下的数据，从而可以进行位姿迭代计算
					  //得到当前视点的投影点数据，需要得到边缘和区域的两个数据
				  srt3d::Transform3fA body2camera_pose_;
				  body2camera_pose_.matrix() << dposex_t.R(0, 0), dposex_t.R(0, 1), dposex_t.R(0, 2), dposex_t.t(0),
					  dposex_t.R(1, 0), dposex_t.R(1, 1), dposex_t.R(1, 2), dposex_t.t(1),
					  dposex_t.R(2, 0), dposex_t.R(2, 1), dposex_t.R(2, 2), dposex_t.t(2);
				  //重新斟酌
				  std::shared_ptr<srt3d::RegionModality> region_modality_ptr = tracker_ptr->region_modality_ptrs_[0];
				  const auto &body_ptr{ region_modality_ptr->body_ptr() };
				  body_ptr->set_body2world_pose(body2camera_pose_);

				  for (int corr_iteration = 3;      //迭代四次？
					  corr_iteration < tracker_ptr->n_corr_iterations() - 2; ++corr_iteration) {
					  int corr_save_idx =
						  iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
					  tracker_ptr->StartOcclusionRendering();
					  begin_time = std::chrono::high_resolution_clock::now();
					  tracker_ptr->CalculateCorrespondences(corr_iteration);
					  execution_times->calculate_correspondences += ElapsedTime(begin_time);
					  tracker_ptr->VisualizeCorrespondences(corr_save_idx);
					  int flag_break = 0;
					  int flag_break_2 = 0;
					  for (int update_iteration = 0;
						  update_iteration < tracker_ptr->n_update_iterations();// - 1
						  ++update_iteration) {
						  int update_save_idx =
							  corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
						  begin_time = std::chrono::high_resolution_clock::now();
						  tracker_ptr->CalculatePoseUpdate(corr_iteration, update_iteration);
						  execution_times->calculate_pose_update += ElapsedTime(begin_time);
						  tracker_ptr->VisualizePoseUpdate(update_save_idx);
					  }
				  }

				  //tracker_ptr->VisualizeResults(iteration);
				  //tracker_ptr->UpdateViewers(iteration);
				  //cv::waitKey(0);
				  /*原来的更新判断*/
				  float eps = 1e-4f;
				  if (tracker_ptr->diff_pose[0] < eps * eps)
				  {
					  break;
				  }
				  dposex_t = tracker_ptr->pose_current[0];

			  }
			  {
				  dposex_t = tracker_ptr->pose_current[0];

				  if (tracker_ptr->err_current[0] < errMin)
				  {
					  errMin = tracker_ptr->err_current[0];
					  /*最小误差时的位姿*/
					  dpose = dposex_t;
					  min_error_pose_.matrix() << dposex_t.R(0, 0), dposex_t.R(0, 1), dposex_t.R(0, 2), dposex_t.t(0),
						  dposex_t.R(1, 0), dposex_t.R(1, 1), dposex_t.R(1, 2), dposex_t.t(1),
						  dposex_t.R(2, 0), dposex_t.R(2, 1), dposex_t.R(2, 2), dposex_t.t(2);
				  }
				  if (errMin < errT)
				  {
					  break;
				  }
			  }
		  }
	  }

#endif

	  body_ptr->set_body2world_pose(min_error_pose_);
	  /*cv::Mat path = traj_._pathMask;
	  path.convertTo(path, CV_8U, 255.0);
	  cv::namedWindow("path", 0);
	  cv::imshow("path", path);	  
	  cv::waitKey(1);*/
  }

#endif

  // Visualize results and update viewers
  tracker_ptr->VisualizeResults(iteration);
  if (visualize_all_results_) tracker_ptr->UpdateViewers(iteration);
  
  execution_times->complete_cycle =
      execution_times->calculate_before_camera_update +
      execution_times->start_occlusion_rendering +
      execution_times->calculate_correspondences +
      execution_times->calculate_pose_update;
}

void RBOTEvaluator::CalculatePoseResults(
    const srt3d::Transform3fA &body2world_pose,
    const srt3d::Transform3fA &body2world_pose_gt, DataResult *result) const {


	//cout<< "222"<<body2world_pose.matrix()<< endl;
  result->translation_error = (body2world_pose.translation().matrix() -
                               body2world_pose_gt.translation().matrix())
                                  .norm();
  result->rotation_error =
      acos(((body2world_pose.rotation().matrix().transpose() *
             body2world_pose_gt.rotation().matrix())
                .trace() -
            1.0f) /
           2.0f);
  if (result->translation_error > translation_error_threshold_ ||
      result->rotation_error > rotation_error_threshold_)
    result->tracking_success = 0.0f;
  else
    result->tracking_success = 1.0f;
}

void RBOTEvaluator::CalculateAverageResult(
    const std::vector<DataResult> &results, DataResult *average_result) {
  average_result->rotation_error = 0.0f;
  average_result->translation_error = 0.0f;
  average_result->tracking_success = 0.0f;
  average_result->execution_times.calculate_before_camera_update = 0.0;
  average_result->execution_times.calculate_correspondences = 0.0f;
  average_result->execution_times.calculate_pose_update = 0.0f;
  average_result->execution_times.complete_cycle = 0.0f;
  average_result->execution_times.start_occlusion_rendering = 0.0f;

  for (auto &result : results) {
    average_result->rotation_error += result.rotation_error;
    average_result->translation_error += result.translation_error;
    average_result->tracking_success += result.tracking_success;
    average_result->execution_times.calculate_before_camera_update +=
        result.execution_times.calculate_before_camera_update;
    average_result->execution_times.calculate_correspondences +=
        result.execution_times.calculate_correspondences;
    average_result->execution_times.calculate_pose_update +=
        result.execution_times.calculate_pose_update;
    average_result->execution_times.complete_cycle +=
        result.execution_times.complete_cycle;
    average_result->execution_times.start_occlusion_rendering +=
        result.execution_times.start_occlusion_rendering;
  }

  float n = float(results.size());
  average_result->frame_index = 0;
  average_result->rotation_error /= n;
  average_result->translation_error /= n;
  average_result->tracking_success /= n;
  average_result->execution_times.calculate_before_camera_update /= n;
  average_result->execution_times.calculate_correspondences /= n;
  average_result->execution_times.calculate_pose_update /= n;
  average_result->execution_times.complete_cycle /= n;
  average_result->execution_times.start_occlusion_rendering /= n;
}

void RBOTEvaluator::VisualizeFrameResult(const DataResult &result,
                                         const std::string &title) {
#if 0
  std::cout << title << ": "
            << "frame " << result.frame_index << ": "
            << "execution_time = " << result.execution_times.complete_cycle
            << " us "
            << "rotation_error = "
            << result.rotation_error * 180.0f / srt3d::kPi << ", "
            << "translation_error = " << result.translation_error << ", "
            << "tracking success = " << result.tracking_success << std::endl;
#endif
  if (result.tracking_success != 1.0f)
  {
	  std::cout << title << ": "
		  << "frame " << result.frame_index << ": "
		  << "execution_time = " << result.execution_times.complete_cycle
		  << " us "
		  << "rotation_error = "
		  << result.rotation_error * 180.0f / srt3d::kPi << ", "
		  << "translation_error = " << result.translation_error << ", "
		  << "tracking success = " << result.tracking_success << std::endl;
	  //cv::waitKey(0);
  }
  //std::cout << result.rotation_error * 180.0f / srt3d::kPi << std::endl;
  //std::cout << result.translation_error * 1000 << std::endl;
}

void RBOTEvaluator::VisualizeFinalResult(const DataResult &results,
                                         const std::string &title) {
  std::cout << std::string(80, '-') << std::endl;
  std::cout << title << ":" << std::endl;
  std::cout << "success rate = " << results.tracking_success << std::endl;
  std::cout << "execution times:" << std::endl;
  std::cout << "complete cycle = " << results.execution_times.complete_cycle
            << " us" << std::endl;
  std::cout << "calculate before camera update = "
            << results.execution_times.calculate_before_camera_update << " us"
            << std::endl;
  std::cout << "start occlusion rendering = "
            << results.execution_times.start_occlusion_rendering << " us"
            << std::endl;
  std::cout << "calculate correspondences = "
            << results.execution_times.calculate_correspondences << " us"
            << std::endl;
  std::cout << "calculate pose update = "
            << results.execution_times.calculate_pose_update << " us"
            << std::endl;
}

void RBOTEvaluator::SaveFinalResult(const DataResult &result,
                                    const std::string &title) const {
  std::ofstream ofs{save_directory_ / ("results_" + title + ".txt")};
  ofs << result.tracking_success << "," << result.execution_times.complete_cycle
      << "," << result.execution_times.calculate_before_camera_update << ","
      << result.execution_times.start_occlusion_rendering << ","
      << result.execution_times.calculate_correspondences << ","
      << result.execution_times.calculate_pose_update << std::endl;
  ofs.flush();
  ofs.close();
}

bool RBOTEvaluator::GenerateModels() {
  for (auto &body_name : body_names_) {
    if (!GenerateSingleModel(body_name, dataset_directory_ / body_name))
      return false;
  }
  if (!GenerateSingleModel("squirrel_small", dataset_directory_)) return false;
  return true;
}

bool RBOTEvaluator::GenerateSingleModel(
    const std::string &body_name, const std::experimental::filesystem::path &directory) {
  auto body_ptr{std::make_shared<srt3d::Body>(
      body_name, directory / (body_name + ".obj"), 0.001f, true, false, 0.3f,  //0.3   0.5 原为0.3
      srt3d::Transform3fA::Identity())};

  auto model_ptr{std::make_shared<srt3d::Model>("model", body_ptr, directory,
                                                body_name + "_model.bin", 0.8f,     //原为0.8
                                                4, 200, false, 2000)};
  if (flag_edge)
  {
	  auto model_ptr_edge{ std::make_shared<srt3d::EdgeModel>("edge_model", body_ptr, directory,
												body_name + "_edge_model.bin", 0.8f,   //原为0.8
												4, 300, false, 2000) };
	  model_setter_(model_ptr);
	  //model_setter_(model_ptr_edge);
	  return model_ptr->SetUp() && model_ptr_edge->SetUp();  //
  }
  if (flag_edge && flag_line)
  {
	  auto model_ptr_edge{ std::make_shared<srt3d::EdgeModel>("edge_model", body_ptr, directory,
												body_name + "_edge_model.bin", 0.8f,
												4, 300, false, 2000) };

	  auto model_ptr_line{ std::make_shared<srt3d::LineModel>("line_model", body_ptr, directory,
												body_name + "_line_model.bin", 0.8f,
												4, 300, false, 2000) };
	  model_setter_(model_ptr);
	  //model_setter_(model_ptr_edge);
	  return model_ptr->SetUp() && model_ptr_edge->SetUp() && model_ptr_line->SetUp();  //
  }

  if(!flag_edge && !flag_line)
	  return model_ptr->SetUp();
}

bool RBOTEvaluator::ReadPosesRBOTDataset(
    const std::experimental::filesystem::path &path,
    std::vector<srt3d::Transform3fA> *poses) {
  std::ifstream ifs{path.string(), std::ios::binary};
  if (!ifs.is_open() || ifs.fail()) {
    ifs.close();
    std::cerr << "Could not open file stream " << path.string() << std::endl;
    return false;
  }

  poses->resize(kNFrames_ + 1);
  std::string parsed;
  std::getline(ifs, parsed);
  for (auto &pose : *poses) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        std::getline(ifs, parsed, '\t');
        pose.matrix()(i, j) = stof(parsed);
      }
    }
    std::getline(ifs, parsed, '\t');
    pose.matrix()(0, 3) = stof(parsed)* 0.001f;  //* 0.001f
    std::getline(ifs, parsed, '\t');
    pose.matrix()(1, 3) = stof(parsed)* 0.001f;
    std::getline(ifs, parsed);
    pose.matrix()(2, 3) = stof(parsed)* 0.001f;  //1.5
  }
  return true;
}

float RBOTEvaluator::ElapsedTime(
    const std::chrono::high_resolution_clock::time_point &begin_time) {
  auto end_time{std::chrono::high_resolution_clock::now()};
  return float(std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     begin_time)
                   .count());
}

cv::Matx44f RBOTEvaluator::rotationMatrix(float angle, cv::Vec3f axis)
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
