// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#include <srt3d/tracker.h>

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <srt3d/RegionTrajectory.h>


namespace srt3d {

Tracker::Tracker(const std::string &name) : name_{name} {
	
}
Tracker::~Tracker()
{
	//optimizedPoseFile_1.close();
}

void Tracker::AddRegionModality(
    std::shared_ptr<RegionModality> region_modality_ptr) {
  region_modality_ptrs_.push_back(std::move(region_modality_ptr));
}
//添加边缘模型
void Tracker::AddEdgeModality(
	std::shared_ptr<EdgeModality> edge_modality_ptr) {
	edge_modality_ptrs_.push_back(std::move(edge_modality_ptr));
}
//添加边缘模型
void Tracker::AddLineModality(
	std::shared_ptr<LineModality> line_modality_ptr) {
	line_modality_ptrs_.push_back(std::move(line_modality_ptr));
}



void Tracker::AddViewer(std::shared_ptr<Viewer> viewer_ptr) {
  viewer_ptrs_.push_back(std::move(viewer_ptr));
}

void Tracker::set_n_corr_iterations(int n_corr_iterations) {
  n_corr_iterations_ = n_corr_iterations;
}

void Tracker::set_n_update_iterations(int n_update_iterations) {
  n_update_iterations_ = n_update_iterations;
}

void Tracker::set_visualization_time(int visualization_time) {
  visualization_time_ = visualization_time;
}

void Tracker::set_viewer_time(int viewer_time) { viewer_time_ = viewer_time; }

bool Tracker::SetUpTracker() {
  AssambleDerivedObjectPtrs();
  if (!SetUpAllObjects()) return false;
  set_up_ = true;
  /*保存每一帧图像的位姿结果*/
  /*optimizedPoseFile_1.open("pose1.txt", ios::out | ios::in);
  if (!optimizedPoseFile_1.is_open()) {
	  printf("Cannot write optimized pose\n");
  }*/
  return true;
}

bool Tracker::StartTracker(bool start_tracking) {
  if (!set_up_) {
    std::cerr << "Set up tracker " << name_ << " first" << std::endl;
    return false;
  }

  start_tracking_ = start_tracking;
  //====iteration<2表示只循环一次就返回了=====
  auto begin_time{ std::chrono::high_resolution_clock::now() };

  for (int iteration = 0;iteration < 2; ++iteration) {
    if (start_tracking_) 
	{
      if (!StartRegionModalities()) return false;
      tracking_started_ = true;
      start_tracking_ = false;
    }
    if (tracking_started_) 
	{
      if (!ExecuteTrackingCycle(iteration)) break;
    } 
	else 
	{
      if (!ExecuteViewingCycle(iteration)) break;
    }
  }

  auto end_time{ std::chrono::high_resolution_clock::now() };
  float spend_time = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count());
  std::cout << "spend_time: " << spend_time / 1000 << std::endl;
  return true;
}



bool Tracker::ExecuteViewingCycle(int iteration) {
  if (!UpdateCameras()) return false;

  //是否启动跟踪
  return UpdateViewers(iteration);
}

bool Tracker::ExecuteTrackingCycle(int iteration) {
	//直方图更新
  if (!CalculateBeforeCameraUpdate()) return false;
  /*下一帧图像*/
  if (!UpdateCameras()) return false;
  //预计算图像的边缘和直线
  //CalculateEdgeAndLine();

  for (int corr_iteration = 0; corr_iteration < n_corr_iterations_;
       ++corr_iteration) {
	  int corr_save_idx = iteration * n_corr_iterations_ + corr_iteration;
	  if (!StartOcclusionRendering()) return false;

	  if (!CalculateCorrespondences(corr_iteration)) return false;
	  if (!VisualizeCorrespondences(corr_save_idx)) return false;

	  for (int update_iteration = 0; update_iteration < n_update_iterations_;
		  ++update_iteration) {
		  int update_save_idx =
			  corr_save_idx * n_update_iterations_ + update_iteration;
		  if (!CalculatePoseUpdate(corr_iteration, update_iteration)) return false;

		  if (!VisualizePoseUpdate(update_save_idx)) return false;
	  }
#if 0
	  if (corr_iteration == 0)
	  {
		  for (auto &region_modality_ptr : region_modality_ptrs_) {
			  if (!region_modality_ptr->CalculateCorrespondences(corr_iteration))
				  return false;
		  }

		  bool imshow_correspondences = false;
		  for (auto &region_modality_ptr : region_modality_ptrs_) {
			  if (!region_modality_ptr->VisualizeCorrespondences(corr_save_idx)) return false;
			  if (region_modality_ptr->imshow_correspondence())
				  imshow_correspondences = true;
		  }

		  for (int update_iteration = 0; update_iteration < n_update_iterations_;
			  ++update_iteration) {
			  int update_save_idx =
				  corr_save_idx * n_update_iterations_ + update_iteration;
			  if (!CalculatePoseUpdate_Region(corr_iteration, update_iteration)) return false;
			  if (!VisualizePoseUpdate(update_save_idx)) return false;
		  }
	  }
	  else
	  {

		  if (!CalculateCorrespondences(corr_iteration-1)) return false;
		  if (!VisualizeCorrespondences(corr_save_idx-1)) return false;

		  for (int update_iteration = 0; update_iteration < n_update_iterations_;
			  ++update_iteration) {
			  int update_save_idx =
				  corr_save_idx * n_update_iterations_ + update_iteration;
			  if (!CalculatePoseUpdate(corr_iteration-1, update_iteration)) return false;
			  if (!VisualizePoseUpdate(update_save_idx)) return false;
		  }
	  }  
#endif
  }
  

  cv::Matx44f pose;
  /*for (auto &region_modality_ptr : region_modality_ptrs_) {
	  const auto &body_ptr{ region_modality_ptr->body_ptr() };
	  cv::eigen2cv(body_ptr->body2world_pose().matrix(), pose);
  } */
  /*第一个物体*/
  /*std::shared_ptr<RegionModality> region_modality_ptr = region_modality_ptrs_[0];
  const auto &body_ptr{ region_modality_ptr->body_ptr() };
  cv::eigen2cv(body_ptr->body2world_pose().matrix(), pose);

  std::ofstream optimizedPoseFile_1("pose1_11-8.txt", ios::app);
  optimizedPoseFile_1 << pose(0, 0) << "\t" << pose(0, 1) << "\t" << pose(0, 2) << "\t" <<
	  pose(1, 0) << "\t" << pose(1, 1) << "\t" << pose(1, 2) << "\t" <<
	  pose(2, 0) << "\t" << pose(2, 1) << "\t" << pose(2, 2) << "\t" <<
	  pose(0, 3) << "\t" << pose(1, 3) << "\t" << pose(2, 3) << "\t" << "\n";*/

#if 0
  //经过上述的优化，仍然误差很大
  //进行非局部优化搜索--视点搜索
  srt3d::Transform3fA min_error_pose_;
  auto dpose = pose_current[0];
  min_error_pose_.matrix() << dpose.R(0, 0), dpose.R(0, 1), dpose.R(0, 2), dpose.t(0),
	  dpose.R(1, 0), dpose.R(1, 1), dpose.R(1, 2), dpose.t(1),
	  dpose.R(2, 0), dpose.R(2, 1), dpose.R(2, 2), dpose.t(2);

  float errT = 1.5f;   //2.0
  float thetaT = CV_PI / 30;   //30
  float errMin = err_current[0];
  //cout<< errMin <<endl;
  if (errMin > errT)
  {
	  //获取当前位姿的R矩阵
	  //cout << "local opti start====" << endl;
	  const auto R0 = R_[0];
	  const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;   //12
	  //cout<< N <<endl;
	  //划分角度
	  const float dc = thetaT / N;
	  //子划分
	  const int subDiv = 3;
	  const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
	  //轨迹？
	  //cout<< subRegionSize <<endl;
	  RegionTrajectory_ traj_(cv::Size(subRegionSize, subRegionSize), dc / subDiv);

	  cv::Mat1b label = cv::Mat1b::zeros(2 * N + 1, 2 * N + 1);
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
		  checkAdd(curSeed, 0, -1);
		  checkAdd(curSeed, -1, 0);
		  checkAdd(curSeed, 1, 0);
		  checkAdd(curSeed, 0, 1);
		  //计算外平面旋转
		  auto dR = srt3d::theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);

		  /*auto dposex = pose_current[0];
		  dposex.R = dR * R0;*/
		  auto dposex = dpose;  //dpose
		  dposex.R = dR * R0;

		  cv::Point2f start = srt3d::dir2Theta(srt3d::viewDirFromR(dposex.R * R0.t()));
		  int outerItrs = 1;
		  int innerItrs = 1;
		  for (int itr = 0; itr < outerItrs*innerItrs; ++itr)
		  {
			  //获取当前最近的视点，为了获取响应视点下的数据，从而可以进行位姿迭代计算
				  //得到当前视点的投影点数据，需要得到边缘和区域的两个数据
			  srt3d::Transform3fA body2camera_pose_;
			  body2camera_pose_.matrix() << dposex.R(0, 0), dposex.R(0, 1), dposex.R(0, 2), dposex.t(0),
				  dposex.R(1, 0), dposex.R(1, 1), dposex.R(1, 2), dposex.t(1),
				  dposex.R(2, 0), dposex.R(2, 1), dposex.R(2, 2), dposex.t(2);
			  //重新斟酌
			  for (auto &region_modality_ptr : region_modality_ptrs_)
			  {
				  const auto &body_ptr{ region_modality_ptr->body_ptr() };
				  body_ptr->set_body2world_pose(body2camera_pose_);
			  }
			  /*for (auto &edge_modality_ptr : edge_modality_ptrs_)
			  {
				  const auto &body_ptr{ edge_modality_ptr->body_ptr() };
				  body_ptr->set_body2world_pose(body2camera_pose_);
			  }*/

			  //计算一次看位姿变动量
			  //tracker_ptr->CalculateBeforeCameraUpdate();
			  //tracker_ptr->UpdateCameras();

			  for (int corr_iteration = 3;
				  corr_iteration < n_corr_iterations(); ++corr_iteration) {

				  int corr_save_idx =
					  iteration * n_corr_iterations() + corr_iteration;
				  StartOcclusionRendering();
				  CalculateCorrespondences(corr_iteration);
				  VisualizeCorrespondences(corr_save_idx);
				  for (int update_iteration = 0;
					  update_iteration < n_update_iterations();
					  ++update_iteration) {
					  int update_save_idx =
						  corr_save_idx * n_update_iterations() + update_iteration;
					  CalculatePoseUpdate(corr_iteration, update_iteration);
					  VisualizePoseUpdate(update_save_idx);
				  }
			  }

			  //VisualizeResults(iteration);
			  //UpdateViewers(iteration);
			  //cv::waitKey(0);
			  //如果位姿变动量足够小，则跳出循环
			  //代码实现=====
			  float eps = 1e-4f;
			  if (diff_pose[0] < eps * eps)
			  {
				  //cout << "diff_pose" << endl;
				  //break;
			  }
			  ///============
			  dposex = pose_current[0];
			  cv::Point2f end = srt3d::dir2Theta(srt3d::viewDirFromR(dposex.R * R0.t()));
			  //没有运动
			  if (traj_.addStep(start, end))
			  {
				  //cout<<"traj_.addStep"<<endl;
				  //break;
			  }
			  start = end;
		  }
		  {
			  dposex = pose_current[0];
			  if (err_current[0] < errMin)
			  {
				  errMin = err_current[0];
				  dpose = dposex;
				  min_error_pose_.matrix() << dposex.R(0, 0), dposex.R(0, 1), dposex.R(0, 2), dposex.t(0),
					  dposex.R(1, 0), dposex.R(1, 1), dposex.R(1, 2), dposex.t(1),
					  dposex.R(2, 0), dposex.R(2, 1), dposex.R(2, 2), dposex.t(2);
			  }
			  //满足提出条件
			  //if (errMin < errT)
			 //cout << "当前误差：" << errMin << endl;
			  if (errMin < errT)
				  break;
		  }
	  }
	  std::shared_ptr<srt3d::RegionModality> region_modality_ptr = region_modality_ptrs_[0];
	  const auto &body_ptr{ region_modality_ptr->body_ptr() };
	  body_ptr->set_body2world_pose(min_error_pose_);
	  //cout << "local opti over====" << endl;
  }

#endif

#if 0
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

	  const auto R0 = R_[0];

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

		  auto dpose = pose_current[0];
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
			  std::shared_ptr<srt3d::RegionModality> region_modality_ptr = region_modality_ptrs_[0];
			  const auto &body_ptr{ region_modality_ptr->body_ptr() };
			  body_ptr->set_body2world_pose(body2camera_pose_);

			  int break_flag = 0;
			  for (int corr_iteration = 3;      //迭代四次？    -2
				  corr_iteration < n_corr_iterations() - 1; ++corr_iteration) {
				  int corr_save_idx =
					  iteration * n_corr_iterations() + corr_iteration;
				  StartOcclusionRendering();				  
				  CalculateCorrespondences(corr_iteration);			  
				  VisualizeCorrespondences(corr_save_idx);
				  for (int update_iteration = 0;
					  update_iteration < n_update_iterations();// - 1
					  ++update_iteration) {
					  int update_save_idx =
						  corr_save_idx * n_update_iterations() + update_iteration;					  
					 CalculatePoseUpdate(corr_iteration, update_iteration);
                     VisualizePoseUpdate(update_save_idx);
				  }
				  /*原来的更新判断*/
				  float eps = 1e-4f;
				  if (diff_pose[0] < eps * eps)
				  {
					  break_flag = 1;
					  break;
				  }
				  dposex = pose_current[0];

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
			  dposex = pose_current[0];
			  if (err_current[0] < errMin)
			  {
				  errMin = err_current[0];
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
	  std::shared_ptr<srt3d::RegionModality> region_modality_ptr = region_modality_ptrs_[0];
	  const auto &body_ptr{ region_modality_ptr->body_ptr() };
	  body_ptr->set_body2world_pose(min_error_pose_);
  }

#endif
  /*if (!VisualizeResults(iteration)) return false;
  if (!UpdateViewers(iteration)) return false;*/
  if (!VisualizeResults(iteration)) return false;
  if (!UpdateViewers(iteration)) return false;
  return true;
}
void Tracker::CalculateEdgeAndLine()
{
	for (auto &edge_modality_ptr : edge_modality_ptrs_) {
		edge_modality_ptr->PrecalculateExtractEdge();
	}
	for (auto &line_modality_ptr : line_modality_ptrs_) {
		line_modality_ptr->PrecalculateExtractEdge();
	}
}
bool Tracker::StartRegionModalities() {

  for (auto &region_modality_ptr : region_modality_ptrs_) {
    if (!region_modality_ptr->StartModality()) return false;
  }
  //cout << "xxxxx"<<endl;
	for (auto &edge_modality_ptr : edge_modality_ptrs_) {
		if (!edge_modality_ptr->StartModality()) return false;
	}
	//cout << "11111" << endl;
	for (auto &line_modality_ptr : line_modality_ptrs_) {
		if (!line_modality_ptr->StartModality()) return false;
	}

  return true;
}

bool Tracker::CalculateBeforeCameraUpdate() {
  for (auto &region_modality_ptr : region_modality_ptrs_) {
    if (!region_modality_ptr->CalculateBeforeCameraUpdate()) return false;
  }
	for (auto &edge_modality_ptr : edge_modality_ptrs_) {
		if (!edge_modality_ptr->CalculateBeforeCameraUpdate()) return false;
	}
	for (auto &line_modality_ptr : line_modality_ptrs_) {
		if (!line_modality_ptr->CalculateBeforeCameraUpdate()) return false;
	}
  return true;
}

bool Tracker::UpdateCameras() {
  for (auto &camera_ptr : camera_ptrs_) {
    if (!camera_ptr->UpdateImage()) return false;
  }
  return true;
}

bool Tracker::StartOcclusionRendering() {
  for (auto &occlusion_renderer_ptr : occlusion_renderer_ptrs_) {
    if (!occlusion_renderer_ptr->StartRendering()) return false;
  }
  return true;
}

bool Tracker::CalculateCorrespondences(int corr_iteration) {

		for (auto &region_modality_ptr : region_modality_ptrs_) {
			if (!region_modality_ptr->CalculateCorrespondences(corr_iteration))
				return false;
		}
		for (auto &line_modality_ptr : line_modality_ptrs_) {
			if (!line_modality_ptr->CalculateCorrespondences(corr_iteration))
				return false;
		}
		for (auto &edge_modality_ptr : edge_modality_ptrs_) {
			if (!edge_modality_ptr->CalculateCorrespondences(corr_iteration))
				return false;
		}
		
		/*std::shared_ptr<EdgeModality> edge_modality_ptr = edge_modality_ptrs_[0];
		std::shared_ptr<RegionModality> region_modality_ptr = region_modality_ptrs_[0];
		auto sovle1 = std::bind(&EdgeModality::CalculateCorrespondences, edge_modality_ptr,std::ref(corr_iteration));
		auto sovle2 = std::bind(&RegionModality::CalculateCorrespondences, region_modality_ptr, std::ref(corr_iteration));
		thread t1(sovle1);
		thread t2(sovle2);
		t1.join();
		t2.join();*/

  return true;
}

bool Tracker::VisualizeCorrespondences(int save_idx) {
  bool imshow_correspondences = false;
  for (auto &region_modality_ptr : region_modality_ptrs_) {
    if (!region_modality_ptr->VisualizeCorrespondences(save_idx)) return false;
    if (region_modality_ptr->imshow_correspondence())
      imshow_correspondences = true;
  }

  for (auto &edge_modality_ptr : edge_modality_ptrs_) {
	  if (!edge_modality_ptr->VisualizeCorrespondences(save_idx)) return false;
	  if (edge_modality_ptr->imshow_correspondence())
		  imshow_correspondences = true;
  }

  for (auto &line_modality_ptr : line_modality_ptrs_) {
	  if (!line_modality_ptr->VisualizeCorrespondences(save_idx)) return false;
	  if (line_modality_ptr->imshow_correspondence())
		  imshow_correspondences = true;
  }

  if (imshow_correspondences) {
    if (cv::waitKey(visualization_time_) == 'q') return false;
  }
  return true;
}
bool Tracker::CalculatePoseUpdate_Region(int corr_iteration, int update_iteration)
{
	return true;
}
bool Tracker::CalculatePoseUpdate(int corr_iteration, int update_iteration) {

	/*for (auto &region_modality_ptr : region_modality_ptrs_) {
	if (!region_modality_ptr->CalculatePoseUpdate(corr_iteration,
													update_iteration))
		return false;
	}
	for (auto &edge_modality_ptr : edge_modality_ptrs_) {
		if (!edge_modality_ptr->CalculatePoseUpdate(corr_iteration,
			update_iteration))
			return false;
	}
	for (auto &line_modality_ptr : line_modality_ptrs_) {
		if (!line_modality_ptr->CalculatePoseUpdate(corr_iteration,
			update_iteration))
			return false;
	}*/

#if 0
	std::shared_ptr<RegionModality>  region_modality_ptr = region_modality_ptrs_[0];
	std::shared_ptr<EdgeModality>  edge_modality_ptr = edge_modality_ptrs_[0];
	if (!region_modality_ptr->CalculatePoseUpdate(corr_iteration,
		update_iteration))
		return false;
	if (!edge_modality_ptr->CalculatePoseUpdate(corr_iteration,
		update_iteration))
		return false;

	Eigen::Matrix<float, 6, 6> a{ Eigen::Matrix<float, 6, 6>::Zero() };
	Eigen::Matrix<float, 6, 1> b{ Eigen::Matrix<float, 6, 1>::Zero() };

	//融合的方式 会造成 位姿的波动
	//缩小100倍的效果比较好，但优化过程变慢
	//误差如何计算
	err_current = 0;
	float error_count_sum = 0;
	float cost_shape = 0;
	float cost_match_ratio = 0;

	float edge_match_ratio = 0;
	float region_match_ratio = 0;
	
	error_count_sum += region_modality_ptr->error_count_;
	cost_shape += region_modality_ptr->shape_cost_;
	cost_match_ratio += region_modality_ptr->match_ratio_;
	region_match_ratio = region_modality_ptr->match_ratio_;

	//for (auto &region_modality_ptr : region_modality_ptrs_) {
	//	error_count_sum += region_modality_ptr->error_count_;
	//	cost_shape += region_modality_ptr->shape_cost_;
	//	cost_match_ratio += region_modality_ptr->match_ratio_;
	//	region_match_ratio = region_modality_ptr->match_ratio_;
	//	//cout<< region_modality_ptr->error_count_ <<endl;
	//}
	
	error_count_sum += edge_modality_ptr->error_count_;
	cost_shape += edge_modality_ptr->shape_cost_;
	cost_match_ratio += edge_modality_ptr->match_ratio_;
	edge_match_ratio = edge_modality_ptr->match_ratio_;

	//for (auto &edge_modality_ptr : edge_modality_ptrs_) {
	//	error_count_sum += edge_modality_ptr->error_count_;
	//	cost_shape += edge_modality_ptr->shape_cost_;
	//	cost_match_ratio += edge_modality_ptr->match_ratio_;
	//	edge_match_ratio = edge_modality_ptr->match_ratio_;
	//	//cout << edge_modality_ptr->scale_ << endl;
	//	//cout << edge_modality_ptr->error_count_ << endl;
	//}	

	for (auto &line_modality_ptr : line_modality_ptrs_) {
		//error_count_sum += line_modality_ptr->error_count_;
		//cost_shape += line_modality_ptr->shape_cost_;
		//cout << edge_modality_ptr->scale_ << endl;
		//cout << edge_modality_ptr->error_count_ << endl;
	}
	//像素误差
	//error_count_sum = error_count_sum / 2 ;
	if (error_count_sum != 0)
	{
		err_current += error_count_sum;
		//cout <<"all error: "<< err_current << endl;
		//cout << "all diff_pose: " << diff_pose * 1000 << endl;
		//cout << "all cost_shape: " << cost_shape << endl;
	}	
	//权重如何分配
	//大位移时：1.0   0.1
	//小位移：1.0  1.0
	//如何分配逐像素权重？？
	//基于模糊数学理论计算

	//一级权重设计===a1 轮廓残差  a2 位姿变动量 a3 形状相似度
	float a1 = 0.5;
	float a2 = 0.0;
	float a3 = 0.0;
	float a4 = 0.5;
	//计算隶属度函数
	float contours_eror = 0;
	float diffpose_eror = 0;
	float shape_eror = 0;
	float match_error = 0;
	//cout<<"误差输出"<<endl;
#if 0 
	//cout<< err_current <<endl;
	//cout << diff_pose * 1000<< endl;
	//cout << cost_shape << endl;
	//cout << cost_match_ratio << endl;
	/****位姿变动量权重计算****/
	//==位姿变动非常小时，region权重比较大===
	CalTup(5.0, 0.6, diff_pose * 1000, diffpose_eror);
	
	/****形状相似度的计算******/
	//==形状越相似，region权重越小，edge越大====
	CalTdown(5.0, 0.02, cost_shape, shape_eror);
	/***匹配的残差**/
	//==残差越大，region权重越大==
	CalTdown(5.0, 1.0, err_current, contours_eror);
	/*****匹配率**********/
	//==-匹配率越大，edge越大====
	CalTdown(0.1, 1.0, cost_match_ratio, match_error);

	/**计算region的权重**/
	Eigen::Vector4f region_w(a1, a2, a3, a4);
	Eigen::Vector4f region_cal{diffpose_eror , 1 - shape_eror ,1 - contours_eror ,match_error };
	
	float w_region = region_w.dot(region_cal);
	float w_edge = 1.4 - w_region;
#endif
	float w_edge_match = 0;
	float w_region_match = 0;
	//第一个参数为斜率
	
	CalTup(2.5, 0.2, edge_match_ratio, w_edge_match);    //1.5
	CalTup(6.0, 0.5, region_match_ratio, w_region_match);  //3.0

	//cout << w_edge_match << endl;
	//cout<< w_region_match <<endl;
	//cout<< corr_iteration <<endl;
	vector<float> w_region_{1.0, 1.0, 1.0, 1.0, 1.0};
	vector<float> w_edge_{ 0.4, 0.4, 0.4, 0.4, 0.4 };
	//cout<< "w_region_match: "<<w_region_match <<endl;
	//cout << "edge_match_ratio: " << edge_match_ratio << endl;
	
	/*分阶段进行并行和串行*/
	float s_region = 1.1;  //1.1
	float s_edge = 0.01;  //0.007   
	int even = corr_iteration % 2;
	//cout<< corr_iteration <<endl;
	/*采用以下策略也会存在问题*/
#if 1
	/*if (corr_iteration < 2)
	{
		s_region = 1.1;
		s_edge = 0;
	}
	if (corr_iteration >= 2 && corr_iteration < 4)
	{
		s_region = 1.1;
		s_edge = 0.005;
	}
	if (corr_iteration >= 4 && corr_iteration <= 5)
	{
		s_region = 0.0;
		s_edge = 0.005;
	}
	if (corr_iteration > 5)
	{
		s_region = 1.1;
		s_edge = 0.005;
	}*/

#endif
	float weight_region = s_region * w_region_match;// s_region * w_region_match;// s_region * w_region_match;// s_region * w_region_match;  //1.1 * w_edge_match
	//float weight_edge = 0.8 * w_edge;
	float weight_edge = s_edge * w_edge_match;  //s_edge * w_edge_match 0.6 * w_edge_match  0.4  0.040   0.07 * w_edge_match
	//cout << weight_edge << endl;

	float weight_line = 0.0;  //w_line

	//float weight_region = 1.0;
	//float weight_edge = 1.0;
	//cout << "weight_edge: " << weight_edge << endl;
	//cout << "weight_region: " << weight_region << endl;
	//float weight_edge = 0.05;
	//float weight_region = 0.01;
	a -= weight_region * region_modality_ptr->hessian;
	b += weight_region * region_modality_ptr->gradient;
	a -= weight_edge * edge_modality_ptr->hessian_edge;
	b += weight_edge * edge_modality_ptr->gradient_edge;

	//for (auto &region_modality_ptr : region_modality_ptrs_) {
	//	a -= weight_region * region_modality_ptr->hessian;
	//	b += weight_region * region_modality_ptr->gradient;	
	//	//cout << " region_modality_ptr->hessian: "<< region_modality_ptr->hessian << endl;
	//	//cout << " region_modality_ptr->gradient: " << region_modality_ptr->gradient << endl;
	//}
	//for (auto &edge_modality_ptr : edge_modality_ptrs_) {
	//	a -= weight_edge * edge_modality_ptr->hessian_edge;
	//	b += weight_edge * edge_modality_ptr->gradient_edge;
	//	//cout << " region_modality_ptr->hessian: "<< edge_modality_ptr->hessian_edge << endl;
	//	//cout << " region_modality_ptr->gradient: " << edge_modality_ptr->gradient_edge << endl;
	//}

	/*for (auto &line_modality_ptr : line_modality_ptrs_) {
		a -= weight_line * line_modality_ptr->hessian_edge;
		b += weight_line * line_modality_ptr->gradient_edge;
	}*/

	a.diagonal().head<3>().array() += 5000.0f;
	a.diagonal().tail<3>().array() += 500000.0f;  //500000.0f

	// Optimize and update pose
#if 1
	Eigen::FullPivLU<Eigen::Matrix<float, 6, 6>> lu{ a };
	if (lu.isInvertible()) {
		Eigen::Matrix<float, 6, 1> theta{ lu.solve(b) };
		//cout<< "theta"<<theta <<endl;
		Transform3fA pose_variation{ Transform3fA::Identity() };
		pose_variation.rotate(Vector2Skewsymmetric(theta.head<3>()).exp());
		pose_variation.translate(theta.tail<3>());

		//for (auto &region_modality_ptr : region_modality_ptrs_)
		//{
			const auto &body_ptr{ region_modality_ptr->body_ptr() };
			//cout << "FullPivLU" << (body_ptr->body2world_pose()).matrix() << endl;
			body_ptr->set_body2world_pose(body_ptr->body2world_pose() * pose_variation);
			//cout << "aa" << body_ptr->geometry2body_pose_.matrix() << endl;

			cv::Matx44f pose;
			cv::eigen2cv((body_ptr->body2world_pose() * pose_variation).matrix(), pose);
			R_ = cv::Matx33f(pose(0, 0), pose(0, 1), pose(0, 2),
				pose(1, 0), pose(1, 1), pose(1, 2), 
				pose(2, 0), pose(2, 1), pose(2, 2));
			cv::Vec3f t_(pose(0, 3), pose(1, 3), pose(2, 3));
			pose_current.R = R_;
			pose_current.t = t_;
			diff_pose = theta.dot(theta);
		//}

		//下面这句话会改变谁的量  相当于变动了两次，所以会出现错误

		/*for (auto &edge_modality_ptr : edge_modality_ptrs_)
		{
			const auto &body_ptr{ edge_modality_ptr->body_ptr() };
			body_ptr->set_body2world_pose(body_ptr->body2world_pose() * pose_variation);
		}*/
		//for (auto &line_modality_ptr : line_modality_ptrs_)
		//{
		//	const auto &body_ptr{ line_modality_ptr->body_ptr() };
		//	body_ptr->set_body2world_pose(body_ptr->body2world_pose() * pose_variation);
		//	//cout << body_ptr->body2world_pose().inverse().matrix() << endl;
		//}
	}
	else
		cout << "lu.isInvertible()"<< endl;
#endif
#endif

#if 0
	std::shared_ptr<RegionModality>  region_modality_ptr_occ = region_modality_ptrs_[1];
	std::shared_ptr<EdgeModality>  edge_modality_ptr_occ = edge_modality_ptrs_[1];
	if (!region_modality_ptr_occ->CalculatePoseUpdate(corr_iteration,
		update_iteration))
		return false;
	if (!edge_modality_ptr_occ->CalculatePoseUpdate(corr_iteration,
		update_iteration))
		return false;

	Eigen::Matrix<float, 6, 6> a_occ{ Eigen::Matrix<float, 6, 6>::Zero() };
	Eigen::Matrix<float, 6, 1> b_occ{ Eigen::Matrix<float, 6, 1>::Zero() };

	
	float error_count_sum_occ = 0;
	float cost_shape_occ = 0;
	float cost_match_ratio_occ = 0;

	float edge_match_ratio_occ = 0;
	float region_match_ratio_occ = 0;

	error_count_sum_occ += region_modality_ptr_occ->error_count_;
	cost_shape_occ += region_modality_ptr_occ->shape_cost_;
	cost_match_ratio_occ += region_modality_ptr_occ->match_ratio_;
	region_match_ratio_occ = region_modality_ptr_occ->match_ratio_;

	//for (auto &region_modality_ptr : region_modality_ptrs_) {
	//	error_count_sum += region_modality_ptr->error_count_;
	//	cost_shape += region_modality_ptr->shape_cost_;
	//	cost_match_ratio += region_modality_ptr->match_ratio_;
	//	region_match_ratio = region_modality_ptr->match_ratio_;
	//	//cout<< region_modality_ptr->error_count_ <<endl;
	//}

	error_count_sum_occ += edge_modality_ptr_occ->error_count_;
	cost_shape_occ += edge_modality_ptr_occ->shape_cost_;
	cost_match_ratio_occ += edge_modality_ptr_occ->match_ratio_;
	edge_match_ratio_occ = edge_modality_ptr_occ->match_ratio_;

	//for (auto &edge_modality_ptr : edge_modality_ptrs_) {
	//	error_count_sum += edge_modality_ptr->error_count_;
	//	cost_shape += edge_modality_ptr->shape_cost_;
	//	cost_match_ratio += edge_modality_ptr->match_ratio_;
	//	edge_match_ratio = edge_modality_ptr->match_ratio_;
	//	//cout << edge_modality_ptr->scale_ << endl;
	//	//cout << edge_modality_ptr->error_count_ << endl;
	//}	

	for (auto &line_modality_ptr : line_modality_ptrs_) {
		//error_count_sum += line_modality_ptr->error_count_;
		//cost_shape += line_modality_ptr->shape_cost_;
		//cout << edge_modality_ptr->scale_ << endl;
		//cout << edge_modality_ptr->error_count_ << endl;
	}
	//像素误差
	//error_count_sum = error_count_sum / 2 ;
	if (error_count_sum_occ != 0)
	{
		//err_current += error_count_sum_occ;
		//cout <<"all error: "<< err_current << endl;
		//cout << "all diff_pose: " << diff_pose * 1000 << endl;
		//cout << "all cost_shape: " << cost_shape << endl;
	}
	//权重如何分配
	//大位移时：1.0   0.1
	//小位移：1.0  1.0
	//如何分配逐像素权重？？
	//基于模糊数学理论计算

	//一级权重设计===a1 轮廓残差  a2 位姿变动量 a3 形状相似度
	float a1_occ = 0.5;
	float a2_occ = 0.0;
	float a3_occ = 0.0;
	float a4_occ = 0.5;
	//计算隶属度函数
	float contours_eror_occ = 0;
	float diffpose_eror_occ = 0;
	float shape_eror_occ = 0;
	float match_error_occ = 0;
	//cout<<"误差输出"<<endl;

	float w_edge_match_occ = 0;
	float w_region_match_occ = 0;
	//第一个参数为斜率

	CalTup(2.5, 0.2, edge_match_ratio_occ, w_edge_match_occ);    //1.5
	CalTup(6.0, 0.5, region_match_ratio_occ, w_region_match_occ);  //3.0

	//cout << w_edge_match << endl;
	//cout<< w_region_match <<endl;
	//cout<< corr_iteration <<endl;
	vector<float> w_region_occ{ 1.0, 1.0, 1.0, 1.0, 1.0 };
	vector<float> w_edge_occ{ 0.4, 0.4, 0.4, 0.4, 0.4 };
	//cout<< "w_region_match: "<<w_region_match <<endl;
	//cout << "edge_match_ratio: " << edge_match_ratio << endl;

	/*分阶段进行并行和串行*/
	float s_region_occ = 1.1;  //1.1
	float s_edge_occ = 0.01;  //0.007   

	float weight_region_occ = s_region_occ * w_region_match_occ;// s_region * w_region_match;// s_region * w_region_match;// s_region * w_region_match;  //1.1 * w_edge_match
	//float weight_edge = 0.8 * w_edge;
	float weight_edge_occ = s_edge_occ * w_edge_match_occ;  //s_edge * w_edge_match 0.6 * w_edge_match  0.4  0.040   0.07 * w_edge_match
	//cout << weight_edge << endl;

	float weight_line_occ = 0.0;  //w_line

	//float weight_region = 1.0;
	//float weight_edge = 1.0;
	//cout << "weight_edge: " << weight_edge << endl;
	//cout << "weight_region: " << weight_region << endl;
	//float weight_edge = 0.05;
	//float weight_region = 0.01;
	a_occ -= weight_region_occ * region_modality_ptr_occ->hessian;
	b_occ += weight_region_occ * region_modality_ptr_occ->gradient;
	a_occ -= weight_edge_occ * edge_modality_ptr_occ->hessian_edge;
	b_occ += weight_edge_occ * edge_modality_ptr_occ->gradient_edge;

	//for (auto &region_modality_ptr : region_modality_ptrs_) {
	//	a -= weight_region * region_modality_ptr->hessian;
	//	b += weight_region * region_modality_ptr->gradient;	
	//	//cout << " region_modality_ptr->hessian: "<< region_modality_ptr->hessian << endl;
	//	//cout << " region_modality_ptr->gradient: " << region_modality_ptr->gradient << endl;
	//}
	//for (auto &edge_modality_ptr : edge_modality_ptrs_) {
	//	a -= weight_edge * edge_modality_ptr->hessian_edge;
	//	b += weight_edge * edge_modality_ptr->gradient_edge;
	//	//cout << " region_modality_ptr->hessian: "<< edge_modality_ptr->hessian_edge << endl;
	//	//cout << " region_modality_ptr->gradient: " << edge_modality_ptr->gradient_edge << endl;
	//}

	/*for (auto &line_modality_ptr : line_modality_ptrs_) {
		a -= weight_line * line_modality_ptr->hessian_edge;
		b += weight_line * line_modality_ptr->gradient_edge;
	}*/

	a_occ.diagonal().head<3>().array() += 5000.0f;
	a_occ.diagonal().tail<3>().array() += 500000.0f;  //500000.0f

	Eigen::FullPivLU<Eigen::Matrix<float, 6, 6>> lu_occ{ a_occ };
	if (lu_occ.isInvertible()) {
		Eigen::Matrix<float, 6, 1> theta_occ{ lu_occ.solve(b_occ) };
		//cout<< "theta"<<theta <<endl;
		Transform3fA pose_variation_occ{ Transform3fA::Identity() };
		pose_variation_occ.rotate(Vector2Skewsymmetric(theta_occ.head<3>()).exp());
		pose_variation_occ.translate(theta_occ.tail<3>());

		//for (auto &region_modality_ptr : region_modality_ptrs_)
		//{
		const auto &body_ptr_occ{ region_modality_ptr_occ->body_ptr() };
		//cout << "FullPivLU" << (body_ptr->body2world_pose()).matrix() << endl;
		body_ptr_occ->set_body2world_pose(body_ptr_occ->body2world_pose() * pose_variation_occ);
		//cout << "aa" << body_ptr->geometry2body_pose_.matrix() << endl;

		/*cv::Matx44f pose_occ;
		cv::eigen2cv((body_ptr_occ->body2world_pose() * pose_variation_occ).matrix(), pose_occ);
		R_ = cv::Matx33f(pose_occ(0, 0), pose_occ(0, 1), pose_occ(0, 2),
			pose_occ(1, 0), pose_occ(1, 1), pose_occ(1, 2),
			pose_occ(2, 0), pose_occ(2, 1), pose_occ(2, 2));
		cv::Vec3f t_occ(pose_occ(0, 3), pose_occ(1, 3), pose_occ(2, 3));*/
		//pose_current.R = R_;
		//pose_current.t = t_occ;
		//diff_pose = theta_occ.dot(theta_occ);
		//}

		//下面这句话会改变谁的量  相当于变动了两次，所以会出现错误

		/*for (auto &edge_modality_ptr : edge_modality_ptrs_)
		{
			const auto &body_ptr{ edge_modality_ptr->body_ptr() };
			body_ptr->set_body2world_pose(body_ptr->body2world_pose() * pose_variation);
		}*/
		//for (auto &line_modality_ptr : line_modality_ptrs_)
		//{
		//	const auto &body_ptr{ line_modality_ptr->body_ptr() };
		//	body_ptr->set_body2world_pose(body_ptr->body2world_pose() * pose_variation);
		//	//cout << body_ptr->body2world_pose().inverse().matrix() << endl;
		//}
	}
	else
		cout << "lu.isInvertible()" << endl;
#endif

#if 1
	/*对上述过程进行重写*/
	err_current.resize(region_modality_ptrs_.size());
	R_.resize(region_modality_ptrs_.size());
	pose_current.resize(region_modality_ptrs_.size());
	diff_pose.resize(region_modality_ptrs_.size());

	std::fill(std::begin(err_current), std::end(err_current), 0.0f);
	if (region_modality_ptrs_.size() == edge_modality_ptrs_.size())
	{
		for (int index = 0; index < region_modality_ptrs_.size(); index++)
		{
			std::shared_ptr<RegionModality>  region_modality_ptr = region_modality_ptrs_[index];
			std::shared_ptr<EdgeModality>  edge_modality_ptr = edge_modality_ptrs_[index];
			if (!region_modality_ptr->CalculatePoseUpdate(corr_iteration,
				update_iteration))
				return false;
			if (!edge_modality_ptr->CalculatePoseUpdate(corr_iteration,
				update_iteration))
				return false;

			Eigen::Matrix<float, 6, 6> a{ Eigen::Matrix<float, 6, 6>::Zero() };
			Eigen::Matrix<float, 6, 1> b{ Eigen::Matrix<float, 6, 1>::Zero() };

			//融合的方式 会造成 位姿的波动
			//缩小100倍的效果比较好，但优化过程变慢
			//误差如何计算
			err_current[index] = 0.0;
			float error_count_sum = 0;

			float edge_match_ratio = 0;
			float region_match_ratio = 0;
			error_count_sum += region_modality_ptr->error_count_;
			region_match_ratio = region_modality_ptr->match_ratio_;

			error_count_sum += edge_modality_ptr->error_count_;
			edge_match_ratio = edge_modality_ptr->match_ratio_;

			if (error_count_sum != 0)
			{
				err_current[index] += error_count_sum;
			}

			float w_edge_match = 0;
			float w_region_match = 0;
			//第一个参数为斜率
			//cout << edge_match_ratio << endl;
			
			//for (float i = 0;i < 1; i = i + 0.1)
			//{
			//	float out;
			//	//CalTup(0.8, 0.5, i, out);
			//	//cout<< exp(-(1 - i)) <<endl;
			//	//cout << 1 - exp(-(i - 0.0)) << endl;
			//	//cout << exp(-2.0 * (1 - i)) << endl;
			//}
			/*0.5*/
			/*当边缘方法匹配率很低时，仍然需要应用，而当匹配率很高时，斜率比区域法低。*/
			CalTup(2.5, 0.1, edge_match_ratio, w_edge_match);    //2.5   原来2.5   0.5			
			/*0.5*/
			/*当区域方法匹配率非常低的时候，仍然需要应用它，而当其匹配率很高时，说明很容易区分前背景，则将其一定程度的调整*/
			CalTup(6.0, 0.1, region_match_ratio, w_region_match);  //3.0   原来6.0
			
			/*k值越大，曲线越低*/
			/*float w_edge_match_1;
			float w_region_match_1;
			if (edge_match_ratio <= 0.5)
			{
				w_edge_match_1 = 0.5;
			}
			else
			{
				float k1 = 5.0;
				w_edge_match_1 = exp(-k1 * (1.0 - edge_match_ratio));
			}
			if (region_match_ratio <= 0.5)
			{
				w_region_match_1 = 0.5;
			}
			else
			{
				float k2 = 12.0;
				w_region_match_1 = exp(-k2 * (1.0 - region_match_ratio));
			}*/
			//cout << 1 - w_edge_match_1 << endl;

			//cout << w_edge_match << endl;
			//cout << w_region_match << endl;
			//将误差映射为比例函数
			float error_contrl_ratio = 0;
			//cout <<err_current[index] << endl;
		
			//cout << err_current[index] << endl;
			///修改这里
			CalTup(0.8, 0.1, err_current[index], error_contrl_ratio);  //0.5  0.7

			float error = err_current[index] / 60.0;
			float test_error;
			
			//if (corr_iteration == 0)	
				//cout << err_current[index] << endl;
			/*分阶段进行并行和串行*/
			float s_region = 1.0 *  error_contrl_ratio;  // 1.0
			
			float s_edge = 1.8 * (1 - error_contrl_ratio);  //0.8

			//cout<< s_region <<endl;
			/*采用以下策略也会存在问题*/

			float weight_region = s_region * w_region_match; //s_region * w_region_match
			float weight_edge = s_edge * w_edge_match;  //s_edge * w_edge_match
			//cout << "edge_match_ratio: "<< edge_match_ratio << endl;
			//cout << "region_match_ratio: " << region_match_ratio << endl;

			a -= weight_region * (region_modality_ptr->hessian / 200);// / 200
			b += weight_region * (region_modality_ptr->gradient / 200);/// 200
			a -= weight_edge * (edge_modality_ptr->hessian_edge / 300);/// 300
			b += weight_edge * (edge_modality_ptr->gradient_edge / 300);// / 300

			//a -= 3.5 * (region_modality_ptr->hessian / 200);// / 200
			//b += 3.5  * (region_modality_ptr->gradient / 200);/// 200
			//a -= 1.0 * (edge_modality_ptr->hessian_edge / 300);/// 300
			//b += 1.0  * (edge_modality_ptr->gradient_edge / 300);// / 300
			
			//a -= 0.2 * edge_modality_ptr->hessian_edge / 300;/// 300
			//b += 0.2 * edge_modality_ptr->gradient_edge / 300;// / 300

			/*动态的高斯牛顿正则化参数，越大，步长越小，收敛速度越慢，反之亦然*/
			float value = 100; //300  100
			//float value = 1.0;
			/*衰减因子*/
			/*if (corr_iteration <= 4)
			{
				value = 100 * (corr_iteration + 1);
			}
			else
			{
				value = 20 * (corr_iteration + 1);
			}*/			

			if (corr_iteration == 0)
			{
				value = 500;  //100
			}
			if (corr_iteration == 1)
			{
				value = 300; //200
			}
			if (corr_iteration == 2)
			{
				value = 200; //300
			}
			/*if (corr_iteration == 3)
			{
				value = 400;
			}*/
			//value = 0.15 * (7 - corr_iteration);
			//335
			//value = -300 * corr_iteration + 6000;
			//std::cout << value << endl;
			a.diagonal().head<3>().array() += 50000 / value;//5000   / 100  180   / value
			a.diagonal().tail<3>().array() += 5000000 / value;//500000  / 100 180   / value

			Eigen::FullPivLU<Eigen::Matrix<float, 6, 6>> lu{ a };
			if (lu.isInvertible()) {
				//Eigen::LLT<Eigen::MatrixXf> lltOfA(b);
				//if (float(lltOfA.info()) == float(Eigen::NumericalIssue))
				//{
				//	/*不正定时，阻尼系数与雅可比的大小成正比*/
				//	cout << "Possibly semi-positive definitie matrix! 半正定" << endl;
				//}

				Eigen::Matrix<float, 6, 1> theta{ lu.solve(b) };
				Transform3fA pose_variation{ Transform3fA::Identity() };
				pose_variation.rotate(Vector2Skewsymmetric(theta.head<3>()).exp());
				pose_variation.translate(theta.tail<3>());
				const auto &body_ptr{ region_modality_ptr->body_ptr() };
				body_ptr->set_body2world_pose(body_ptr->body2world_pose() * pose_variation);
				//cout << body_ptr->body2world_pose() * pose_variation.matrix() << endl;
				
				cv::Matx44f pose;
				cv::eigen2cv((body_ptr->body2world_pose() * pose_variation).matrix(), pose);
				R_[index] = cv::Matx33f(pose(0, 0), pose(0, 1), pose(0, 2),
					pose(1, 0), pose(1, 1), pose(1, 2),
					pose(2, 0), pose(2, 1), pose(2, 2));
				cv::Vec3f t_(pose(0, 3), pose(1, 3), pose(2, 3));
				pose_current[index].R = R_[index];
				pose_current[index].t = t_;
				diff_pose[index] = theta.dot(theta);
			}
			else
				cout << "lu.isInvertible()" << endl;
		}
	}
	/*只有区域的方法*/
	else
	{
		for (int index = 0; index < region_modality_ptrs_.size(); index++)
		{
			std::shared_ptr<RegionModality>  region_modality_ptr = region_modality_ptrs_[index];
			if (!region_modality_ptr->CalculatePoseUpdate(corr_iteration,
				update_iteration))
				return false;

			Eigen::Matrix<float, 6, 6> a{ Eigen::Matrix<float, 6, 6>::Zero() };
			Eigen::Matrix<float, 6, 1> b{ Eigen::Matrix<float, 6, 1>::Zero() };

			//融合的方式 会造成 位姿的波动
			//缩小100倍的效果比较好，但优化过程变慢
			//误差如何计算
			err_current[index] = 0;
			float error_count_sum = 0;
			error_count_sum += region_modality_ptr->error_count_;
			if (error_count_sum != 0)
			{
				err_current[index] += error_count_sum;
			}

			/*分阶段进行并行和串行*/
			float s_region = 1.0;  //1.1

			/*采用以下策略也会存在问题*/

			float weight_region = s_region;

			a -= weight_region * region_modality_ptr->hessian;
			b += weight_region * region_modality_ptr->gradient;

			a.diagonal().head<3>().array() += 5000.0f / 100;
			a.diagonal().tail<3>().array() += 500000.0f /100;

			Eigen::FullPivLU<Eigen::Matrix<float, 6, 6>> lu{ a };
			if (lu.isInvertible()) {
				Eigen::Matrix<float, 6, 1> theta{ lu.solve(b) };
				Transform3fA pose_variation{ Transform3fA::Identity() };
				pose_variation.rotate(Vector2Skewsymmetric(theta.head<3>()).exp());
				pose_variation.translate(theta.tail<3>());
				const auto &body_ptr{ region_modality_ptr->body_ptr() };
				body_ptr->set_body2world_pose(body_ptr->body2world_pose() * pose_variation);
				//cout << pose_variation.matrix() << endl;
				cv::Matx44f pose;
				cv::eigen2cv((body_ptr->body2world_pose() * pose_variation).matrix(), pose);
				R_[index] = cv::Matx33f(pose(0, 0), pose(0, 1), pose(0, 2),
					pose(1, 0), pose(1, 1), pose(1, 2),
					pose(2, 0), pose(2, 1), pose(2, 2));
				cv::Vec3f t_(pose(0, 3), pose(1, 3), pose(2, 3));
				pose_current[index].R = R_[index];
				pose_current[index].t = t_;
				diff_pose[index] = theta.dot(theta);
			}
			else
				cout << "lu.isInvertible()" << endl;
		}
	}
#endif	
  return true;
}
void Tracker::CalTup(float k, float s, float a, float &b)
{
	if (a < s)
	{
		b = 0.5;

	}
	else
	{
		b = 1 - exp(-k * (a - s));		
	}
		
	//std::cout << "CalTup ： " << b << std::endl;
}
void Tracker::CalTdown(float k, float s, float a, float &b)
{
	if (a < s)
	{
		b = 0.99;
	}
	else
		b = exp(-k * (a - s));
}

bool Tracker::VisualizePoseUpdate(int save_idx) {
  bool imshow_pose_update = false;
  for (auto &region_modality_ptr : region_modality_ptrs_) {
    if (!region_modality_ptr->VisualizePoseUpdate(save_idx)) return false;
    if (region_modality_ptr->imshow_pose_update()) imshow_pose_update = true;
  }

  for (auto &edge_modality_ptr : edge_modality_ptrs_) {
	  if (!edge_modality_ptr->VisualizePoseUpdate(save_idx)) return false;
	  if (edge_modality_ptr->imshow_pose_update()) imshow_pose_update = true;
  }
  for (auto &line_modality_ptr : line_modality_ptrs_) {
	  if (!line_modality_ptr->VisualizePoseUpdate(save_idx)) return false;
	  if (line_modality_ptr->imshow_pose_update()) imshow_pose_update = true;
  }

  if (imshow_pose_update) {
    if (cv::waitKey(visualization_time_) == 'q') return false;
  }
  return true;
}

bool Tracker::VisualizeResults(int save_idx) {
  bool imshow_result = false;
  for (auto &region_modality_ptr : region_modality_ptrs_) {
    if (!region_modality_ptr->VisualizeResults(save_idx)) return false;
    if (region_modality_ptr->imshow_result()) imshow_result = true;
  }

  for (auto &edge_modality_ptr : edge_modality_ptrs_) {
	  if (!edge_modality_ptr->VisualizeResults(save_idx)) return false;
	  if (edge_modality_ptr->imshow_result()) imshow_result = true;
  }
  for (auto &line_modality_ptr : line_modality_ptrs_) {
	  if (!line_modality_ptr->VisualizeResults(save_idx)) return false;
	  if (line_modality_ptr->imshow_result()) imshow_result = true;
  }

  if (imshow_result) {
    if (cv::waitKey(visualization_time_) == 'q') return false;
  }
  return true;
}

bool Tracker::UpdateViewers(int iteration) {
  if (!viewer_ptrs_.empty()) {
    for (auto &viewer_ptr : viewer_ptrs_) {
      viewer_ptr->UpdateViewer(iteration);
	  /*用于保存view图像*/
	  //viewer_ptr->StartSavingImages("E:\\Paper\\dataset\\my_dataset\\17-3\\viewResult", "jpg");
    }
    char key = cv::waitKey(viewer_time_);
	if (key == 's' && tracking_started_)
	{
		start_tracking_ = false;
		tracking_started_ = false;
	}

    if (key == 't' && !tracking_started_)
	//if (!tracking_started_)
      start_tracking_ = true;
    else if (key == 'q')
      return false;
  }
  return true;
}

std::vector<std::shared_ptr<RegionModality>> Tracker::region_modality_ptrs()
    const {
  return region_modality_ptrs_;
}

std::vector<std::shared_ptr<EdgeModality>> Tracker::edge_modality_ptrs()
const {
	return edge_modality_ptrs_;
}

std::vector<std::shared_ptr<LineModality>> Tracker::line_modality_ptrs()
const {
	return line_modality_ptrs_;
}

const std::string &Tracker::name() const { return name_; }

std::vector<std::shared_ptr<Viewer>> Tracker::viewer_ptrs() const {
  return viewer_ptrs_;
}

int Tracker::n_corr_iterations() const { return n_corr_iterations_; }

int Tracker::n_update_iterations() const { return n_update_iterations_; }

int Tracker::visualization_time() const { return visualization_time_; }

int Tracker::viewer_time() const { return viewer_time_; }

bool Tracker::set_up() const { return set_up_; }

void Tracker::AssambleDerivedObjectPtrs() {
  camera_ptrs_.clear();
  occlusion_renderer_ptrs_.clear();
  renderer_geometry_ptrs_.clear();
  //基于区域
  for (auto &region_modality_ptr : region_modality_ptrs_) {
    if (region_modality_ptr->camera_ptr())
      AddPtrIfNameNotExists(region_modality_ptr->camera_ptr(), &camera_ptrs_);
    if (region_modality_ptr->model_ptr())
      AddPtrIfNameNotExists(region_modality_ptr->model_ptr(), &model_ptrs_);
    if (region_modality_ptr->occlusion_renderer_ptr())
      AddPtrIfNameNotExists(region_modality_ptr->occlusion_renderer_ptr(),
                            &occlusion_renderer_ptrs_);
  }
  //基于边缘
  for (auto &edge_modality_ptr : edge_modality_ptrs_) {
	  if (edge_modality_ptr->camera_ptr())
		  AddPtrIfNameNotExists(edge_modality_ptr->camera_ptr(), &camera_ptrs_);
	  if (edge_modality_ptr->EdgeModel_ptr())
		  AddPtrIfNameNotExists(edge_modality_ptr->EdgeModel_ptr(), &edge_model_ptrs_);
	  if (edge_modality_ptr->occlusion_renderer_ptr())
		  AddPtrIfNameNotExists(edge_modality_ptr->occlusion_renderer_ptr(),
			  &occlusion_renderer_ptrs_);
  }

  //基于直线
  for (auto &line_modality_ptr : line_modality_ptrs_) {
	  if (line_modality_ptr->camera_ptr())
		  AddPtrIfNameNotExists(line_modality_ptr->camera_ptr(), &camera_ptrs_);
	  if (line_modality_ptr->LineModel_ptr())
		  AddPtrIfNameNotExists(line_modality_ptr->LineModel_ptr(), &line_model_ptrs_);
	  if (line_modality_ptr->occlusion_renderer_ptr())
		  AddPtrIfNameNotExists(line_modality_ptr->occlusion_renderer_ptr(),
			  &occlusion_renderer_ptrs_);
  }

  //遮挡渲染
  for (auto &occlusion_renderer_ptr : occlusion_renderer_ptrs_) {
    if (occlusion_renderer_ptr->renderer_geometry_ptr())
      AddPtrIfNameNotExists(occlusion_renderer_ptr->renderer_geometry_ptr(),
                            &renderer_geometry_ptrs_);
  }
  for (auto &viewer_ptr : viewer_ptrs_) {
    if (viewer_ptr->camera_ptr())
      AddPtrIfNameNotExists(viewer_ptr->camera_ptr(), &camera_ptrs_);
    if (viewer_ptr->renderer_geometry_ptr())
      AddPtrIfNameNotExists(viewer_ptr->renderer_geometry_ptr(),
                            &renderer_geometry_ptrs_);
  }
}

bool Tracker::SetUpAllObjects() {
  return SetUpObjectPtrs(&renderer_geometry_ptrs_) &&
         SetUpObjectPtrs(&camera_ptrs_) && SetUpObjectPtrs(&viewer_ptrs_) &&
         SetUpObjectPtrs(&model_ptrs_) &&
         SetUpObjectPtrs(&occlusion_renderer_ptrs_) &&
         SetUpObjectPtrs(&region_modality_ptrs_) && 
	  SetUpObjectPtrs(&edge_model_ptrs_)&&
	  SetUpObjectPtrs(&edge_modality_ptrs_) && 
	  SetUpObjectPtrs(&line_model_ptrs_) &&
	  SetUpObjectPtrs(&line_modality_ptrs_);
}

}  // namespace srt3d
