// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef OBJECT_TRACKING_INCLUDE_SRT3D_TRACKER_H_
#define OBJECT_TRACKING_INCLUDE_SRT3D_TRACKER_H_

#include <srt3d/body.h>
#include <srt3d/camera.h>
#include <srt3d/common.h>
#include <srt3d/occlusion_renderer.h>
#include <srt3d/region_modality.h>
#include <srt3d/renderer_geometry.h>
#include <srt3d/viewer.h>
#include <srt3d/EdgeModality.h>
#include <srt3d/edge_model.h>

#include <srt3d/LineModality.h>
#include <srt3d/line_model.h>

#include <memory>
#include <vector>

namespace srt3d {

// Class that holds multiple region modalities and viewers for visualization. It
// coordinates all objects and referenced objects for continuous tracking. Also,
// it defines how many correspondence iterations and pose update iterations are
// executed and implements functionality that defines how long visualizations
// are shown.
class Tracker {
 public:
  // Constructor and setter methods
  Tracker(const std::string &name);
  ~Tracker();
  void AddRegionModality(std::shared_ptr<RegionModality> region_modality_ptr);

  void AddEdgeModality(std::shared_ptr<EdgeModality> edge_modality_ptr);
  void AddLineModality(std::shared_ptr<LineModality> line_modality_ptr);

  void AddViewer(std::shared_ptr<Viewer> viewer_ptr);
  void set_n_corr_iterations(int n_corr_iterations);
  void set_n_update_iterations(int n_update_iterations);
  void set_visualization_time(int visualization_time);
  void set_viewer_time(int viewer_time);

  // Main method
  bool SetUpTracker();
  bool StartTracker(bool start_tracking);

  // Methods for advanced use
  bool ExecuteViewingCycle(int iteration);
  bool ExecuteTrackingCycle(int iteration);

  // Individual steps of tracking cycle for advanced use
  bool StartRegionModalities();
  bool CalculateBeforeCameraUpdate();
  bool UpdateCameras();
  bool StartOcclusionRendering();
  bool CalculateCorrespondences(int corr_iteration);
  bool VisualizeCorrespondences(int save_idx);
  bool CalculatePoseUpdate(int corr_iteration, int update_iteration);
  bool VisualizePoseUpdate(int save_idx);
  bool VisualizeResults(int save_idx);
  bool UpdateViewers(int save_idx);
  bool CalculatePoseUpdate_Region(int corr_iteration, int update_iteration);
  // Getters
  const std::string &name() const;
  std::vector<std::shared_ptr<RegionModality>> region_modality_ptrs() const;
  std::vector<std::shared_ptr<EdgeModality>> edge_modality_ptrs() const;
  std::vector<std::shared_ptr<LineModality>> line_modality_ptrs() const;

  std::vector<std::shared_ptr<Viewer>> viewer_ptrs() const;
  int n_corr_iterations() const;
  int n_update_iterations() const;
  int visualization_time() const;
  int viewer_time() const;
  bool set_up() const;

 private:
  // Helper methods
  void AssambleDerivedObjectPtrs();
  bool SetUpAllObjects();

  // Objects
  std::vector<std::shared_ptr<RendererGeometry>> renderer_geometry_ptrs_;
  std::vector<std::shared_ptr<Camera>> camera_ptrs_;
  std::vector<std::shared_ptr<Viewer>> viewer_ptrs_;
  std::vector<std::shared_ptr<Model>> model_ptrs_;
  std::vector<std::shared_ptr<OcclusionRenderer>> occlusion_renderer_ptrs_;
public:
  std::vector<std::shared_ptr<RegionModality>> region_modality_ptrs_;

  //lnb添加
  std::vector<std::shared_ptr<EdgeModel>> edge_model_ptrs_;
  std::vector<std::shared_ptr<EdgeModality>> edge_modality_ptrs_;

  std::vector<std::shared_ptr<LineModel>> line_model_ptrs_;
  std::vector<std::shared_ptr<LineModality>> line_modality_ptrs_;

  void Tracker::CalculateEdgeAndLine();
  // Parameters
  std::string name_{};
  int n_corr_iterations_ = 5;    //7   4
  int n_update_iterations_ = 2;  //2
  int visualization_time_ = 0;
  int viewer_time_ = 1;

  // State variables
  bool start_tracking_ = false;
  bool tracking_started_ = false;
  bool set_up_ = false;

  std::vector<float> err_current;
  std::vector<cv::Matx33f> R_;

  struct Pose
  {
	  cv::Matx33f R;
	  cv::Vec3f   t;
  };
  //当前pose
  std::vector<Pose> pose_current;
  /*判断是否需要非局部优化*/
  int flag_use_no_local = 0;
  //位姿变动量
  std::vector<float> diff_pose;
  
  float lastFrame_error = 0;

  void CalTup(float k, float s, float a, float &b);
  void CalTdown(float k, float s, float a, float &b);

    //

};

template <typename T>
void AddPtrIfNameNotExists(T &&ptr, std::vector<T> *ptrs) {
  if (std::none_of(begin(*ptrs), end(*ptrs),
                   [&ptr](const T &p) { return p->name() == ptr->name(); })) {
    ptrs->push_back(std::move(ptr));
  }
}

template <typename T>
bool SetUpObjectPtrs(std::vector<T> *ptrs) {
  for (T &ptr : *ptrs) {
    if (!ptr->SetUp()) {
      std::cout << "Error setting up " << ptr->name() << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace srt3d

#endif  // OBJECT_TRACKING_INCLUDE_SRT3D_TRACKER_H_
