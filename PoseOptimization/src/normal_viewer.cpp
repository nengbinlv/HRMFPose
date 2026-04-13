// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#include <srt3d/normal_viewer.h>

namespace srt3d {

NormalViewer::NormalViewer(
    const std::string &name, std::shared_ptr<Camera> camera_ptr,
    std::shared_ptr<RendererGeometry> renderer_geometry_ptr, float opacity)
    : Viewer{name, camera_ptr},
      renderer_geometry_ptr_{renderer_geometry_ptr},
      opacity_{opacity},
      renderer_{"renderer", std::move(renderer_geometry_ptr),
                std::move(camera_ptr)} {}

bool NormalViewer::SetUp() {
  set_up_ = false;

  // Check if all required objects are set up
  if (!camera_ptr_->set_up()) {
    std::cerr << "Camera " << camera_ptr_->name() << " was not set up"
              << std::endl;
    return false;
  }
  if (!renderer_geometry_ptr_->set_up()) {
    std::cerr << "Renderer geometry " << renderer_geometry_ptr_->name()
              << " was not set up" << std::endl;
    return false;
  }

  if (!renderer_.SetUp()) return false;
  set_up_ = true;
  return true;
}

void NormalViewer::set_renderer_geometry_ptr(
    std::shared_ptr<RendererGeometry> renderer_geometry_ptr) {
  renderer_geometry_ptr_ = std::move(renderer_geometry_ptr);
  set_up_ = false;
}

void NormalViewer::set_opacity(float opacity) { opacity_ = opacity; }

bool NormalViewer::UpdateViewer(int save_index) {
  if (!set_up_) {
    std::cerr << "Set up viewer " << name_ << " first" << std::endl;
    return false;
  }

  // Calculate viewer image
  cv::Mat viewer_image{camera_ptr_->image().size(), CV_8UC3};
  renderer_.StartRendering();
  renderer_.FetchNormalImage();
  renderer_.FetchDepthImage();
  CalculateAlphaBlend(camera_ptr_->image(), renderer_.normal_image(),
                      &viewer_image);
  /*为了显示轮廓和边缘*/

#if 0
  cv::Mat color_img = camera_ptr_->image().clone();
  cv::Mat normal_img = renderer_.normal_image().clone();
  std::vector<cv::Mat> normal_image_channels(4);
  cv::split(normal_img, normal_image_channels);
  std::vector<std::vector<cv::Point2i>> contours;
  cv::findContours(normal_image_channels[3], contours, cv::RetrievalModes::RETR_LIST,
	  cv::ContourApproximationModes::CHAIN_APPROX_NONE);
  cv::Mat normal_canny;
  int low_thre = 20;
  int height_thre = 40;
  cv::Mat normal_image_temp;
  cv::cvtColor(normal_img, normal_image_temp, CV_BGRA2GRAY);
  //cv::imshow("normal_image_temp", normal_image_temp);
  //cv::waitKey(0);
  cv::Canny(normal_image_temp, normal_canny, low_thre, height_thre);
  std::vector<std::vector<cv::Point2i>> contours_edge;
  cv::findContours(normal_canny, contours_edge, cv::RetrievalModes::RETR_LIST,
	  cv::ContourApproximationModes::CHAIN_APPROX_NONE);
  contours_edge.erase(std::remove_if(begin(contours_edge), end(contours_edge),
	  [](const std::vector<cv::Point2i> &contour) {
	  return contour.size() < 15;
  }),
	  end(contours_edge));

  //提取角点
  //int maxCorners = 20;//检测角点数目
  //double quality_level = 0.1;//质量等级
  //double  minDistance = 0.4;//两个角点之间的最小欧式距离
  //vector<cv::Point2f> corners;
  //goodFeaturesToTrack(normal_image_temp, corners, maxCorners, quality_level, minDistance, cv::Mat(), 3, false);
  ////绘制角点
  //vector<cv::KeyPoint> keyPoints;//存放角点的KeyPoint类，用于后期绘制角点时使用
  //for (int i = 0; i < corners.size(); i++) {
	 // //将角点存放在KeyPoint类中
	 // cv::KeyPoint keyPoint;
	 // keyPoint.pt = corners[i];
	 // keyPoints.push_back(keyPoint);
  //}
  ////用drwaKeyPoints()函数绘制角点坐标
  //drawKeypoints(normal_image_temp, keyPoints, normal_image_temp);
  //cv::namedWindow("corner", 0);
  //cv::imshow("corner", normal_image_temp);
  //cv::waitKey(0);


  cv::Mat color_img2 = camera_ptr_->image().clone();

  cv::drawContours(color_img, contours_edge, -1, cv::Scalar(0, 255, 0), 1);
  //cv::imshow("color_img", color_img);
  //cv::waitKey(0);
  cv::namedWindow("color_img",0);
  cv::imshow("color_img", color_img);
  std::experimental::filesystem::path path{ "data/edge_12/" +
							   (name_ + "_image_" + std::to_string(save_index) +
								"." + "png") };
  cv::resize(color_img, color_img, cv::Size(color_img.cols * 2, color_img.rows * 2));
  //cv::imwrite(path.string(), color_img);

  cv::drawContours(color_img2, contours, -1, cv::Scalar(0, 0, 255), 1);
  cv::namedWindow("color_img2", 0);
  cv::imshow("color_img2", color_img2);

  std::experimental::filesystem::path path2{ "data/region/" +
							   (name_ + "_image_" + std::to_string(save_index) +
								"." + "png") };
  cv::resize(color_img2, color_img2, cv::Size(color_img2.cols * 2, color_img2.rows * 2));
  //cv::imwrite(path2.string(), color_img2);

  //cv::resize(color_img, color_img,cv::Size(color_img.cols * 2, color_img.rows * 1));
  //cv::imwrite("color_img.jpg", color_img);
  cv::waitKey(1);

#endif
  // Display and save images
  DisplayAndSaveImage(save_index, viewer_image);
}

std::shared_ptr<RendererGeometry> NormalViewer::renderer_geometry_ptr() const {
  return renderer_geometry_ptr_;
}

float NormalViewer::opacity() const { return opacity_; }

void NormalViewer::CalculateAlphaBlend(const cv::Mat &camera_image,
                                       const cv::Mat &renderer_image,
                                       cv::Mat *viewer_image) const {
  // Declare variables
  int v, u;
  const cv::Vec3b *ptr_camera_image;
  const cv::Vec4b *ptr_renderer_image;
  cv::Vec3b *ptr_viewer_image;
  const uchar *val_camera_image;
  const uchar *val_renderer_image;
  uchar *val_viewer_image;
  float alpha, alpha_inv;
  float alpha_scale = opacity_ / 255.0f;

  // Iterate over all pixels
  for (v = 0; v < camera_image.rows; ++v) {
    ptr_camera_image = camera_image.ptr<cv::Vec3b>(v);
    ptr_renderer_image = renderer_image.ptr<cv::Vec4b>(v);
    ptr_viewer_image = viewer_image->ptr<cv::Vec3b>(v);
    for (u = 0; u < camera_image.cols; ++u) {
      val_camera_image = ptr_camera_image[u].val;
      val_renderer_image = ptr_renderer_image[u].val;
      val_viewer_image = ptr_viewer_image[u].val;

      // Blend images
      alpha = float(val_renderer_image[3]) * alpha_scale;
      alpha_inv = 1.0f - alpha;
      val_viewer_image[0] =
          char(val_camera_image[0] * alpha_inv + val_renderer_image[0] * alpha);
      val_viewer_image[1] =
          char(val_camera_image[1] * alpha_inv + val_renderer_image[1] * alpha);
      val_viewer_image[2] =
          char(val_camera_image[2] * alpha_inv + val_renderer_image[2] * alpha);
    }
  }
}

}  // namespace srt3d
