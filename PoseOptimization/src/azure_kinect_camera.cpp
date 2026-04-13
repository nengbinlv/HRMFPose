// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)


#include <srt3d/azure_kinect_camera.h>

//从这个类中修改，从而对大恒相机的调用。
namespace srt3d {

AzureKinectCamera::AzureKinectCamera(const std::string &name, float image_scale)
    : Camera{name}, image_scale_{image_scale} {
	//temp_image = cv::imread("E:\\Paper\\dataset\\DenseTrackingDatasets\\DenseTrackingDatasets\\ATLAS\\scene\\referenceFrame1.png");
	timeStame = 0;
}

AzureKinectCamera::~AzureKinectCamera() {
  if (initial_set_up_) {
  }
}

bool AzureKinectCamera::SetUp() {
  set_up_ = false;
  if (!initial_set_up_)
    if (!StartAzureKinect()) return false;
  GetIntrinsicsAndDistortionMap();
  SaveMetaDataIfDesired();
  set_up_ = true;
  initial_set_up_ = true;
  return UpdateImage();
}

void AzureKinectCamera::set_image_scale(float image_scale) {
  image_scale_ = image_scale;
  set_up_ = false;
}

bool AzureKinectCamera::UpdateImage() {
  if (!set_up_) {
    std::cerr << "Set up camera " << name_ << " first" << std::endl;
    return false;
  }
  // Get image
  // Undistort image
  //cv::Mat temp_image;
  //capture_ >> temp_image;
  //隔帧读取

  /*capture_.set(cv::CAP_PROP_POS_FRAMES, timeStame);
  capture_ .read(temp_image);
  timeStame += 1;
  if (temp_image.empty())
  {
	  std::cerr << "temp_image.empty()" << std::endl;
	  return false;
  }*/
  //std::cout<<"frame:"<< timeStame << std::endl;

  //cv::imshow("temp_image", temp_image);
  //cv::waitKey(0);
  //如何更新图像
  //extern cv::Mat global_img_;
  //用copyto就不报错了
  //使用相机时解开
  //global_img_.copyTo(temp_image);
  
  cv::remap(temp_image, image_, distortion_map_, cv::Mat(), cv::INTER_NEAREST,cv::BORDER_CONSTANT);
  cv::remap(preEdge_image, image_edge_, distortion_map_, cv::Mat(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
  cv::Mat image_mask_temp;
  cv::remap(preMask_image, image_mask_temp, distortion_map_, cv::Mat(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
  if(image_mask_temp.channels() == 3)
	cv::cvtColor(image_mask_temp, image_mask_temp, CV_BGR2GRAY);
  cv::Mat struct1, struct2;
  struct1 = cv::getStructuringElement(0, cv::Size(5, 5));  //矩形结构元素
  struct2 = cv::getStructuringElement(1, cv::Size(5, 5));  //十字结构元素

  cv::Mat image_mask_temp_dilate;
  cv::dilate(image_mask_temp, image_mask_temp_dilate, struct1);
  image_mask_temp_dilate.convertTo(image_mask_dilate_, CV_32FC1, 1.0 / 255);
  
  //cv::imshow("image_mask_temp", image_mask_temp);
  //cv::waitKey(0);
  image_mask_temp.convertTo(image_mask_, CV_32FC1, 1.0 / 255);
  SaveImageIfDesired();
  return true;
}

float AzureKinectCamera::image_scale() const { return image_scale_; }

bool AzureKinectCamera::StartAzureKinect() {
  //capture_ = 0;
  //capture_.set(CAP_PROP_FRAME_WIDTH, 1920);   //CAP_PROP_FRAME_WIDTH
  //capture_.set(CAP_PROP_FRAME_HEIGHT, 1080);   //CAP_PROP_FRAME_HEIGHT

	//capture_.open("C:\\my_dataset\\videoData\\video11-8.mp4");

  return true;
}

void AzureKinectCamera::GetIntrinsicsAndDistortionMap() {
	//intrinsics_.fu = 2944.69865;  // 650.048
	//intrinsics_.fv = 2932.01052;  //647.183
	//intrinsics_.ppu = 1082.00100; //324.328
	//intrinsics_.ppv = 695.83815;  //257.323
	//intrinsics_.width = 1920;     //640
	//intrinsics_.height = 1080;    //512
  // Scale intrinsics acording to image scale
  intrinsics_.fu *= image_scale_;
  intrinsics_.fv *= image_scale_;

  // Calculate distortion map
  cv::Mat1f camera_matrix(3, 3);
  camera_matrix << intrinsics_.fu, 0, intrinsics_.ppu, 0, intrinsics_.fv,
	  intrinsics_.ppv, 0, 0, 1;
  cv::Mat1f new_camera_matrix(3, 3);
  new_camera_matrix << intrinsics_.fu, 0, intrinsics_.ppu, 0, intrinsics_.fv,
      intrinsics_.ppv, 0, 0, 1;

  //0.0866510402, 0.165377621, 0.00303108473, -0.000381850
  //cv::Mat1f distortion_coeff(1, 8);
  //distortion_coeff << 0.0, 0.0, 0, 0, 0.0,0, 0, 0;

  //cv::Mat1f distortion_coeff(1, 5);
  //distortion_coeff << -0.17874492, 0.12391957, 0.00006609, -0.00043841, -0.07488668;
  cv::Mat map1, map2, map3;
  cv::initUndistortRectifyMap(
      camera_matrix, distortion_coeff_, cv::Mat{}, new_camera_matrix,
      cv::Size{intrinsics_.width, intrinsics_.height}, CV_32FC1, map1, map2);
  cv::convertMaps(map1, map2, distortion_map_, map3, CV_16SC2, true);
}

}  // namespace srt3d
