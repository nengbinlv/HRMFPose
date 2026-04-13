#include <srt3d/body.h>
#include <srt3d/common.h>
#include <srt3d/normal_renderer.h>
#include <srt3d/teture_render.h>
#include <srt3d/renderer_geometry.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include <opencv2/ximgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/line_descriptor/descriptor.hpp>
namespace srt3d {
	class RegionTrajectory_
	{
	public:
		cv::Mat1b  _pathMask;
		float  _delta;

		cv::Point2f _uv2Pt(const cv::Point2f& uv)
		{
			return cv::Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
		}
	public:
		RegionTrajectory_(cv::Size regionSize, float delta)
		{
			_pathMask = cv::Mat1b::zeros(regionSize);
			_delta = delta;
		}
		bool  addStep(cv::Point2f start, cv::Point2f end)
		{
			start = _uv2Pt(start);
			end = _uv2Pt(end);

			auto dv = end - start;
			float len = sqrt(dv.dot(dv)) + 1e-6f;
			float dx = dv.x / len, dy = dv.y / len;
			const int  N = int(len) + 1;
			cv::Point2f p = start;
			for (int i = 0; i < N; ++i)
			{
				int x = int(p.x + 0.5), y = int(p.y + 0.5);
				if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
				{
					if (_pathMask(y, x) != 0 && len > 1.f)
						return true;
					_pathMask(y, x) = 1;
				}
				else
					return false;

				p.x += dx; p.y += dy;
			}

			return false;
		}

	};
}