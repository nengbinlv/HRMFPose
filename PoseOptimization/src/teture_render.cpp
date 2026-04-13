
#include <srt3d/teture_render.h>
//#include <srt3d/teture_render.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


namespace srt3d {

	std::string teture_render::vertex_shader_code_ =
	"#version 330 core\n"
	"layout(location = 0) in vec3 aPos;\n"
	"layout(location = 1) in vec3 aNormal;\n"
	"layout(location = 2) in vec2 aTexCoords;\n"
	"out vec2 TexCoords;\n"
	"uniform mat4 Trans;\n"
	"uniform mat3 Rot;\n"
	"uniform mat4 model;\n"
	"uniform Mat{\n"
	"vec4 aAmbient;\n"
	"vec4 aDiffuse;\n"
	"vec4 aSpecular;\n"
	"};\n"
	"out vec3 FragPos;\n"
	"out vec3 Normal;\n"
	"out vec4 Ambient;\n"
	"out vec4 Diffuse;\n"
	"out vec4 Specular;\n"
	"void main()\n"
	"{"
		"FragPos = vec3(model * vec4(aPos, 1.0));\n"
		//"Normal = mat3(transpose(inverse(model))) * aNormal;\n"
		"Normal = Rot * aNormal;\n"
		"Ambient = aAmbient;\n"
		"Diffuse = aDiffuse;\n"
		"Specular = aSpecular;\n"
		"TexCoords = aTexCoords;\n"
		"gl_Position = Trans * vec4(aPos, 1.0);\n"
	"}";

	std::string teture_render::fragment_shader_code_ =
		"#version 330 core\n"
		"out vec4 FragColor;\n"

	"in vec2 TexCoords;\n"

	"uniform sampler2D texture_diffuse1;\n"

	"struct Light {\n"
		"vec3 position;\n"

		"vec3 ambient;\n"
		"vec3 diffuse;\n"
		"vec3 specular;\n"

		"float constant;\n"
		"float linear;\n"
		"float quadratic;\n"
	"};"
	"in vec3 FragPos;\n"
	"in vec3 Normal;\n"

	"in vec4 Ambient;\n"
	"in vec4 Diffuse;\n"
	"in vec4 Specular;\n"

	"uniform vec3 viewPos;\n"
	"uniform Light light;\n"

	"uniform float shininess;\n"

	"void main()\n"
	"{\n"
		"vec3 ambient = light.ambient * Diffuse.rgb;\n"
		"vec3 norm = normalize(Normal);\n"
		"vec3 lightDir = normalize(light.position - FragPos);\n"
		"float diff = max(dot(norm, lightDir), 0.0);\n"
		"vec3 diffuse = light.diffuse * diff *Diffuse.rgb;\n"
		"float distance = length(light.position - FragPos);\n"
		"float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));\n"
		"vec3 viewDir = normalize(viewPos - FragPos);\n"
		"vec3 reflectDir = reflect(-lightDir, norm);\n"
		"float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);\n"
		"vec3 specular = light.specular * spec *   Specular.rgb;\n"
		"diffuse *= attenuation;\n"
		"specular *= attenuation;\n"
		"vec3 result = ambient + diffuse + specular;\n"
		//"FragColor = vec4(result, 1.0);\n"	
		"FragColor = vec4(0.5 - 0.5 * Normal, 1.0).zyxw;\n" 
	"}";

	
	teture_render::teture_render(
		const std::string &name,
		std::shared_ptr<RendererGeometry> renderer_geometry_ptr,
		const Transform3fA &world2camera_pose, const Intrinsics &intrinsics,
		float z_min, float z_max, float depth_scale)
		: Renderer{ name,
		std::move(renderer_geometry_ptr),
		world2camera_pose,
		intrinsics,
		z_min,
		z_max },
		depth_scale_{ depth_scale } {}

	teture_render::teture_render(
		const std::string &name,
		std::shared_ptr<RendererGeometry> renderer_geometry_ptr,
		std::shared_ptr<Camera> camera_ptr, float z_min, float z_max,
		float depth_scale)
		: Renderer{ name, std::move(renderer_geometry_ptr), std::move(camera_ptr),
		z_min, z_max },
		depth_scale_{ depth_scale } {}

	teture_render::~teture_render() {
		const std::lock_guard<std::mutex> lock{ mutex_ };
		if (initial_set_up_) DeleteBufferObjects();
	}

	bool teture_render::SetUp() {
		const std::lock_guard<std::mutex> lock{ mutex_ };
		set_up_ = false;
		image_rendered_ = false;

		// Check if all required objects are set up
		if (!renderer_geometry_ptr_->set_up()) {
			std::cout << "Renderer geometry " << renderer_geometry_ptr_->name()
				<< " was not set up" << std::endl;
			return false;
		}
		if (camera_ptr_) {
			if (!InitParametersFromCamera()) return false;
		}

		// Create shader programs
		if (!initial_set_up_) {

			//注释掉
			if (!CreateShaderProgram(vertex_shader_code_.c_str(),
				fragment_shader_code_.c_str(), &shader_program_))
				return false;
			//Our_Shader_.use();
		}

		// Set up everything
		CalculateProjectionMatrix();
		CalculateProjectionTerms();
		ClearImages();
		if (initial_set_up_) DeleteBufferObjects();
		CreateBufferObjects();

		initial_set_up_ = true;
		set_up_ = true;
		return true;
	}

	void teture_render::set_depth_scale(float depth_scale) {
		const std::lock_guard<std::mutex> lock{ mutex_ };
		depth_scale_ = depth_scale;
		set_up_ = false;
	}

	bool teture_render::StartRendering() {
		const std::lock_guard<std::mutex> lock{ mutex_ };
		if (!set_up_) {
			std::cerr << "Set up renderer " << name_ << " first" << std::endl;
			return false;
		}
		
		renderer_geometry_ptr_->MakeContextCurrent();
		glViewport(0, 0, intrinsics_.width, intrinsics_.height);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
		stbi_set_flip_vertically_on_load(true);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glFrontFace(GL_CCW);
		glCullFace(GL_FRONT);

		glUseProgram(shader_program_);
		glm::vec3 lampPos(-1.0f, -1.0f, -1.5f);

		GLint lightAmbientLoc = glGetUniformLocation(shader_program_, "light.ambient");
		GLint lightDiffuseLoc = glGetUniformLocation(shader_program_, "light.diffuse");
		GLint lightSpecularLoc = glGetUniformLocation(shader_program_, "light.specular");
		GLint lightPosLoc = glGetUniformLocation(shader_program_, "light.position");
		GLint attConstant = glGetUniformLocation(shader_program_, "light.constant");
		GLint attLinear = glGetUniformLocation(shader_program_, "light.linear");
		GLint attQuadratic = glGetUniformLocation(shader_program_, "light.quadratic");
		//增加会使光变得亮一些
		////RGBA模式的环境光，
		glUniform3f(lightAmbientLoc, 0.5f, 0.5f, 0.5f);
		////RGBA模式的漫反射光，全白光
		glUniform3f(lightDiffuseLoc, 0.5f, 0.5f, 0.5f);
		////RGBA模式下的镜面光 ，全白光
		glUniform3f(lightSpecularLoc, 0.5f, 0.5f, 0.5f);
		glUniform3f(lightPosLoc, lampPos.x, lampPos.y, lampPos.z);
		// 设置衰减系数
		glUniform1f(attConstant, 1.0f);
		glUniform1f(attLinear, 0.0f);   //0.09
		glUniform1f(attQuadratic, 0.0f);   //0.032
		// 设置观察者位置
		//渲染出现了问题，原因是viewpos没有设置
		GLint viewPosLoc = glGetUniformLocation(shader_program_, "viewPos");
		glUniform3f(viewPosLoc, 0, 0, 0);

		GLint shininess = glGetUniformLocation(shader_program_, "shininess");
		glUniform1f(shininess, 2.0f);

		for (const auto &render_data_body :
			renderer_geometry_ptr_->render_data_bodies()) {
			Transform3fA trans_without_projection{
				world2camera_pose_ * render_data_body.body_ptr->geometry2world_pose() };  //lnb修改  * render_data_body.body_ptr->T_n()
			Eigen::Matrix4f trans{ projection_matrix_ *
				trans_without_projection.matrix() };
			Eigen::Matrix3f rot{ trans_without_projection.rotation().matrix() };

			//std::cout << render_data_body.body_ptr->geometry2world_pose().matrix() << std::endl;
			//std::cout << trans_without_projection.matrix() << std::endl;
			//计算模型相对于世界的坐标系
			/*Transform3fA model_matrix_;
			model_matrix_.matrix() << 1,0, 0, 0,
				0, -1, 0, 0,
				0, 0, -1, 0,
				0, 0, 0, 1;*/
				/*Transform3fA model_matrix_camera2world{
				model_matrix_ * render_data_body.body_ptr->geometry2world_pose().inverse() };
				model_matrix_camera2world = model_matrix_camera2world.inverse();*/

			Eigen::Matrix4f model_matrix_shader{ render_data_body.body_ptr->geometry2world_pose().matrix() };
			//model_matrix_shader = model_matrix_shader.inverse();

			//std::cout << model_matrix_shader.matrix() << std::endl;
			unsigned loc;
			loc = glGetUniformLocation(shader_program_, "Trans");
			glUniformMatrix4fv(loc, 1, GL_FALSE, trans.data());
			loc = glGetUniformLocation(shader_program_, "Rot");
			glUniformMatrix3fv(loc, 1, GL_FALSE, rot.data());
			loc = glGetUniformLocation(shader_program_, "model");

			glUniformMatrix4fv(loc, 1, GL_FALSE, model_matrix_shader.data());

			if (render_data_body.body_ptr->geometry_enable_culling())
				glEnable(GL_CULL_FACE);
			else
				glDisable(GL_CULL_FACE);

			//换模型载入时，需要进行修改

			/*glBindVertexArray(render_data_body.vao);
			glDrawArrays(GL_TRIANGLES, 0, render_data_body.n_vertices);
			glBindVertexArray(0);*/
			Modelassimp model_assimp = *render_data_body.modelassimp_ptr.get();
			for (unsigned int i = 0; i < model_assimp.meshes.size(); i++)
			{
				// bind appropriate textures
				unsigned int diffuseNr = 1;
				unsigned int specularNr = 1;
				unsigned int normalNr = 1;
				unsigned int heightNr = 1;
				for (unsigned int j = 0; j < model_assimp.meshes[i].textures.size(); j++)
				{
					glActiveTexture(GL_TEXTURE0 + j); // active proper texture unit before binding
													  // retrieve texture number (the N in diffuse_textureN)
					string number;
					string name = model_assimp.meshes[i].textures[j].type;
					if (name == "texture_diffuse")
						number = std::to_string(diffuseNr++);
					else if (name == "texture_specular")
						number = std::to_string(specularNr++); // transfer unsigned int to stream
					else if (name == "texture_normal")
						number = std::to_string(normalNr++); // transfer unsigned int to stream
					else if (name == "texture_height")
						number = std::to_string(heightNr++); // transfer unsigned int to stream

															 // now set the sampler to the correct texture unit
					glUniform1i(glGetUniformLocation(shader_program_, (name + number).c_str()), j);
					// and finally bind the texture
					glBindTexture(GL_TEXTURE_2D, model_assimp.meshes[i].textures[j].id);
				}

				// draw mesh
				glBindVertexArray(model_assimp.meshes[i].VAO);
				//lnb
				//可有可无
				//通过ubo传入着色器中uniform变量
				GLuint Materi_id = glGetUniformBlockIndex(shader_program_, "Mat");
				//这里的0表示id，用于将uniform buffer和block uniform连接起来。
				glUniformBlockBinding(shader_program_, Materi_id, 0);

				glBindBufferRange(GL_UNIFORM_BUFFER, 0, model_assimp.meshes[i].uniformBlockIndex, 0, sizeof(Material));
				glDrawElements(GL_TRIANGLES, model_assimp.meshes[i].indices.size(), GL_UNSIGNED_INT, 0);
				glBindVertexArray(0);

				// always good practice to set everything back to defaults once configured.
				glActiveTexture(GL_TEXTURE0);
			}
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		renderer_geometry_ptr_->DetachContext();

		image_rendered_ = true;
		normal_image_fetched_ = false;
		depth_image_fetched_ = false;

		//cv::Mat img;
		//img.create(cv::Size{ intrinsics_.width, intrinsics_.height }, CV_8UC3);
		////use fast 4-byte alignment (default anyway) if possible
		//glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
		////set length of one complete row in destination data (doesn't need to equal img.cols)
		//glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
		//glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);//GL_DEPTH_COMPONENT  GL_RGB GL_FLOAT GL_UNSIGNED_BYTE
		//																		   //std::cout<<img.data[0]<<std::endl;
		//cv::Mat flipped;
		//cv::flip(img, img, 0);  //沿x轴翻转 
		//cv::namedWindow("texture_img", 0);
		//cv::imshow("texture_img", img);
		
		//FetchNormalImage();
		//cv::waitKey(0);
		return true;
	}

	bool teture_render::FetchNormalImage() {
		const std::lock_guard<std::mutex> lock{ mutex_ };
		if (!set_up_ || !image_rendered_) return false;
		if (normal_image_fetched_) return true;
		renderer_geometry_ptr_->MakeContextCurrent();
		glPixelStorei(GL_PACK_ALIGNMENT, (normal_image_.step & 3) ? 1 : 4);
		glPixelStorei(GL_PACK_ROW_LENGTH,
			GLint(normal_image_.step / normal_image_.elemSize()));
		glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo_normal_);
		glReadPixels(0, 0, normal_image_.cols, normal_image_.rows, GL_BGRA,
			GL_UNSIGNED_BYTE, normal_image_.data);   //GL_RGBA8
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		renderer_geometry_ptr_->DetachContext();
		normal_image_fetched_ = true;
		/*cv::namedWindow("teture_render", 0);
		cv::imshow("teture_render", normal_image_);
		cv::waitKey(0);*/

		return true;
	}

	bool teture_render::FetchDepthImage() {
		const std::lock_guard<std::mutex> lock{ mutex_ };
		if (!set_up_ || !image_rendered_) return false;
		if (depth_image_fetched_) return true;
		renderer_geometry_ptr_->MakeContextCurrent();
		glPixelStorei(GL_PACK_ALIGNMENT, (depth_image_.step & 3) ? 1 : 4);
		glPixelStorei(GL_PACK_ROW_LENGTH,
			GLint(depth_image_.step / depth_image_.elemSize()));
		glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth_);
		glReadPixels(0, 0, depth_image_.cols, depth_image_.rows, GL_DEPTH_COMPONENT,
			GL_UNSIGNED_SHORT, depth_image_.data);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		renderer_geometry_ptr_->DetachContext();
		depth_image_fetched_ = true;
		/*cv::namedWindow("depth", 0);
		cv::imshow("depth", depth_image_);
		cv::waitKey(0);*/
		return true;
	}

	const cv::Mat &teture_render::normal_image() const { return normal_image_; }

	const cv::Mat &teture_render::depth_image() const { return depth_image_; }

	float teture_render::depth_scale() const { return depth_scale_; }

	Eigen::Vector3f teture_render::GetPointVector(
		const cv::Point2i &image_coordinate) const {
		//通过图像坐标得到深度值
		float depth = depth_image_.at<ushort>(image_coordinate);
		depth = (projection_term_a_ / (projection_term_b_ - depth)) / depth_scale_;
		//返回在相机坐标系的坐标
		return Eigen::Vector3f{
			depth * (image_coordinate.x - intrinsics_.ppu) / intrinsics_.fu,
			depth * (image_coordinate.y - intrinsics_.ppv) / intrinsics_.fv, depth };
	}

	void teture_render::ClearImages() {
		normal_image_.create(cv::Size{ intrinsics_.width, intrinsics_.height },
			CV_8UC4);
		normal_image_.setTo(cv::Vec4b{ 0, 0, 0, 0 });
		depth_image_.create(cv::Size{ intrinsics_.width, intrinsics_.height }, CV_16U);
		depth_image_.setTo(cv::Scalar{ 0 });
	}

	void teture_render::CalculateProjectionTerms() {
		projection_term_a_ =
			depth_scale_ * z_max_ * z_min_ * USHRT_MAX / (z_max_ - z_min_);
		projection_term_b_ = z_max_ * USHRT_MAX / (z_max_ - z_min_);
	}

	void teture_render::CreateBufferObjects() {
		renderer_geometry_ptr_->MakeContextCurrent();

		// Initialize renderbuffer bodies_render_data
		glGenRenderbuffers(1, &rbo_normal_);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo_normal_);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, intrinsics_.width,
			intrinsics_.height);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glGenRenderbuffers(1, &rbo_depth_);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth_);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16,
			intrinsics_.width, intrinsics_.height);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		// Initialize framebuffer bodies_render_data
		glGenFramebuffers(1, &fbo_);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_RENDERBUFFER, rbo_normal_);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_RENDERBUFFER, rbo_depth_);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		renderer_geometry_ptr_->DetachContext();
	}

	void teture_render::DeleteBufferObjects() {
		if (renderer_geometry_ptr_ != nullptr) {
			renderer_geometry_ptr_->MakeContextCurrent();
			glDeleteRenderbuffers(1, &rbo_normal_);
			glDeleteRenderbuffers(1, &rbo_depth_);
			glDeleteFramebuffers(1, &fbo_);
			renderer_geometry_ptr_->DetachContext();
		}
	}

}