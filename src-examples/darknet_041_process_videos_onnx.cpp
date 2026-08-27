/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2026 Stephane Charette
 */

#include "darknet.hpp"
#include "darknet_cfg_and_state.hpp"
#include <cctype>

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <fstream>


namespace
{
	struct Detection
	{
		int class_id = -1;
		float confidence = 0.0f;
		cv::Rect rect;
	};

	struct NetDims
	{
		int width = 0;
		int height = 0;
		int channels = 0;
	};


	std::filesystem::path find_onnx_filename(const Darknet::Parms & parms)
	{
		for (const auto & parm : parms)
		{
			if (parm.type != Darknet::EParmType::kFilename)
			{
				continue;
			}

			const std::filesystem::path path(parm.string);
			if (path.extension() == ".onnx")
			{
				return path;
			}
		}

		for (const auto & parm : parms)
		{
			if (parm.type != Darknet::EParmType::kOther)
			{
				continue;
			}

			std::filesystem::path path(parm.string);
			path.replace_extension(".onnx");
			if (std::filesystem::exists(path))
			{
				return path;
			}
		}

		return {};
	}

	NetDims load_network_dims_from_cfg(const std::filesystem::path & cfg_fn)
	{
		std::ifstream ifs(cfg_fn);
		if (not ifs.good())
		{
			throw std::runtime_error("failed to open cfg file " + cfg_fn.string());
		}

		bool in_net_section = false;
		NetDims dims;

		std::string line;
		while (std::getline(ifs, line))
		{
			Darknet::trim(line);
			if (line.empty())
			{
				continue;
			}

			const auto comment_pos = line.find('#');
			if (comment_pos != std::string::npos)
			{
				line.erase(comment_pos);
				Darknet::trim(line);
				if (line.empty())
				{
					continue;
				}
			}

			if (line.front() == '[' && line.back() == ']')
			{
				std::string section = line.substr(1, line.size() - 2);
				Darknet::trim(section);
				for (auto & c : section)
				{
					c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
				}
				in_net_section = (section == "net" || section == "network");
				continue;
			}

			if (not in_net_section)
			{
				continue;
			}

			const auto eq_pos = line.find('=');
			if (eq_pos == std::string::npos)
			{
				continue;
			}

			std::string key = line.substr(0, eq_pos);
			std::string val = line.substr(eq_pos + 1);
			Darknet::trim(key);
			Darknet::trim(val);
			for (auto & c : key)
			{
				c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
			}

			if (key == "width")
			{
				dims.width = std::stoi(val);
			}
			else if (key == "height")
			{
				dims.height = std::stoi(val);
			}
			else if (key == "channels")
			{
				dims.channels = std::stoi(val);
			}
		}

		if (dims.width <= 0 || dims.height <= 0)
		{
			throw std::runtime_error("failed to find width/height in cfg file " + cfg_fn.string());
		}
		if (dims.channels <= 0)
		{
			dims.channels = 3;
		}

		return dims;
	}


	std::vector<std::string> load_names_file(const std::filesystem::path & filename)
	{
		std::vector<std::string> names;
		if (filename.empty() || not std::filesystem::exists(filename))
		{
			return names;
		}

		std::ifstream ifs(filename);
		if (not ifs.good())
		{
			return names;
		}

		std::string line;
		while (std::getline(ifs, line))
		{
			Darknet::trim(line);
			if (not line.empty())
			{
				names.push_back(line);
			}
		}

		return names;
	}


	void ensure_class_names(std::vector<std::string> & names, const int count)
	{
		if (count <= 0)
		{
			return;
		}

		if (static_cast<int>(names.size()) < count)
		{
			for (int idx = names.size(); idx < count; idx ++)
			{
				names.push_back("class_" + std::to_string(idx));
			}
		}
	}


	std::vector<cv::Scalar> build_class_colours(const size_t count)
	{
		std::vector<cv::Scalar> colours;
		colours.reserve(count);

		for (size_t idx = 0; idx < count; idx ++)
		{
			const int hue = (idx * 37) % 180;
			cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 255));
			cv::Mat bgr;
			cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
			const auto pixel = bgr.at<cv::Vec3b>(0, 0);
			colours.emplace_back(pixel[0], pixel[1], pixel[2]);
		}

		return colours;
	}


	void resolve_output_names(cv::dnn::Net & net, std::vector<cv::String> & output_names, int & boxes_idx, int & confs_idx)
	{
		const auto names = net.getUnconnectedOutLayersNames();
		cv::String boxes_name;
		cv::String confs_name;

		for (const auto & name : names)
		{
			if (name == "boxes")
			{
				boxes_name = name;
			}
			else if (name == "confs")
			{
				confs_name = name;
			}
		}

		if (not boxes_name.empty() && not confs_name.empty())
		{
			output_names = {boxes_name, confs_name};
			boxes_idx = 0;
			confs_idx = 1;
			return;
		}

		output_names = names;
		boxes_idx = -1;
		confs_idx = -1;

		for (int idx = 0; idx < output_names.size(); idx ++)
		{
			if (output_names[idx].find("boxes") != cv::String::npos)
			{
				boxes_idx = idx;
			}
			else if (output_names[idx].find("confs") != cv::String::npos)
			{
				confs_idx = idx;
			}
		}

		if (boxes_idx < 0 || confs_idx < 0)
		{
			throw std::runtime_error("failed to locate the expected ONNX outputs \"boxes\" and \"confs\"");
		}
	}


	std::vector<Detection> run_inference(
		cv::dnn::Net & net,
		cv::Mat & frame,
		const int net_w,
		const int net_h,
		const float thresh,
		const float nms_thresh,
		std::vector<std::string> & class_names,
		std::vector<cv::Scalar> & class_colours,
		const std::vector<cv::String> & output_names,
		const int boxes_idx,
		const int confs_idx,
		const int input_depth,
		const double input_scale)
	{
		cv::Mat blob = cv::dnn::blobFromImage(frame, input_scale, cv::Size(net_w, net_h), cv::Scalar(), true, false, input_depth);
		net.setInput(blob);

		std::vector<cv::Mat> outputs;
		net.forward(outputs, output_names);

		cv::Mat boxes = outputs.at(boxes_idx);
		cv::Mat confs = outputs.at(confs_idx);

		if (boxes.type() != CV_32F)
		{
			boxes.convertTo(boxes, CV_32F);
		}
		if (confs.type() != CV_32F)
		{
			confs.convertTo(confs, CV_32F);
		}
		if (not boxes.isContinuous())
		{
			boxes = boxes.clone();
		}
		if (not confs.isContinuous())
		{
			confs = confs.clone();
		}

		if (boxes.total() % 4 != 0)
		{
			throw std::runtime_error("unexpected ONNX output shape for \"boxes\"");
		}
		const int num_boxes = boxes.total() / 4;
		if (num_boxes == 0)
		{
			return {};
		}

		if (confs.total() % num_boxes != 0)
		{
			throw std::runtime_error("unexpected ONNX output shape for \"confs\"");
		}
		const int num_classes = confs.total() / num_boxes;

		ensure_class_names(class_names, num_classes);
		if (class_colours.size() != class_names.size())
		{
			class_colours = build_class_colours(class_names.size());
		}

		const cv::Mat boxes_2d = boxes.reshape(1, num_boxes);
		const cv::Mat confs_2d = confs.reshape(1, num_boxes);

		std::vector<Detection> detections;
		detections.reserve(num_boxes);

		const float scale_x = static_cast<float>(frame.cols) / static_cast<float>(net_w);
		const float scale_y = static_cast<float>(frame.rows) / static_cast<float>(net_h);

		for (int idx = 0; idx < num_boxes; idx ++)
		{
			const float * box = boxes_2d.ptr<float>(idx);
			const float * conf = confs_2d.ptr<float>(idx);

			int best_class = -1;
			float best_conf = 0.0f;
			for (int c = 0; c < num_classes; c ++)
			{
				if (conf[c] > best_conf)
				{
					best_conf = conf[c];
					best_class = c;
				}
			}

			if (best_class < 0 || best_conf < thresh)
			{
				continue;
			}

			const int left = std::clamp(static_cast<int>(std::round(box[0] * scale_x)), 0, frame.cols - 1);
			const int top = std::clamp(static_cast<int>(std::round(box[1] * scale_y)), 0, frame.rows - 1);
			const int right = std::clamp(static_cast<int>(std::round(box[2] * scale_x)), 0, frame.cols - 1);
			const int bottom = std::clamp(static_cast<int>(std::round(box[3] * scale_y)), 0, frame.rows - 1);

			if (right <= left || bottom <= top)
			{
				continue;
			}

			Detection det;
			det.class_id = best_class;
			det.confidence = best_conf;
			det.rect = cv::Rect(left, top, right - left, bottom - top);
			detections.push_back(det);
		}

		std::vector<Detection> filtered;
		if (detections.empty())
		{
			return filtered;
		}

		std::vector<std::vector<int>> per_class(num_classes);
		for (int idx = 0; idx < detections.size(); idx ++)
		{
			const int class_id = detections[idx].class_id;
			if (class_id >= 0 && class_id < num_classes)
			{
				per_class[class_id].push_back(idx);
			}
		}

		for (int class_id = 0; class_id < num_classes; class_id ++)
		{
			const auto & indices = per_class[class_id];
			if (indices.empty())
			{
				continue;
			}

			std::vector<cv::Rect> boxes_for_class;
			std::vector<float> scores_for_class;
			std::vector<int> index_map;
			boxes_for_class.reserve(indices.size());
			scores_for_class.reserve(indices.size());
			index_map.reserve(indices.size());

			for (const int idx : indices)
			{
				boxes_for_class.push_back(detections[idx].rect);
				scores_for_class.push_back(detections[idx].confidence);
				index_map.push_back(idx);
			}

			std::vector<int> kept;
			cv::dnn::NMSBoxes(boxes_for_class, scores_for_class, thresh, nms_thresh, kept);
			for (const int kept_idx : kept)
			{
				filtered.push_back(detections.at(index_map.at(kept_idx)));
			}
		}

		for (const auto & det : filtered)
		{
			const auto & name = class_names.at(det.class_id);
			const auto & colour = class_colours.at(det.class_id);
			cv::rectangle(frame, det.rect, colour, 2);

			const int percent = static_cast<int>(std::round(100.0f * det.confidence));
			const std::string label = name + " " + std::to_string(percent) + "%";

			int baseline = 0;
			const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
			int label_x = det.rect.x;
			int label_y = det.rect.y - text_size.height - baseline;
			if (label_y < 0)
			{
				label_y = 0;
			}
			if (label_x + text_size.width + 2 > frame.cols)
			{
				label_x = std::max(0, frame.cols - text_size.width - 2);
			}

			cv::Rect label_rect(label_x, label_y, text_size.width + 2, text_size.height + baseline);
			cv::rectangle(frame, label_rect, colour, cv::FILLED);
			cv::putText(frame, label, cv::Point(label_x + 1, label_y + text_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
		}

		return filtered;
	}


	bool is_opencv_dtype_error(const std::string & message)
	{
		std::string upper = message;
		std::transform(upper.begin(), upper.end(), upper.begin(),
			[](unsigned char c) { return static_cast<char>(std::toupper(c)); });
		return upper.find("FLOAT16") != std::string::npos ||
			upper.find("DEQUANTIZELINEAR") != std::string::npos ||
			upper.find("INT8") != std::string::npos ||
			upper.find("QUANT") != std::string::npos;
	}
}


/** @file
 * This application will process one or more videos as fast as possible on a single thread and save a new output video
 * to disk.  The results are not shown to the user.  Call it like this:
 *
 *     darknet_041_process_videos LegoGears.cfg LegoGears.onnx DSCN1582A.MOV
 *
 * The output should be similar to this:
 *
 *     processing DSCN1582A.MOV:
 *     -> neural network size ...... 224 x 160 x 3
 *     -> input video dimensions ... 640 x 480
 *     -> input video frame count .. 1230
 *     -> input video frame rate ... 29.970030 FPS
 *     -> input video length ....... 41041 milliseconds
 *     -> output filename .......... DSCN1582A_output.m4v
 *     -> total frames processed ... 1230
 *     -> time to process video .... 3207 milliseconds
 *     -> processed frame rate ..... 383.536015 FPS
 *     -> total objects found ...... 6189
 *     -> average objects/frame .... 5.031707
 */


int main(int argc, char * argv[])
{
	try
	{
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		const auto cfg_fn = Darknet::get_config_filename(parms);
		const auto onnx_fn = find_onnx_filename(parms);
		if (cfg_fn.empty() || onnx_fn.empty())
		{
			std::cout << "Usage: darknet_041_process_videos <model.cfg> <model.onnx> <video file>" << std::endl;
			return 1;
		}

		const NetDims dims = load_network_dims_from_cfg(cfg_fn);
		const int network_width = dims.width;
		const int network_height = dims.height;
		const int network_channels = dims.channels;

		cv::dnn::Net net;
		try
		{
			net = cv::dnn::readNetFromONNX(onnx_fn.string());
		}
		catch (const cv::Exception & e)
		{
			const std::string msg = e.what();
			if (is_opencv_dtype_error(msg))
			{
				std::cout << "OpenCV cannot load this ONNX model. It appears to use FP16 or INT8, which OpenCV DNN 4.5.4 does not support." << std::endl;
			}
			std::cout << "Exception: " << msg << std::endl;
			return 1;
		}
		if (net.empty())
		{
			std::cout << "Failed to open the ONNX file " << onnx_fn << std::endl;
			return 1;
		}

		std::vector<cv::String> output_names;
		int boxes_idx = -1;
		int confs_idx = -1;
		resolve_output_names(net, output_names, boxes_idx, confs_idx);

		auto names_fn = Darknet::get_names_filename(parms);
		if (names_fn.empty())
		{
			names_fn = cfg_fn;
			names_fn.replace_extension(".names");
			if (not std::filesystem::exists(names_fn))
			{
				names_fn.clear();
			}
		}

		std::vector<std::string> class_names = load_names_file(names_fn);
		std::vector<cv::Scalar> class_colours = build_class_colours(class_names.size());

		const auto & cfg = Darknet::CfgAndState::get();
		const float thresh = cfg.get("thresh", 0.24f);
		const float nms_thresh = cfg.get("nms", 0.45f);

		const bool use_int8 = cfg.is_set("int8");
		const int input_depth = use_int8 ? CV_8U : CV_32F;
		const double input_scale = use_int8 ? 1.0 : (1.0 / 255.0);

		for (const auto & parm : parms)
		{
			if (parm.type != Darknet::EParmType::kFilename)
			{
				continue;
			}

			const std::filesystem::path path(parm.string);
			if (path.extension() == ".onnx")
			{
				continue;
			}

			std::cout << "processing " << parm.string << ":" << std::endl;

			cv::VideoCapture cap(parm.string);
			if (not cap.isOpened())
			{
				std::cout << "Failed to open the input video file " << parm.string << std::endl;
				continue;
			}

			cv::Mat mat;
			cap >> mat;
			cap.set(cv::CAP_PROP_POS_FRAMES, 0.0);

			const std::string output_filename			= path.stem().string() + "_output.m4v";
			const size_t video_width					= mat.cols;
			const size_t video_height					= mat.rows;
			const size_t video_channels				= mat.channels();
			const size_t video_frames_count			= cap.get(cv::CAP_PROP_FRAME_COUNT);
			const double fps							= cap.get(cv::CAP_PROP_FPS);
			const size_t fps_rounded					= std::round(fps);
			const size_t nanoseconds_per_frame			= std::round(1000000000.0 / fps);
			const size_t video_length_milliseconds		= std::round(nanoseconds_per_frame / 1000000.0 * video_frames_count);

			std::cout
				<< "-> neural network size ...... " << network_width	<< " x " << network_height	<< " x " << network_channels	<< std::endl
				<< "-> input video dimensions ... " << video_width		<< " x " << video_height	<< " x " << video_channels		<< std::endl
				<< "-> input video frame count .. " << video_frames_count							<< std::endl
				<< "-> input video frame rate ... " << fps << " FPS"								<< std::endl
				<< "-> input video length ....... " << Darknet::format_duration_string(std::chrono::milliseconds(video_length_milliseconds)) << std::endl
				<< "-> output filename .......... " << output_filename								<< std::endl;

			cv::VideoWriter out(output_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(video_width, video_height));
			if (not out.isOpened())
			{
				std::cout << "Failed to open the output video file " << output_filename << std::endl;
				continue;
			}

			size_t frame_counter = 0;
			size_t total_objects_found = 0;
			const auto timestamp_when_video_started = std::chrono::high_resolution_clock::now();

			while (true)
			{
				cap >> mat;
				if (mat.empty())
				{
					break;
				}

				const auto results = run_inference(net, mat, network_width, network_height, thresh, nms_thresh, class_names, class_colours, output_names, boxes_idx, confs_idx, input_depth, input_scale);
				out.write(mat);
				frame_counter ++;
				total_objects_found += results.size();

				if (frame_counter % fps_rounded == 0)
				{
					const int percentage = std::round(100.0f * frame_counter / video_frames_count);
					std::cout
						<< "-> frame #" << frame_counter << "/" << video_frames_count
						<< " (" << percentage << "%)\r"
						<< std::flush;
				}
			}

			const auto timestamp_when_video_ended = std::chrono::high_resolution_clock::now();
			const auto processing_duration = timestamp_when_video_ended - timestamp_when_video_started;
			const size_t processing_time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(processing_duration).count();
			const double final_fps = 1000.0 * frame_counter / processing_time_in_milliseconds;

			std::cout
				<< "-> total frames processed ... " << frame_counter											<< std::endl
				<< "-> time to process video .... " << Darknet::format_duration_string(processing_duration)		<< std::endl
				<< "-> processed frame rate ..... " << final_fps << " FPS"										<< std::endl
				<< "-> total objects found ...... " << total_objects_found										<< std::endl
				<< "-> average objects/frame .... " << static_cast<float>(total_objects_found) / frame_counter	<< std::endl;
		}
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
