#include "darknet_internal.hpp"
#include "centernet_layer.hpp"
#include "yolonas_layer.hpp"
#include "ppyoloe_layer.hpp"
#include "yolox_layer.hpp"
#ifdef DARKNET_HAS_FP4
#include "fp4_gemm.hpp"
#endif
#ifdef DARKNET_HAS_FP8
#include "fp8_gemm.hpp"
#endif
#include "gemm.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

#if defined(DARKNET_GPU) && defined(CUDNN)
	static void set_layer_cudnn_bf16_mode(Darknet::Layer & l)
	{
#if defined(CUDNN_HALF) && defined(CUDNN_DATA_BFLOAT16)
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL || l.type == Darknet::ELayerType::CONNECTED)
		{
			set_convolutional_cudnn_16bit_mode(&l, DARKNET_CUDNN_16BIT_BF16);
			if (l.antialiasing && l.input_layer)
			{
				set_layer_cudnn_bf16_mode(*(l.input_layer));
			}
		}
		else if (l.type == Darknet::ELayerType::DCNV4)
		{
			set_dcnv4_cudnn_16bit_mode(&l, DARKNET_CUDNN_16BIT_BF16);
		}
		else if (l.type == Darknet::ELayerType::CRNN)
		{
			if (l.input_layer)
			{
				set_layer_cudnn_bf16_mode(*(l.input_layer));
			}
			if (l.self_layer)
			{
				set_layer_cudnn_bf16_mode(*(l.self_layer));
			}
			if (l.output_layer)
			{
				set_layer_cudnn_bf16_mode(*(l.output_layer));
			}
		}
		else if (l.type == Darknet::ELayerType::RNN)
		{
			if (l.input_layer) set_layer_cudnn_bf16_mode(*(l.input_layer));
			if (l.self_layer) set_layer_cudnn_bf16_mode(*(l.self_layer));
			if (l.output_layer) set_layer_cudnn_bf16_mode(*(l.output_layer));
		}
		else if (l.type == Darknet::ELayerType::LSTM)
		{
			if (l.wf) set_layer_cudnn_bf16_mode(*(l.wf));
			if (l.wi) set_layer_cudnn_bf16_mode(*(l.wi));
			if (l.wg) set_layer_cudnn_bf16_mode(*(l.wg));
			if (l.wo) set_layer_cudnn_bf16_mode(*(l.wo));
			if (l.uf) set_layer_cudnn_bf16_mode(*(l.uf));
			if (l.ui) set_layer_cudnn_bf16_mode(*(l.ui));
			if (l.ug) set_layer_cudnn_bf16_mode(*(l.ug));
			if (l.uo) set_layer_cudnn_bf16_mode(*(l.uo));
		}
#else
		(void) l;
#endif
	}
#endif

	static void set_train_only_bn(Darknet::Network & net)
	{
		TAT(TATPARMS);

		// BN == "batch normalization"

		int train_only_bn = 0;

		for (auto idx = net.n - 1; idx >= 0; --idx)
		{
			if (net.layers[idx].train_only_bn)
			{
				// set l.train_only_bn for all previous layers
				train_only_bn = net.layers[idx].train_only_bn;
			}

			if (train_only_bn)
			{
				net.layers[idx].train_only_bn = train_only_bn;

				if (net.layers[idx].type == Darknet::ELayerType::CRNN)
				{
					net.layers[idx].input_layer	->train_only_bn = train_only_bn;
					net.layers[idx].self_layer	->train_only_bn = train_only_bn;
					net.layers[idx].output_layer->train_only_bn = train_only_bn;
				}
			}
		}

		return;
	}


	static float * get_classes_multipliers(Darknet::VInt & vi, const int classes, const float max_delta)
	{
		TAT(TATPARMS);

		float *classes_multipliers = nullptr;

		if (not vi.empty())
		{
			const int * counters_per_class = vi.data();
			if (vi.size() != classes)
			{
				darknet_fatal_error(DARKNET_LOC, "number of values in counters_per_class=%ld doesn't match classes=%d", vi.size(), classes);
			}

			float max_counter = 0.0f;
			for (auto & i : vi)
			{
				if (i < 1)
				{
					i = 1;
				}
				if (max_counter < i)
				{
					max_counter = i;
				}
			}

			classes_multipliers = (float *)calloc(vi.size(), sizeof(float));

			for (size_t i = 0; i < vi.size(); ++i)
			{
				classes_multipliers[i] = max_counter / counters_per_class[i];
				if (classes_multipliers[i] > max_delta)
				{
					classes_multipliers[i] = max_delta;
				}
			}

			*cfg_and_state.output << "classes_multipliers: ";
			for (size_t i = 0; i < vi.size(); ++i)
			{
				*cfg_and_state.output << classes_multipliers[i] << ", ";
			}
			*cfg_and_state.output << std::endl;
		}

		return classes_multipliers;
	}
}


Darknet::CfgLine::CfgLine() :
	line_number(0),
	used(false)
{
	TAT(TATPARMS);
	return;
}


Darknet::CfgLine::CfgLine(const std::string & l, const size_t ln, const std::string & lhs, const std::string & rhs) :
	line_number(ln),
	line(l),
	key(lhs),
	val(rhs),
	used(false)
{
	TAT(TATPARMS);

	if (val.size() > 0 and val.find_first_not_of("-.0123456789") == std::string::npos)
	{
		// looks like this might be a number!
		try
		{
			f = std::stof(val);
		}
		catch (...)
		{
			darknet_fatal_error(DARKNET_LOC, "line #%d: failed to parse numeric value from \"%s\"", line_number, l.c_str());
		}
	}

//	*cfg_and_state.output << "#" << line_number << ": line=\"" << line << "\" key=" << key << " val=" << val << std::endl;

	return;
};


Darknet::CfgLine::~CfgLine()
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgLine & Darknet::CfgLine::clear()
{
	TAT(TATPARMS);

	line_number	= 0;
	used		= false;
	line		.clear();
	key			.clear();
	val			.clear();
	f			.reset();

	return *this;
}


std::string Darknet::CfgLine::debug() const
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << line_number << " used=" << (used ? "YES" : "no") << " key=" << key << " val=" << val;

	if (f.has_value())
	{
		ss << " f=" << f.value();
	}

	return ss.str();
}


Darknet::CfgSection::CfgSection() :
	type(ELayerType::BLANK),
	line_number(0),
	index(0)
{
	TAT(TATPARMS);
	return;
}


Darknet::CfgSection::CfgSection(const std::string & l, const size_t ln) :
	type(Darknet::get_layer_type_from_name(l)),
	name(l),
	line_number(ln),
	index(0)
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgSection::~CfgSection()
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgSection & Darknet::CfgSection::clear()
{
	TAT(TATPARMS);

	line_number	= 0;
	type		= ELayerType::EMPTY;
	name		.clear();
	lines		.clear();

	return *this;
}


const Darknet::CfgSection & Darknet::CfgSection::find_unused_lines() const
{
	TAT(TATPARMS);

	for (const auto & [key, line] : lines)
	{
		if (line.used == false)
		{
			darknet_fatal_error(DARKNET_LOC, "config line #%ld from section [%s] is unused or invalid: %s", line.line_number, name.c_str(), line.line.c_str());
		}
	}

	return *this;
}


bool Darknet::CfgSection::exists(const std::string & key) const
{
	TAT(TATPARMS);

	return lines.count(key) > 0;
}


int Darknet::CfgSection::find_int(const std::string & key)
{
	TAT(TATPARMS);

	if (lines.count(key) == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "section [%s] at line %ld was expecting to find a key named \"%s\" but it does not exist", name.c_str(), line_number, key.c_str());
	}

	return find_int(key, 0);
}


int Darknet::CfgSection::find_int(const std::string & key, const int default_value)
{
	TAT(TATPARMS);

	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;
		if (l.f)
		{
			float iptr = 0.0f;
			float frac = std::modf(l.f.value(), &iptr);

			if (frac != 0.0f)
			{
				*cfg_and_state.output << "-> expected an integer for \"" << key << "\" on line #" << l.line_number << " but found \"" << l.f.value() << "\"" << std::endl;
			}

			const int i = static_cast<int>(iptr);

			if (cfg_and_state.is_trace)
			{
				*cfg_and_state.output << "[" << name << "] #" << l.line_number << " " << key << "=" << i << std::endl;
			}

			l.used = true;
			return i;
		}
		else
		{
			darknet_fatal_error(DARKNET_LOC, "key \"%s\" on line #%ld should be numeric, but found \"%s\"", key.c_str(), l.line_number, l.val.c_str());
		}
	}

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "[" << name << "] #" << line_number << " DEFAULT " << key << "=" << default_value << std::endl;
	}

	return default_value;
}


float Darknet::CfgSection::find_float(const std::string & key, const float default_value)
{
	TAT(TATPARMS);

	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;
		if (l.f)
		{
			const float f = l.f.value();

			if (cfg_and_state.is_trace)
			{
				*cfg_and_state.output << "[" << name << "] #" << l.line_number << " " << key << "=" << f << std::endl;
			}

			l.used = true;
			return f;
		}
		else
		{
			darknet_fatal_error(DARKNET_LOC, "key \"%s\" on line #%ld should be numeric, but found \"%s\"", key.c_str(), l.line_number, l.val.c_str());
		}
	}

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "[" << name << "] #" << line_number << " DEFAULT " << key << "=" << default_value << std::endl;
	}

	return default_value;
}


std::string Darknet::CfgSection::find_str(const std::string & key, const std::string & default_value)
{
	TAT(TATPARMS);

	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;

		if (cfg_and_state.is_trace)
		{
			*cfg_and_state.output << "[" << name << "] #" << l.line_number << " " << key << "=" << l.val << std::endl;
		}

		l.used = true;
		return l.val;
	}

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "[" << name << "] #" << line_number << " DEFAULT " << key << "=" << default_value << std::endl;
	}

	return default_value;
}


Darknet::VFloat Darknet::CfgSection::find_float_array(const std::string & key)
{
	TAT(TATPARMS);

	VFloat v;
	auto line = line_number;

	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;
		l.used = true;
		line = l.line_number;
		auto val = l.val;

		size_t pos = 0;
		while (pos < val.size())
		{
			const size_t digit = val.find_first_of("-.0123456789", pos);
			if (digit == std::string::npos)
			{
				// no numbers left to read
				break;
			}

			const size_t comma = val.find_first_not_of("-.0123456789", digit);
			std::string tmp;
			if (comma == std::string::npos)
			{
				tmp = val.substr(digit);
			}
			else
			{
				tmp = val.substr(digit, comma - digit);
			}

			try
			{
				const float f = std::stof(tmp);
				v.push_back(f);
			}
			catch(...)
			{
				break;
			}

			pos = comma;
		}
	}

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "[" << name << "] #" << line << " " << key << "=[";
		for (size_t idx = 0; idx < v.size(); idx ++)
		{
			if (idx > 0) *cfg_and_state.output << ", ";
			*cfg_and_state.output << v[idx];
		}
		*cfg_and_state.output << "]" << std::endl;
	}

	return v;
}


Darknet::VInt Darknet::CfgSection::find_int_array(const std::string & key)
{
	TAT(TATPARMS);

	VInt v;
	auto line = line_number;

	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;
		l.used = true;
		line = l.line_number;
		auto val = l.val;

		size_t pos = 0;
		while (pos < val.size())
		{
			const size_t digit = val.find_first_of("-0123456789", pos);
			if (digit == std::string::npos)
			{
				// no numbers left to read
				break;
			}

			const size_t comma = val.find_first_not_of("-0123456789", digit);
			std::string tmp;
			if (comma == std::string::npos)
			{
				tmp = val.substr(digit);
			}
			else
			{
				tmp = val.substr(digit, comma - digit);
			}

			try
			{
				const int i = std::stoi(tmp);
				v.push_back(i);
			}
			catch(...)
			{
				break;
			}

			pos = comma;
		}
	}

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "[" << name << "] #" << line << " " << key << "=[";
		for (size_t idx = 0; idx < v.size(); idx ++)
		{
			if (idx > 0) *cfg_and_state.output << ", ";
			*cfg_and_state.output << v[idx];
		}
		*cfg_and_state.output << "]" << std::endl;
	}

	return v;
}


std::string Darknet::CfgSection::debug() const
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << line_number << ": [" << Darknet::to_string(type) << "]" << std::endl;

	for (const auto & [key, line] : lines)
	{
		ss << line.debug() << std::endl;
	}

	return ss.str();
}


Darknet::CfgFile::CfgFile() :
	total_lines(0)
{
	TAT(TATPARMS);

	net = {0};
	parms = {0};

	return;
}


Darknet::CfgFile::CfgFile(const std::filesystem::path & fn) :
	CfgFile()
{
	TAT(TATPARMS);

	read(fn);

	return;
}


Darknet::CfgFile::~CfgFile()
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgFile & Darknet::CfgFile::clear()
{
	TAT(TATPARMS);

	filename		.clear();
	network_section	.clear();
	sections		.clear();
	total_lines		= 0;
	net				= {0};
	parms			= {0};

	return *this;
}


Darknet::CfgFile & Darknet::CfgFile::read(const std::filesystem::path & fn)
{
	TAT(TATPARMS);

	filename = fn;

	return read();
}


Darknet::CfgFile & Darknet::CfgFile::read()
{
	TAT(TATPARMS);

	total_lines = 0;
	sections.clear();

	if (not std::filesystem::exists(filename))
	{
		darknet_fatal_error(DARKNET_LOC, "config file does not exist: \"%s\"", filename.string().c_str());
	}

	filename = std::filesystem::canonical(filename);

	if (filename.extension() != ".cfg")
	{
		// Not necessarily an error...maybe the user has named their .cfg files something else.
		// But in most cases, if someone uses a .names or .weights file in the place of a .cfg
		// then Darknet will obviously not run correctly, so at least attempt to warn them.

		Darknet::display_warning_msg("expected a .cfg filename but got this instead: " + filename.string() + "\n");
	}

	/* find lines such as these:
	 *
	 *		[net]
	 *		width=224
	 *		height=160 # this is a test
	 *		channels = 3
	 *		momentum=0.9
	 *
	 * ...and parse the following 3 things:
	 *
	 *		1) The name of the sections, such as "net" in "[net]"
	 *		2) The name of the key, such as "channels" in "channels=3"
	 *		3) The name of the value, such as "0.9" in "momentum=0.9"
	 */
	const std::regex rx(
		"^"				// start of line
		"\\["			// [
		"(.+)"			// group #1:  section name
		"\\]"			// ]
		"|"				// ...or...
		"^"				// start of line
		"[ \t]*"		// optional whitespace
		"("				// group #2
		"[^#= \t]+"		// key (everything up to #, =, or whitespace)
		")"				// end of group #2
		"[ \t]*"		// optional whitespace
		"="				// =
		"[ \t]*"		// optional whitespace
		"("				// group #3
		"[^#]+"			// value (everything up to #)
		")"				// end of group #3
		);

	std::ifstream ifs(filename);

	std::string line;
	while (std::getline(ifs, line))
	{
		total_lines ++;

		// strip whitespace at the end of line, which should help us ignore \n and \r\n problems between Windows and Linux
		Darknet::trim(line);

		std::smatch matches;

		if (std::regex_search(line, matches, rx))
		{
			// line is either a section name, or a key-value pair -- check for a section name first

			const std::string section_name = matches.str(1);
			if (section_name.size() > 0)
			{
				if (network_section.empty())
				{
					network_section = CfgSection(section_name, total_lines);
				}
				else
				{
					sections.emplace_back(CfgSection(section_name, total_lines));
				}
			}
			else
			{
				// a section must exist prior to reading a key=val pair
				if (network_section.line_number == 0 and sections.empty())
				{
					darknet_fatal_error(DARKNET_LOC, "cannot add a configuration line without a section at line #%ld in %s", total_lines, filename.string().c_str());
				}

				const std::string key = Darknet::trim(matches.str(2));
				const std::string val = Darknet::trim(matches.str(3));

				auto & s = (sections.empty() == false ? sections.back() : network_section);

				/* 2024-06-13:  As far as I know, each section should have unique keys.  The lines in each sections are stored in a
				 * map, meaning duplicates are not allowed.  If this is not the case and a particular layer can have duplicate keys
				 * (lines) then we'll have to change out the map for a vector, and fix how the "lines" processing works.  For now,
				 * do a quick check to see if this key already exists in this section, and abort loading the configuration.
				 */
				if (s.lines.count(key))
				{
					const auto iter = s.lines.find(key);
					darknet_fatal_error(DARKNET_LOC, "duplicate key \"%s\" in [%s] at lines #%ld and #%ld in %s", key.c_str(), s.name.c_str(), iter->second.line_number, total_lines, filename.string().c_str());
				}

				s.lines[key] = CfgLine(line, total_lines, key, val);
			}
		}
	}

	// go back and number all of layers (sections)
	for (size_t index = 0; index < sections.size(); index ++)
	{
		sections[index].index = index;
	}

	return *this;
}


std::string Darknet::CfgFile::debug() const
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss	<< "Filename: " << filename.string() << std::endl
		<< "Lines parsed: " << total_lines << std::endl;

	for (const auto & section : sections)
	{
		ss << section.debug() << std::endl;
	}

	return ss.str();
}


Darknet::Network & Darknet::CfgFile::create_network(int batch, int time_steps)
{
	TAT(TATPARMS);

	// taken from:  parse_network_cfg_custom()

	if (sections.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "cannot create a network from empty configuration file \"%s\"", filename.string().c_str());
	}

	if (network_section.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "configuration file %s does not contain a [net] or [network] section", filename.string().c_str());
	}

	if (network_section.type != ELayerType::NETWORK)
	{
		darknet_fatal_error(DARKNET_LOC, "expected to find [net] or [network], but instead found [%s] on line #%ld of %s", network_section.name.c_str(), network_section.line_number, filename.string().c_str());
	}

	net = make_network(sections.size());
	net.gpu_index = cfg_and_state.gpu_index;
	net.details->cfg_path = filename;

	if (batch > 0)
	{
		// allocates memory for inference only
		parms.train = 0;
	}
	else
	{
		// allocates memory for inference & training
		parms.train = 1;
	}

	parse_net_section();

#ifdef DARKNET_GPU
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "net.optimized_memory=" << net.optimized_memory << std::endl;
	}
	if (net.optimized_memory >= 2 && parms.train)
	{
		pre_allocate_pinned_memory((size_t)1024 * 1024 * 1024 * 8);   // pre-allocate 8 GB CPU-RAM for pinned memory
	}
#endif  // DARKNET_GPU

	parms.h = net.h;
	parms.w = net.w;
	parms.c = net.c;
	parms.inputs = net.inputs;
	parms.last_stop_backward = -1;

	if (batch > 0)						net.batch		= batch;
	if (time_steps > 0)					net.time_steps	= time_steps;
	if (net.batch < 1)					net.batch		= 1;
	if (net.time_steps < 1)				net.time_steps	= 1;
	if (net.batch < net.time_steps)		net.batch		= net.time_steps;

	parms.batch			= net.batch;
	parms.time_steps	= net.time_steps;

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output
			<< "mini_batch="	<< net.batch
			<< ", batch="		<< net.batch * net.subdivisions
			<< ", time_steps="	<< net.time_steps
			<< ", train="		<< parms.train
			<< std::endl;
	}

	parms.avg_outputs			= 0;
	parms.avg_counter			= 0;
	parms.bflops				= 0;
	parms.workspace_size		= 0;
	parms.max_inputs			= 0;
	parms.max_outputs			= 0;
	parms.receptive_w			= 1;
	parms.receptive_h			= 1;
	parms.receptive_w_scale		= 1;
	parms.receptive_h_scale		= 1;
	parms.show_receptive_field	= network_section.find_int("show_receptive_field", 0);

	network_section.find_unused_lines();

	// Pre-pass: mark body sections of [recursive_block] layers as consumed so the main loop
	// skips them (they are parsed inline by parse_recursive_block_section).
	// This must happen before the stopbackward scan so that scan uses correct layer counts.
	for (size_t idx = 0; idx < sections.size(); idx ++)
	{
		if (sections.at(idx).type == ELayerType::RECURSIVE_BLOCK)
		{
			const int body_n = sections.at(idx).find_int("body", 1);
			for (int j = 1; j <= body_n; ++j)
			{
				if (idx + j < sections.size())
				{
					sections.at(idx + j).consumed = true;
				}
			}
		}
	}

	// find the last section in which "stopbackward" appears (in layer-index space)
	if (parms.train == 1)
	{
		int lcount = 0;
		for (size_t idx = 0; idx < sections.size(); idx ++)
		{
			if (sections.at(idx).consumed) continue;
			if (sections.at(idx).find_int("stopbackward", 0))
			{
				parms.last_stop_backward = lcount;
			}
			lcount++;
		}
	}

	const auto original_parms_train = parms.train;
	int layer_count = 0;

	for (int idx = 0; idx < (int)sections.size(); idx ++)
	{
		auto & section_ref = sections.at(idx);
		if (section_ref.consumed) continue;  // body section of a recursive_block; already parsed

		parms.train = original_parms_train;
		if (layer_count < parms.last_stop_backward)
		{
			parms.train = 0;
		}

		parms.index = layer_count;  // layer index, not section index

		Darknet::Layer l = {(Darknet::ELayerType)0};

		auto & section = sections.at(idx);

		if (cfg_and_state.is_trace)
		{
			*cfg_and_state.output << "[" << Darknet::to_string(section.type) << "] #" << section.line_number << " START OF LAYER #" << layer_count << std::endl;
		}

		switch (section.type)
		{
			case ELayerType::CONVOLUTIONAL:	{l = parse_convolutional_section(idx);							break;}
			case ELayerType::EML_CONV:		{l = parse_eml_convolutional_section(idx);						break;}
			case ELayerType::DECONVOLUTIONAL:{l = parse_deconvolutional_section(idx);						break;}
			case ELayerType::GRAPH_CONV:	{l = parse_graph_conv_section(idx);							break;}
			case ELayerType::DEFORM_CONV:	{l = parse_deform_conv_section(idx);							break;}
			case ELayerType::DCNV4:			{l = parse_dcnv4_section(idx);									break;}
			case ELayerType::WMHF:			{l = parse_wmhf_section(idx);									break;}
			case ELayerType::YOLOX:			{l = parse_yolox_section(idx);		l.keep_delta_gpu = 1;	break;}
			case ELayerType::PPYOLOE:		{l = parse_ppyoloe_section(idx);	l.keep_delta_gpu = 1;	break;}
			case ELayerType::YOLONAS:		{l = parse_yolonas_section(idx);	l.keep_delta_gpu = 1;	break;}
			case ELayerType::CENTERNET:		{l = parse_centernet_section(idx);	l.keep_delta_gpu = 1;	break;}
			case ELayerType::DETR_DECODER:	{l = parse_detr_decoder_section(idx);	l.keep_delta_gpu = 1;	break;}
			case ELayerType::TRANSFORMER:	{l = parse_transformer_section(idx);							break;}
			case ELayerType::VIT:			{l = parse_vit_section(idx);									break;}
			case ELayerType::MAMBAVISION:	{l = parse_mambavision_section(idx);							break;}
			case ELayerType::TUCKER_ATTENTION: {l = parse_tucker_attention_section(idx);					break;}
			case ELayerType::CLIFFORD:		{l = parse_clifford_section(idx);								break;}
			case ELayerType::RECURSIVE_BLOCK:
			{
				l = parse_recursive_block_section(idx);
				// body sections are already marked consumed in the pre-pass; no idx advance needed
				break;
			}
			case ELayerType::MAXPOOL:		{l = parse_maxpool_section(idx);								break;}
			case ELayerType::UPSAMPLE:		{l = parse_upsample_section(idx);								break;}
			case ELayerType::CHANNEL_SLICE:
			{
				l = parse_channel_slice_section(idx);
				for (int k = 0; k < l.n; ++k)
				{
					net.layers[l.input_layers[k]].use_bin_output = 0;
					net.layers[l.input_layers[k]].keep_delta_gpu = 1;
				}
				break;
			}
			case ELayerType::CHANNEL_SHUFFLE:{l = parse_channel_shuffle_section(idx);						break;}
			case ELayerType::CONNECTED:		{l = parse_connected_section(idx);								break;}
			case ELayerType::CRNN:			{l = parse_crnn_section(idx);									break;}
			case ELayerType::RNN:			{l = parse_rnn_section(idx);									break;}
			case ELayerType::LOCAL_AVGPOOL:	{l = parse_local_avgpool_section(idx);							break;}
			case ELayerType::LSTM:			{l = parse_lstm_section(idx);									break;}
			case ELayerType::REORG:			{l = parse_reorg_section(idx);									break;}
			case ELayerType::AVGPOOL:		{l = parse_avgpool_section(idx);								break;}
			case ELayerType::YOLO:			{l = parse_yolo_section(idx);			l.keep_delta_gpu = 1;	break;}
			case ELayerType::COST:			{l = parse_cost_section(idx);			l.keep_delta_gpu = 1;	break;}
			case ELayerType::REGION:		{l = parse_region_section(idx);			l.keep_delta_gpu = 1;	break;}
			case ELayerType::GAUSSIAN_YOLO:	{l = parse_gaussian_yolo_section(idx);	l.keep_delta_gpu = 1;	break;}
			case ELayerType::CONTRASTIVE:	{l = parse_contrastive_section(idx);	l.keep_delta_gpu = 1;	break;}
			case ELayerType::ROUTE:
			{
				l = parse_route_section(idx);
				for (int k = 0; k < l.n; ++k)
				{
					net.layers[l.input_layers[k]].use_bin_output = 0;
					if (idx >= parms.last_stop_backward)
					{
						net.layers[l.input_layers[k]].keep_delta_gpu = 1;
					}
				}
				break;
			}
			case ELayerType::SHORTCUT:
			{
				l = parse_shortcut_section(idx);
				net.layers[idx - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				if (idx >= parms.last_stop_backward)
				{
					net.layers[l.index].keep_delta_gpu = 1;
				}
				break;
			}
			case ELayerType::SOFTMAX:
			{
				l = parse_softmax_section(idx);
				net.hierarchy = l.softmax_tree;
				l.keep_delta_gpu = 1;
				break;
			}
			case ELayerType::SCALE_CHANNELS:
			{
				l = parse_scale_channels_section(idx);
				net.layers[idx - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				net.layers[l.index].keep_delta_gpu = 1;
				break;
			}
			case ELayerType::SAM:
			{
				l = parse_sam_section(idx);
				net.layers[idx - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				net.layers[l.index].keep_delta_gpu = 1;
				break;
			}
			case ELayerType::DROPOUT:
			{
				l = parse_dropout_section(idx);
				l.output			= net.layers[idx - 1].output;
				l.delta				= net.layers[idx - 1].delta;
				#ifdef DARKNET_GPU
				l.output_gpu		= net.layers[idx - 1].output_gpu;
				l.delta_gpu			= net.layers[idx - 1].delta_gpu;
				l.keep_delta_gpu	= 1;
				#endif
				break;
			}
			default:
			{
				darknet_fatal_error(DARKNET_LOC, "layer type \"%s\" not recognized on line #%ld in \"%s\"", section.name.c_str(), section.line_number, filename.string().c_str());
			}

		} // switch section type

#if defined(DARKNET_GPU) && defined(CUDNN) && defined(CUDNN_HALF) && defined(CUDNN_DATA_BFLOAT16)
		if (net.cudnn_bf16)
		{
			set_layer_cudnn_bf16_mode(l);
		}
#endif

		// calculate receptive field
		if (parms.show_receptive_field)
		{
			int dilation = std::max(1, l.dilation);
			int stride = std::max(1, l.stride);
			int size = std::max(1, l.size);

			if (l.type == Darknet::ELayerType::UPSAMPLE or
				l.type == Darknet::ELayerType::REORG)
			{
				l.receptive_w = parms.receptive_w;
				l.receptive_h = parms.receptive_h;
				l.receptive_w_scale = parms.receptive_w_scale = parms.receptive_w_scale / stride;
				l.receptive_h_scale = parms.receptive_h_scale = parms.receptive_h_scale / stride;
			}
			else
			{
				if (l.type == Darknet::ELayerType::ROUTE)
				{
					parms.receptive_w = parms.receptive_h = parms.receptive_w_scale = parms.receptive_h_scale = 0;
					for (int k = 0; k < l.n; ++k)
					{
						Darknet::Layer & route_l = net.layers[l.input_layers[k]];
						parms.receptive_w = std::max(parms.receptive_w, route_l.receptive_w);
						parms.receptive_h = std::max(parms.receptive_h, route_l.receptive_h);
						parms.receptive_w_scale = std::max(parms.receptive_w_scale, route_l.receptive_w_scale);
						parms.receptive_h_scale = std::max(parms.receptive_h_scale, route_l.receptive_h_scale);
					}
				}
				else
				{
					int increase_receptive = size + (dilation - 1) * 2 - 1;// stride;
					increase_receptive = std::max(0, increase_receptive);

					parms.receptive_w += increase_receptive * parms.receptive_w_scale;
					parms.receptive_h += increase_receptive * parms.receptive_h_scale;
					parms.receptive_w_scale *= stride;
					parms.receptive_h_scale *= stride;
				}

				l.receptive_w = parms.receptive_w;
				l.receptive_h = parms.receptive_h;
				l.receptive_w_scale = parms.receptive_w_scale;
				l.receptive_h_scale = parms.receptive_h_scale;
			}

			*cfg_and_state.output << idx << " - receptive field: " << parms.receptive_w << " x " << parms.receptive_h << std::endl;
		}

#ifdef DARKNET_GPU
		// futher GPU-memory optimization: net.optimized_memory == 2
		l.optimized_memory = net.optimized_memory;
		if (cfg_and_state.gpu_index >= 0 and net.optimized_memory == 1 && parms.train && l.type != Darknet::ELayerType::DROPOUT)
		{
			if (l.delta_gpu)
			{
				cuda_free(l.delta_gpu);
				l.delta_gpu = NULL;
			}
		}
		else if (cfg_and_state.gpu_index >= 0 and net.optimized_memory >= 2 && parms.train && l.type != Darknet::ELayerType::DROPOUT)
		{
			if (l.output_gpu)
			{
				cuda_free(l.output_gpu);
				//l.output_gpu = cuda_make_array_pinned(l.output, l.batch*l.outputs); // l.steps
				l.output_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
			}
			if (l.activation_input_gpu)
			{
				cuda_free(l.activation_input_gpu);
				l.activation_input_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
			}

			if (l.x_gpu)
			{
				cuda_free(l.x_gpu);
				l.x_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
			}

			// maximum optimization
			if (net.optimized_memory >= 3 && l.type != Darknet::ELayerType::DROPOUT)
			{
				if (l.delta_gpu)
				{
					cuda_free(l.delta_gpu);
				}
			}

			if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
			{
				set_specified_workspace_limit(&l, net.workspace_size_limit);   // workspace size limit 1 GB
			}
		}
#endif // DARKNET_GPU

		l.clip					= section.find_float("clip", 0);
		l.dynamic_minibatch		= net.dynamic_minibatch;
		l.onlyforward			= section.find_int("onlyforward", 0);
		l.dont_update			= section.find_int("dont_update", 0);
		l.burnin_update			= section.find_int("burnin_update", 0);
		l.stopbackward			= section.find_int("stopbackward", 0);
		l.train_only_bn			= section.find_int("train_only_bn", 0);
		l.dontload				= section.find_int("dontload", 0);
		l.dontloadscales		= section.find_int("dontloadscales", 0);
		l.learning_rate_scale	= section.find_float("learning_rate", 1);

		section.find_unused_lines();

		if (l.stopbackward == 1)
		{
			*cfg_and_state.output << " ------- previous layers are frozen -------" << std::endl;
		}

		net.layers[layer_count] = l;
		layer_count++;
		if (l.workspace_size > parms.workspace_size)
		{
			parms.workspace_size = l.workspace_size;
		}
		if (l.inputs > parms.max_inputs)
		{
			parms.max_inputs = l.inputs;
		}
		if (l.outputs > parms.max_outputs)
		{
			parms.max_outputs = l.outputs;
		}

		if (l.antialiasing)
		{
			parms.h = l.input_layer->out_h;
			parms.w = l.input_layer->out_w;
			parms.c = l.input_layer->out_c;
			parms.inputs = l.input_layer->outputs;
		}
		else
		{
			parms.h = l.out_h;
			parms.w = l.out_w;
			parms.c = l.out_c;
			parms.inputs = l.outputs;
		}

		if (l.bflops > 0)
		{
			parms.bflops += l.bflops;
		}

		if (l.w > 1 && l.h > 1)
		{
			parms.avg_outputs += l.outputs;
			parms.avg_counter ++;
		}

		if (cfg_and_state.is_verbose)
		{
			if (idx == 0)
			{
				*cfg_and_state.output
					<< "loading network \"" << filename.string() << "\""
					<< " (" << size_to_IEC_string(std::filesystem::file_size(filename)) << ")"
					<< std::endl
					<< "  # line    layer     filters   sz/rte/other    input              output       bflops" << std::endl;
													// "size, stride, dilation, route, anchors, more...
			}

			*cfg_and_state.output << format_layer_summary(idx, section, l) << std::endl;
		}

	} // while loop (sections)

	// Update net.n to the actual number of layers (may be less than sections.size()
	// when [recursive_block] body sections are consumed without adding top-level layers).
	net.n = layer_count;

	if (parms.last_stop_backward > -1)
	{
		for (int k = 0; k < parms.last_stop_backward; ++k)
		{
			Darknet::Layer & l = net.layers[k];
			if (l.keep_delta_gpu)
			{
				if (!l.delta)
				{
					net.layers[k].delta = (float*)xcalloc(l.outputs*l.batch, sizeof(float));
				}
#ifdef DARKNET_GPU
				if (cfg_and_state.gpu_index >= 0 and !l.delta_gpu)
				{
					net.layers[k].delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
				}
#endif
			}

			l.onlyforward = 1;
			l.train = 0;
		}
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0 and net.optimized_memory && parms.train)
	{
		for (int k = 0; k < net.n; ++k)
		{
			Darknet::Layer & l = net.layers[k];
			// delta GPU-memory optimization: net.optimized_memory == 1
			if (!l.keep_delta_gpu)
			{
				const size_t delta_size = l.outputs * l.batch; // l.steps
				if (net.max_delta_gpu_size < delta_size)
				{
					net.max_delta_gpu_size = delta_size;
					if (net.global_delta_gpu) cuda_free(net.global_delta_gpu);
					if (net.state_delta_gpu) cuda_free(net.state_delta_gpu);
					assert(net.max_delta_gpu_size > 0);
					net.global_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
					net.state_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
				}
				if (l.delta_gpu)
				{
					if (net.optimized_memory >= 3)
					{
					}
					else
					{
						cuda_free(l.delta_gpu);
					}
				}
				l.delta_gpu = net.global_delta_gpu;
			}
			else
			{
				if (!l.delta_gpu)
				{
					l.delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
				}
			}

			// maximum optimization
			if (net.optimized_memory >= 3 && l.type != Darknet::ELayerType::DROPOUT)
			{
				if (l.delta_gpu && l.keep_delta_gpu)
				{
					l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
				}
			}
		}
	}
#endif

	set_train_only_bn(net); // set l.train_only_bn for all required layers

	net.outputs = get_network_output_size(net);
	net.output = get_network_output(net);
	parms.avg_outputs = parms.avg_outputs / parms.avg_counter;

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output
			<< "Total BFLOPS = "	<< parms.bflops			<< std::endl
			<< "avg_outputs = "		<< parms.avg_outputs	<< std::endl;
	}
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		get_cuda_stream();
		//get_cuda_memcpy_stream();
		int size = get_network_input_size(net) * net.batch;
		net.input_state_gpu = cuda_make_array(0, size);
		if (cudaSuccess == cudaHostAlloc((void**)&net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped))
		{
			net.input_pinned_cpu_flag = 1;
		}
		else
		{
			std::ignore = cudaGetLastError(); // reset CUDA-error
			net.input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
		}

		// pre-allocate memory for inference on Tensor Cores (fp16)
		*net.max_input16_size = 0;
		*net.max_output16_size = 0;
		if (net.cudnn_half || net.cudnn_bf16)
		{
			*net.max_input16_size = parms.max_inputs;
			CHECK_CUDA(cudaMalloc((void **)net.input16_gpu, *net.max_input16_size * sizeof(short))); //sizeof(half)
			*net.max_output16_size = parms.max_outputs;
			CHECK_CUDA(cudaMalloc((void **)net.output16_gpu, *net.max_output16_size * sizeof(short))); //sizeof(half)
		}

		if (parms.workspace_size)
		{
			*cfg_and_state.output << "Allocating workspace to transfer between CPU and GPU:  " << size_to_IEC_string(parms.workspace_size) << std::endl;

			net.workspace = cuda_make_array(0, parms.workspace_size / sizeof(float) + 1);
		}
		else
		{
			*cfg_and_state.output << "Allocating workspace:  " << size_to_IEC_string(parms.workspace_size) << std::endl;
			net.workspace = (float*)xcalloc(1, parms.workspace_size);
		}
	}
	else if (parms.workspace_size)
	{
		*cfg_and_state.output << "Allocating workspace:  " << size_to_IEC_string(parms.workspace_size) << std::endl;
		net.workspace = (float*)xcalloc(1, parms.workspace_size);
	}
#else
	if (parms.workspace_size)
	{
		*cfg_and_state.output << "Allocating workspace:  " << size_to_IEC_string(parms.workspace_size) << std::endl;
		net.workspace = (float*)xcalloc(1, parms.workspace_size);
	}
#endif

	Darknet::ELayerType lt = net.layers[net.n - 1].type;
	if (lt == Darknet::ELayerType::YOLO    ||
		lt == Darknet::ELayerType::YOLOX   ||
		lt == Darknet::ELayerType::PPYOLOE ||
		lt == Darknet::ELayerType::YOLONAS ||
		lt == Darknet::ELayerType::CENTERNET ||
		lt == Darknet::ELayerType::REGION)
	{
		if (net.w % 32 != 0 ||
			net.h % 32 != 0 ||
			net.w < 32      ||
			net.h < 32      )
		{
			darknet_fatal_error(DARKNET_LOC, "width=%d and height=%d in cfg file must be divisible by 32 for YOLO networks", net.w, net.h);
		}
	}

	Darknet::assign_default_class_colours(&net);

	return net;
}


Darknet::CfgFile & Darknet::CfgFile::parse_net_section()
{
	TAT(TATPARMS);

	auto & s = network_section;

	net.max_batches = s.find_int("max_batches", 0);
	net.batch = s.find_int("batch",1);
	net.learning_rate = s.find_float("learning_rate", .001);
	net.learning_rate_min = s.find_float("learning_rate_min", .00001);
	net.batches_per_cycle = s.find_int("sgdr_cycle", net.max_batches);
	net.batches_cycle_mult = s.find_int("sgdr_mult", 2);
	net.momentum = s.find_float("momentum", .9);
	net.decay = s.find_float("decay", .0001);
	const int subdivs = s.find_int("subdivisions",1);
	net.time_steps = s.find_int("time_steps",1);
	net.track = s.find_int("track", 0);
	net.augment_speed = s.find_int("augment_speed", 2);
	net.init_sequential_subdivisions = net.sequential_subdivisions = s.find_int("sequential_subdivisions", subdivs);
	if (net.sequential_subdivisions > subdivs)
	{
		net.init_sequential_subdivisions = net.sequential_subdivisions = subdivs;
	}
	const bool has_deformable_sampling_layer = std::any_of(
		sections.begin(),
		sections.end(),
		[](const auto & section)
		{
			return section.type == ELayerType::DEFORM_CONV ||
				section.type == ELayerType::DCNV4;
		});
	net.try_fix_nan = s.find_int("try_fix_nan", has_deformable_sampling_layer ? 1 : 0);
	if (has_deformable_sampling_layer && !s.exists("try_fix_nan"))
	{
		Darknet::display_warning_msg("try_fix_nan was not set; enabling it automatically for deformable/snake convolution training.\n");
	}
	net.batch /= subdivs;          // mini_batch
	const int mini_batch = net.batch;
	net.batch *= net.time_steps;  // mini_batch * time_steps
	net.subdivisions = subdivs;    // number of mini_batches

	net.weights_reject_freq = s.find_int("weights_reject_freq", 0);
	net.equidistant_point = s.find_int("equidistant_point", 0);
	net.badlabels_rejection_percentage = s.find_float("badlabels_rejection_percentage", 0);
	net.num_sigmas_reject_badlabels = s.find_float("num_sigmas_reject_badlabels", 0);
	net.ema_alpha = s.find_float("ema_alpha", 0);
	*net.badlabels_reject_threshold = 0;
	*net.delta_rolling_max = 0;
	*net.delta_rolling_avg = 0;
	*net.delta_rolling_std = 0;
	*net.seen = 0;
		*net.cur_iteration = 0;
		*net.cuda_graph_ready = 0;
	net.use_cuda_graph = s.find_int("use_cuda_graph", 0);
	// A command-line precision choice is an explicit request for this process,
	// so it must take precedence over precision keys embedded in a reusable
	// model cfg.  This is especially important for precisionbenchmark: each
	// BF16/FP8/FP4 run must measure the requested mode, not the cfg's default.
	const bool cli_bf16 = cfg_and_state.is_set("bf16");
	const bool cli_fp4 = cfg_and_state.is_set("fp4");
	const bool cli_fp8 = cfg_and_state.is_set("fp8");
	const bool cli_tf32 = cfg_and_state.is_set("tf32");
	const bool cli_precision_selected = cli_bf16 || cli_fp4 || cli_fp8 || cli_tf32;
	const bool want_cudnn_bf16 = cli_bf16 ||
		(!cli_precision_selected && s.find_int("cudnn_bf16", 0) != 0);
	// FP8 is strictly opt-in (default is the FP16/cudnn_half path).  Enable it with
	// "fp8=1" in the [net] section or the "-fp8" command-line parameter: a training run
	// then uses FP8 training GEMMs, an inference run uses calibrated FP8 inference.
	// The explicit "fp8_training=..." and "cudnn_fp8=..." keys still override.
	// TF32 is an explicit precision mode ("tf32=1" in [net] or the "-tf32" command-line
	// parameter): FP32 data with TF32 tensor-core math, forcing the FP16/BF16/FP8 paths off.
	const bool tf32_requested = cli_tf32 ||
		(!cli_precision_selected && s.find_int("tf32", 0) != 0);
	const bool fp4_requested = !tf32_requested && (cli_fp4 ||
		(!cli_precision_selected && s.find_int("fp4", 0) != 0));
	// Until the Blackwell execution hook accepts a layer, an FP4 request deliberately
	// prepares the existing FP8 path as its first fallback.  On Ada this is the selected
	// path for every convolution; on older GPUs the normal cuDNN path remains enabled.
	const bool fp8_requested = !tf32_requested && (fp4_requested || cli_fp8 ||
		(!cli_precision_selected && s.find_int("fp8", 0) != 0));
	const bool want_cudnn_fp8 = !tf32_requested &&
		((!cli_precision_selected && s.find_int("cudnn_fp8", 0) != 0) || (fp8_requested && !parms.train));
	const bool want_fp8_training = !tf32_requested &&
		((!cli_precision_selected && s.find_int("fp8_training", 0) != 0) || (fp8_requested && parms.train));
	net.fp8_warmup_iters = std::max(0, s.find_int("fp8_warmup_iters", net.fp8_warmup_iters));
	const auto fp8_skip_layers = s.find_int_array("fp8_skip_layers");
	if (!fp8_skip_layers.empty())
	{
		free(net.fp8_skip_layers);
		net.fp8_skip_layer_count = static_cast<int>(fp8_skip_layers.size());
		net.fp8_skip_layers = static_cast<int *>(xcalloc(net.fp8_skip_layer_count, sizeof(int)));
		for (int i = 0; i < net.fp8_skip_layer_count; ++i)
		{
			net.fp8_skip_layers[i] = fp8_skip_layers[i];
		}
	}
	Darknet::configure_weight_init_rng(s.find_int("random_init_weights", 1) != 0);
	net.loss_scale = s.find_float("loss_scale", 1);
	net.dynamic_minibatch = s.find_int("dynamic_minibatch", 0);
	net.validation_threads = std::clamp(s.find_int("thread_val", 1), 1, 8);  // 1-8 threads for validation prediction
	net.optimized_memory = s.find_int("optimized_memory", 0);
	net.workspace_size_limit = (size_t)1024*1024 * s.find_float("workspace_size_limit_MB", 1024);  // 1024 MB by default

	net.adam = s.find_int("adam", 0);
	if (net.adam)
	{
		net.B1 = s.find_float("B1", .9);
		net.B2 = s.find_float("B2", .999);
		net.eps = s.find_float("eps", .000001);
	}

	net.h = s.find_int("height",0);
	net.w = s.find_int("width",0);
	net.c = s.find_int("channels",0);
	net.inputs = s.find_int("inputs", net.h * net.w * net.c);
//	net.max_crop = s.find_int("max_crop",net.w * 2);
//	net.min_crop = s.find_int("min_crop",net.w);
	net.flip = s.find_int("flip", 1);
	net.blur = s.find_int("blur", 0);
	net.gaussian_noise = s.find_int("gaussian_noise", 0);
	net.fog = s.find_int("fog", 0);
	net.mixup = s.find_int("mixup", 0);
	int cutmix = s.find_int("cutmix", 0);
	int mosaic = s.find_int("mosaic", 0);
	if (mosaic && cutmix)
	{
		net.mixup = 4;
	}
	else if (cutmix)
	{
		net.mixup = 2;
	}
	else if (mosaic)
	{
		net.mixup = 3;
	}
	net.letter_box = s.find_int("letter_box", 0);
	net.mosaic_bound = s.find_int("mosaic_bound", 0);
	net.contrastive = s.find_int("contrastive", 0);
	net.contrastive_jit_flip = s.find_int("contrastive_jit_flip", 0);
	net.contrastive_color = s.find_int("contrastive_color", 0);
//	net.unsupervised = s.find_int("unsupervised", 0);
	if (net.contrastive && mini_batch < 2)
	{
		darknet_fatal_error(DARKNET_LOC, "mini_batch size (batch/subdivisions) should be higher than 1 for contrastive loss");
	}

	net.label_smooth_eps = s.find_float("label_smooth_eps", 0.0f);
	net.resize_step = s.find_float("resize_step", 32);
	net.attention = s.find_int("attention", 0);
	net.adversarial_lr = s.find_float("adversarial_lr", 0);
	net.max_chart_loss = s.find_float("max_chart_loss", 20.0);

	net.angle = s.find_float("angle", 0);
	net.aspect = s.find_float("aspect", 1);
	net.saturation = s.find_float("saturation", 1);
	net.exposure = s.find_float("exposure", 1);
	net.hue = s.find_float("hue", 0);
	net.power = s.find_float("power", 4);

	if (!net.inputs && !(net.h && net.w && net.c))
	{
		darknet_fatal_error(DARKNET_LOC, "no input parameters supplied");
	}

	net.policy = static_cast<learning_rate_policy>(Darknet::get_learning_rate_policy_from_name(s.find_str("policy", "constant")));

	net.burn_in = s.find_int("burn_in", 0);

#ifdef DARKNET_GPU
	bool cudnn_bf16_handled = false;
	bool fp4_handled = false;
	bool cudnn_fp8_handled = false;
	bool fp8_training_handled = false;
	bool tf32_handled = false;
	if (net.gpu_index >= 0)
	{
		char device_name[1024];
		int compute_capability = get_gpu_compute_capability(net.gpu_index, device_name);
#ifdef CUDNN_HALF
		if (compute_capability >= 700)
		{
			net.cudnn_half = 1;
		}
		else
		{
			net.cudnn_half = 0;
		}
#endif // CUDNN_HALF
#if defined(CUDNN_HALF) && defined(CUDNN_DATA_BFLOAT16)
		if (want_cudnn_bf16)
		{
			cudnn_bf16_handled = true;
			if (compute_capability >= 800)
			{
				net.cudnn_bf16 = 1;
				net.cudnn_half = 0;
			}
			else
			{
				Darknet::display_warning_msg("cudnn_bf16 requires an Ampere-or-newer GPU (SM80+) and will be ignored.\n");
			}
		}
#else
		if (want_cudnn_bf16)
		{
			cudnn_bf16_handled = true;
			Darknet::display_warning_msg("cudnn_bf16 requires a CUDA+cuDNN build with BF16 descriptors and will be ignored.\n");
		}
#endif
		if (want_cudnn_fp8)
		{
			cudnn_fp8_handled = true;
#if defined(DARKNET_GPU_CUDA) && defined(DARKNET_HAS_FP8)
			if (compute_capability >= 890 && Darknet::fp8_gemm_supported())
			{
				net.fp8_inference = 1;
				net.cudnn_half = 0;
				// FP8 is selective.  Keep cuDNN BF16 descriptors available for
				// unsupported layers and for an activation edge whose loaded scale
				// fails the first-frame saturation guard.
#if defined(CUDNN_HALF) && defined(CUDNN_DATA_BFLOAT16)
				net.cudnn_bf16 = 1;
#else
				net.cudnn_bf16 = 0;
#endif
			}
			else
			{
				Darknet::display_warning_msg("cudnn_fp8 requires Ada-or-newer NVIDIA GPU support (SM89+) and CUDA FP8 cuBLASLt; using the existing inference path.\n");
			}
#else
			Darknet::display_warning_msg("cudnn_fp8 requires a CUDA build with cuBLASLt FP8 support and will be ignored.\n");
#endif
		}
		if (fp4_requested)
		{
			fp4_handled = true;
#if defined(DARKNET_GPU_CUDA) && defined(DARKNET_HAS_FP4)
			if (Darknet::fp4_runtime_supported())
			{
				net.fp4_training = parms.train ? 1 : 0;
				net.fp4_inference = parms.train ? 0 : 1;
				// An FP4 model inevitably has a few unsupported shapes.  Retain
				// cuDNN BF16 Tensor Cores as the fallback underneath FP8/FP4 rather
				// than silently sending those layers to FP32 after the FP8 request
				// cleared the default half mode.
#if defined(CUDNN_HALF) && defined(CUDNN_DATA_BFLOAT16)
				if (compute_capability >= 800)
				{
					net.cudnn_bf16 = 1;
					net.cudnn_half = 0;
				}
#endif
			}
			else
			{
				Darknet::display_warning_msg("fp4 requires a supported Blackwell GPU, CUDA 13.2+, cuDNN 9.24+, and a cuDNN Frontend plan; using the FP8 convolution fallback when available.\n");
			}
#else
			Darknet::display_warning_msg("fp4 is not present in this CUDA target; using the FP8 convolution fallback when available.\n");
#endif
		}
		if (want_fp8_training)
		{
			fp8_training_handled = true;
#if defined(DARKNET_GPU_CUDA) && defined(DARKNET_HAS_FP8)
			if (compute_capability >= 890 && Darknet::fp8_gemm_supported())
			{
				net.fp8_training = 1;
				// keep cudnn_half enabled: the FP8 forward/backward hooks intercept eligible
				// layers before the FP16 path runs, and every layer FP8 does not cover
				// (first conv, detection-head feeders, skipped layers, warmup iterations)
				// would otherwise fall back to slow FP32 convolutions
			}
			else
			{
				Darknet::display_warning_msg("fp8_training requires Ada-or-newer NVIDIA GPU support (SM89+) and CUDA FP8 cuBLASLt; using the existing training path.\n");
			}
#else
			Darknet::display_warning_msg("fp8_training requires a CUDA build with cuBLASLt FP8 support and will be ignored.\n");
#endif
		}
		if (tf32_requested)
		{
			tf32_handled = true;
			if (compute_capability >= 800)
			{
				net.tf32 = 1;
				net.cudnn_half = 0;
				net.cudnn_bf16 = 0;
				net.fp4_inference = 0;
				net.fp4_training = 0;
				net.fp8_inference = 0;
				net.fp8_training = 0;
				gemm_set_tf32(true);
				*cfg_and_state.output << "TF32 tensor-core mode enabled (FP32 data, TF32 math)." << std::endl;
			}
			else
			{
				Darknet::display_warning_msg("tf32 requires an Ampere-or-newer GPU (SM80+) and will be ignored.\n");
			}
		}
		*cfg_and_state.output
			<< net.gpu_index
			<< ": compute_capability=" << compute_capability
			<< ", cudnn_half=" << net.cudnn_half
			<< ", cudnn_bf16=" << net.cudnn_bf16
			<< ", fp4_inference=" << net.fp4_inference
			<< ", fp4_training=" << net.fp4_training
			<< ", fp8_inference=" << net.fp8_inference
			<< ", fp8_training=" << net.fp8_training
			<< ", tf32=" << net.tf32
			<< ", GPU=" << device_name
			<< std::endl;
	}
	else
	{
		*cfg_and_state.output << "GPU not used" << std::endl;
	}
	if (want_cudnn_bf16 && !cudnn_bf16_handled)
	{
		Darknet::display_warning_msg("cudnn_bf16 requires CUDA with cuDNN enabled and will be ignored.\n");
	}
	if (fp4_requested && !fp4_handled)
	{
		Darknet::display_warning_msg("fp4 requires CUDA GPU execution; using lower-precision fallbacks when available.\n");
	}
	if (want_cudnn_fp8 && !cudnn_fp8_handled)
	{
		Darknet::display_warning_msg("cudnn_fp8 requires CUDA GPU inference and will be ignored.\n");
	}
	if (want_fp8_training && !fp8_training_handled)
	{
		Darknet::display_warning_msg("fp8_training requires CUDA GPU training and will be ignored.\n");
	}
	if (tf32_requested && !tf32_handled)
	{
		Darknet::display_warning_msg("tf32 requires a CUDA GPU and will be ignored.\n");
	}
#else
	if (fp4_requested)
	{
		Darknet::display_warning_msg("fp4 requires a CUDA GPU build and will be ignored.\n");
	}
	if (want_cudnn_bf16)
	{
		Darknet::display_warning_msg("cudnn_bf16 requires CUDA with cuDNN enabled and will be ignored.\n");
	}
	if (want_cudnn_fp8)
	{
		Darknet::display_warning_msg("cudnn_fp8 requires CUDA GPU inference and will be ignored.\n");
	}
	if (want_fp8_training)
	{
		Darknet::display_warning_msg("fp8_training requires CUDA GPU training and will be ignored.\n");
	}
	if (tf32_requested)
	{
		Darknet::display_warning_msg("tf32 requires a CUDA GPU and will be ignored.\n");
	}
#endif // DARKNET_GPU

	if (net.policy == STEP)
	{
		net.step = s.find_int("step", 1);
		net.scale = s.find_float("scale", 1);
	}
	else if (net.policy == STEPS || net.policy == SGDR)
	{
		auto steps		= s.find_int_array("steps");
		auto scales		= s.find_float_array("scales");
		auto seq_scales	= s.find_float_array("seq_scales");

		if (net.policy == STEPS && (steps.empty() || scales.empty()))
		{
			darknet_fatal_error(DARKNET_LOC, "STEPS policy must have steps and scales in cfg file");
		}

		// make sure all arrays are the same size
		int n = steps.size();
		scales.resize(n);
		seq_scales.resize(n, 1.0f);

		net.num_steps	= n;
		net.steps		= (int*)xcalloc(n, sizeof(int));
		net.scales		= (float*)xcalloc(n, sizeof(float));
		net.seq_scales	= (float*)xcalloc(n, sizeof(float));

		for (auto i = 0; i < steps.size(); i ++)
		{
			net.steps[i]		= static_cast<int>(steps[i]);
			net.scales[i]		= scales[i];
			net.seq_scales[i]	= seq_scales[i];
		}
	}
	else if (net.policy == EXP)
	{
		net.gamma = s.find_float("gamma", 1);
	}
	else if (net.policy == SIG)
	{
		net.gamma = s.find_float("gamma", 1);
		net.step = s.find_int("step", 1);
	}
	else if (net.policy == POLY || net.policy == RANDOM)
	{
		//net.power = s.find_float("power", 1);
	}

	return *this;
}


Darknet::Layer Darknet::CfgFile::parse_convolutional_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int n = s.find_int("filters", 1);
	int groups = s.find_int("groups", 1);
	int size = s.find_int("size", 1);
	int stride = -1;
	//int stride = s.find_int("stride",1);
	int stride_x = s.find_int("stride_x", -1);
	int stride_y = s.find_int("stride_y", -1);
	if (stride_x < 1 || stride_y < 1)
	{
		stride = s.find_int("stride", 1);
		if (stride_x < 1) stride_x = stride;
		if (stride_y < 1) stride_y = stride;
	}
	else
	{
		stride = s.find_int("stride", 1);
	}
	int dilation = s.find_int("dilation", 1);
	int antialiasing = s.find_int("antialiasing", 0);
	if (size == 1)
	{
		dilation = 1;
	}
	int pad = s.find_int("pad", 0);
	int padding = s.find_int("padding", 0);
	if (pad)
	{
		padding = size / 2;
	}

	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "logistic")));

	int assisted_excitation = s.find_float("assisted_excitation", 0);

	int share_index = s.find_int("share_index", -1000000000);
	Darknet::Layer *share_layer = nullptr;
	if (share_index >= 0)
	{
		share_layer = &net.layers[share_index];
	}
	else if (share_index != -1000000000)
	{
		share_layer = &net.layers[parms.index + share_index];
	}

	int h = parms.h;
	int w = parms.w;
	int c = parms.c;
	int batch = parms.batch;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before convolutional layer must output image");
	}
	int batch_normalize = s.find_int("batch_normalize", 0);
	int cbn = s.find_int("cbn", 0);
	if (cbn)
	{
		batch_normalize = 2;
	}
	int binary = s.find_int("binary", 0);
	int xnor = s.find_int("xnor", 0);
	int use_bin_output = s.find_int("bin_output", 0);
	int sway = s.find_int("sway", 0);
	int rotate = s.find_int("rotate", 0);
	int stretch = s.find_int("stretch", 0);
	int stretch_sway = s.find_int("stretch_sway", 0);

	if ((sway + rotate + stretch + stretch_sway) > 1)
	{
		darknet_fatal_error(DARKNET_LOC, "[convolutional] layer can only set one of sway=1, rotate=1, or stretch=1");
	}
	int deform = sway || rotate || stretch || stretch_sway;
	if (deform && size == 1)
	{
		darknet_fatal_error(DARKNET_LOC, "[convolutional] layer sway, rotate, or stretch must only be used with size >= 3");
	}

	Darknet::Layer l = make_convolutional_layer(batch, 1, h, w, c, n, groups, size, stride_x, stride_y, dilation, padding, activation, batch_normalize, binary, xnor, net.adam, use_bin_output, parms.index, antialiasing, share_layer, assisted_excitation, deform, parms.train);

	l.flipped = s.find_int("flipped", 0);
	l.dot = s.find_float("dot", 0);
	l.sway = sway;
	l.rotate = rotate;
	l.stretch = stretch;
	l.stretch_sway = stretch_sway;
	l.angle = s.find_float("angle", 15);
	l.grad_centr = s.find_int("grad_centr", 0);
	l.reverse = s.find_float("reverse", 0);
	l.coordconv = s.find_int("coordconv", 0);

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	if (net.adam)
	{
		l.B1 = net.B1;
		l.B2 = net.B2;
		l.eps = net.eps;
	}

	return l;
}

Darknet::Layer Darknet::CfgFile::parse_eml_convolutional_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int n = s.find_int("filters", 1);
	int groups = s.find_int("groups", 1);
	if (groups < 1) groups = 1;
	const int size = s.find_int("size", 1);
	int stride = -1;
	int stride_x = s.find_int("stride_x", -1);
	int stride_y = s.find_int("stride_y", -1);
	if (stride_x < 1 || stride_y < 1)
	{
		stride = s.find_int("stride", 1);
		if (stride_x < 1) stride_x = stride;
		if (stride_y < 1) stride_y = stride;
	}
	const int dilation = size == 1 ? 1 : s.find_int("dilation", 1);
	const int pad = s.find_int("pad", 0);
	int padding = s.find_int("padding", 0);
	if (pad)
	{
		padding = size / 2;
	}

	if (!(parms.h && parms.w && parms.c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before [eml_convolutional] on line #%ld must output image", s.line_number);
	}

	const int batch_normalize = s.find_int("batch_normalize", 1);
	const float eml_clamp = s.find_float("eml_clamp", 4.0f);
	const float eml_eps = s.find_float("eml_eps", 0.000001f);
	const float eml_scale = s.find_float("eml_scale", 0.1f);
	const int residual = s.find_int("residual", 1);

	Darknet::Layer l = make_eml_convolutional_layer(
		parms.batch,
		parms.h,
		parms.w,
		parms.c,
		n,
		groups,
		size,
		stride_x,
		stride_y,
		dilation,
		padding,
		batch_normalize,
		net.adam,
		parms.index,
		parms.train,
		eml_clamp,
		eml_eps,
		eml_scale,
		residual);

	if (l.shortcut && l.inputs != l.outputs)
	{
		darknet_fatal_error(DARKNET_LOC, "[eml_convolutional] residual=1 on line #%ld requires input and output tensor sizes to match", s.line_number);
	}

	if (net.adam)
	{
		l.input_layer->B1 = net.B1;
		l.input_layer->B2 = net.B2;
		l.input_layer->eps = net.eps;
		l.self_layer->B1 = net.B1;
		l.self_layer->B2 = net.B2;
		l.self_layer->eps = net.eps;
	}

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_deconvolutional_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int n = s.find_int("filters", 1);
	const int size = s.find_int("size", 1);
	const int stride = s.find_int("stride", 1);
	const int pad = s.find_int("pad", 0);
	const int padding = s.find_int("padding", 0);

	if (pad || padding)
	{
		darknet_fatal_error(DARKNET_LOC, "[deconvolutional] layer on line #%ld does not support pad/padding in this port", s.line_number);
	}

	const ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "logistic")));

	if (!(parms.h && parms.w && parms.c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before [deconvolutional] on line #%ld must output image", s.line_number);
	}

	Darknet::Layer l = make_deconvolutional_layer(parms.batch, parms.h, parms.w, parms.c, n, size, stride, activation);
	return l;
}


Darknet::Layer Darknet::CfgFile::parse_route_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const auto v = s.find_int_array("layers");
	if (v.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "route layer at line #%ld must specify input layers", s.line_number);
	}

	int * layers = (int*)xcalloc(v.size(), sizeof(int));
	int * sizes = (int*)xcalloc(v.size(), sizeof(int));

	for (size_t idx = 0; idx < v.size(); idx ++)
	{
		int route_index = v[idx];
		if (route_index < 0)
		{
			route_index = parms.index + route_index;
		}

		if (route_index >= parms.index)
		{
			darknet_fatal_error(DARKNET_LOC, "cannot route layer #%d in [%s] at line #%ld", route_index, s.name.c_str(), s.line_number);
		}

		layers[idx] = route_index;
		sizes[idx] = net.layers[route_index].outputs;
	}

	int batch = parms.batch;

	int groups = s.find_int("groups", 1);
	int group_id = s.find_int("group_id", 0);

	Darknet::Layer l = make_route_layer(batch, v.size(), layers, sizes, groups, group_id);

	const Darknet::Layer & first = net.layers[layers[0]];
	l.out_w = first.out_w;
	l.out_h = first.out_h;
	l.out_c = first.out_c;

	for (int i = 1; i < v.size(); ++i)
	{
		int index = layers[i];
		const Darknet::Layer & next = net.layers[index];
		if (next.out_w == first.out_w && next.out_h == first.out_h)
		{
			l.out_c += next.out_c;
		}
		else
		{
			display_warning_msg("Line #" + std::to_string(s.line_number) + ":  the width and height of the input layers are different.\n");
			l.out_h = l.out_w = l.out_c = 0;
		}
	}

	l.out_c = l.out_c / l.groups;

	l.w = first.w;
	l.h = first.h;
	l.c = l.out_c;

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_channel_slice_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int from = s.find_int("from");
	int index = from;
	if (index < 0)
	{
		index = parms.index + index;
	}
	if (index < 0 || index >= parms.index)
	{
		darknet_fatal_error(DARKNET_LOC, "[channel_slice] layer on line #%ld has invalid from=%d", s.line_number, from);
	}

	const int begin_slice_point = s.find_int("start", 2);
	const int end_slice_point = s.find_int("end", 2);
	const int axis = s.find_int("axis", 1);

	int *layers = (int*)xcalloc(1, sizeof(int));
	int *sizes = (int*)xcalloc(1, sizeof(int));
	layers[0] = index;
	sizes[0] = net.layers[index].outputs;

	const Darknet::Layer & source_layer = net.layers[index];

	return make_channel_slice_layer(
		parms.batch,
		source_layer.out_w,
		source_layer.out_h,
		source_layer.out_c,
		begin_slice_point,
		end_slice_point,
		axis,
		1,
		layers,
		sizes);
}


Darknet::Layer Darknet::CfgFile::parse_channel_shuffle_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	if (!(parms.h && parms.w && parms.c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before [channel_shuffle] on line #%ld must output image", s.line_number);
	}

	const int groups = s.find_int("groups", 2);
	return make_channel_shuffle_layer(parms.batch, parms.w, parms.h, parms.c, groups);
}


Darknet::Layer Darknet::CfgFile::parse_maxpool_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int stride = s.find_int("stride",1);
	int stride_x = s.find_int("stride_x", stride);
	int stride_y = s.find_int("stride_y", stride);
	int size = s.find_int("size",stride);
	int padding = s.find_int("padding", size-1);
	int maxpool_depth = s.find_int("maxpool_depth", 0);
	int out_channels = s.find_int("out_channels", 1);
	int antialiasing = s.find_int("antialiasing", 0);
	const int avgpool = 0;

	int h = parms.h;
	int w = parms.w;
	int c = parms.c;
	int batch = parms.batch;

	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before [maxpool] at line #%ld must output image", s.line_number);
	}

	Darknet::Layer l = make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, parms.train);
	l.maxpool_zero_nonmax = s.find_int("maxpool_zero_nonmax", 0);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_yolo_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int classes		= s.find_int("classes", 20);
	int total		= s.find_int("num", 1);
	int max_boxes	= s.find_int("max", 200);
	int num			= total;
	const auto v	= s.find_int_array("mask"); // e.g., [0, 1, 2]
	int * mask		= nullptr;

	if (not v.empty())
	{
		mask = (int*)xcalloc(v.size(), sizeof(int));
		for (size_t i = 0; i < v.size(); i ++)
		{
			mask[i] = v[i];
		}
		num = v.size();
	}

	Darknet::Layer l = make_yolo_layer(parms.batch, parms.w, parms.h, num, total, mask, classes, max_boxes);

	if (l.outputs != parms.inputs)
	{
		darknet_fatal_error(DARKNET_LOC, "filters in convolutional layer prior to [%s] on line %ld does not match either the classes or the mask (inputs=%d, outputs=%d)", s.name.c_str(), s.line_number, parms.inputs, l.outputs);
	}

	// The old Darknet had a CLI option called "-show_details".  This was replaced by "--verbose".
	l.show_details = cfg_and_state.is_verbose ? 1 : 0;

	l.max_delta = s.find_float("max_delta", FLT_MAX);   // set 10

	VInt vi = s.find_int_array("counters_per_class");
	l.classes_multipliers = get_classes_multipliers(vi, classes, l.max_delta);

	l.label_smooth_eps = s.find_float("label_smooth_eps", 0.0f);
	l.scale_x_y = s.find_float("scale_x_y", 1);
	l.objectness_smooth = s.find_int("objectness_smooth", 0);
	l.new_coords = s.find_int("new_coords", 0);
	l.iou_normalizer = s.find_float("iou_normalizer", 0.75);
	l.obj_normalizer = s.find_float("obj_normalizer", 1);
	l.cls_normalizer = s.find_float("cls_normalizer", 1);
	l.delta_normalizer = s.find_float("delta_normalizer", 1);

	const std::string iou_loss = s.find_str("iou_loss", "mse");
	l.iou_loss = static_cast<IOU_LOSS>(get_IoU_loss_from_name(iou_loss)); // "iou"

	const std::string iou_thresh_kind = s.find_str("iou_thresh_kind", "iou");
	l.iou_thresh_kind = static_cast<IOU_LOSS>(get_IoU_loss_from_name(iou_thresh_kind));

	l.beta_nms = s.find_float("beta_nms", 0.6);

	const std::string nms_kind = s.find_str("nms_kind", "default");
	l.nms_kind = static_cast<NMS_KIND>(get_NMS_kind_from_name(nms_kind));

	l.jitter = s.find_float("jitter", .2);
	l.resize = s.find_float("resize", 1.0);
	l.focal_loss = s.find_int("focal_loss", 0);

	l.ignore_thresh = s.find_float("ignore_thresh", .5);
	l.truth_thresh = s.find_float("truth_thresh", 1);
	l.iou_thresh = s.find_float("iou_thresh", 1); // recommended to use iou_thresh=0.213 in [yolo]
	l.random = s.find_float("random", 0);

	l.track_history_size = s.find_int("track_history_size", 5);
	l.sim_thresh = s.find_float("sim_thresh", 0.8);
	l.dets_for_track = s.find_int("dets_for_track", 1);
	l.dets_for_show = s.find_int("dets_for_show", 1);
	l.track_ciou_norm = s.find_float("track_ciou_norm", 0.01);
	int embedding_layer_id = s.find_int("embedding_layer", 999999);
	if (embedding_layer_id < 0) embedding_layer_id = parms.index + embedding_layer_id;
	if (embedding_layer_id != 999999)
	{
		const Darknet::Layer & le = net.layers[embedding_layer_id];
		l.embedding_layer_id = embedding_layer_id;
		l.embedding_output = (float*)xcalloc(le.batch * le.outputs, sizeof(float));
		l.embedding_size = le.n / l.n;
		if (le.n % l.n != 0)
		{
			darknet_fatal_error(DARKNET_LOC, "filters=%d number in embedding_layer=%d isn't divisable by number of anchors %d", le.n, embedding_layer_id, l.n);
		}
	}

	const std::string map_file = s.find_str("map");
	if (not map_file.empty())
	{
		l.map = read_map(map_file.c_str());
	}

	const VFloat vf = s.find_float_array("anchors");
	for (size_t i = 0; i < vf.size(); i ++)
	{
		l.biases[i] = vf[i];
	}

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_upsample_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int stride = s.find_int("stride", 2);

	Darknet::Layer l = make_upsample_layer(parms.batch, parms.w, parms.h, parms.c, stride);

	l.scale = s.find_float("scale", 1);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_shortcut_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	auto str = s.find_str("activation", "linear");
	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(str));

	str = s.find_str("weights_type", "none");
	WEIGHTS_TYPE_T weights_type = static_cast<WEIGHTS_TYPE_T>(get_weights_type_from_name(str));

	str = s.find_str("weights_normalization", "none");
	WEIGHTS_NORMALIZATION_T weights_normalization = static_cast<WEIGHTS_NORMALIZATION_T>(get_weights_normalization_from_name(str));

	const auto v = s.find_int_array("from");
	const int n = v.size();
	if (v.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "shortcut layer must specify \"from=...\" input layer");
	}

	int * layers				= (int*)xcalloc(n, sizeof(int));
	int * sizes					= (int*)xcalloc(n, sizeof(int));
	float ** layers_output		= (float **)xcalloc(n, sizeof(float *));
	float ** layers_delta		= (float **)xcalloc(n, sizeof(float *));
	float ** layers_output_gpu	= (float **)xcalloc(n, sizeof(float *));
	float ** layers_delta_gpu	= (float **)xcalloc(n, sizeof(float *));

	for (int i = 0; i < n; ++i)
	{
		int index = v[i];

		if (index < 0)
		{
			index = parms.index + index;
		}

		layers[i]			= index;
		sizes[i]			= net.layers[index].outputs;
		layers_output[i]	= net.layers[index].output;
		layers_delta[i]		= net.layers[index].delta;
	}

	#ifdef DARKNET_GPU
	for (int i = 0; i < n; ++i)
	{
		layers_output_gpu[i]	= net.layers[layers[i]].output_gpu;
		layers_delta_gpu[i]		= net.layers[layers[i]].delta_gpu;
	}
	#endif

	Darknet::Layer l = make_shortcut_layer(parms.batch, n, layers, sizes, parms.w, parms.h, parms.c, layers_output, layers_delta, layers_output_gpu, layers_delta_gpu, weights_type, weights_normalization, activation, parms.train);

	free(layers_output_gpu);
	free(layers_delta_gpu);

	for (int i = 0; i < n; ++i)
	{
		int index = layers[i];
		assert(parms.w == net.layers[index].out_w && parms.h == net.layers[index].out_h);

		if (parms.w != net.layers[index].out_w || parms.h != net.layers[index].out_h || parms.c != net.layers[index].out_c)
		{
			*cfg_and_state.output
				<< "("
				<< parms.w << " x " << parms.h << " x " << parms.c
				<< ") + ("
				<< net.layers[index].out_w << " x " << net.layers[index].out_h << " x " << net.layers[index].out_c
				<< ")" << std::endl;
		}
	}

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_connected_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int output = s.find_int("output", 1);
	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "logistic")));
	int batch_normalize = s.find_int("batch_normalize", 0);

	Darknet::Layer l = make_connected_layer(parms.batch, 1, parms.inputs, output, activation, batch_normalize);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_crnn_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int size			= s.find_int("size"				, 3);
	int stride			= s.find_int("stride"			, 1);
	int dilation		= s.find_int("dilation"			, 1);
	int pad				= s.find_int("pad"				, 0);
	int padding			= s.find_int("padding"			, 0);
	int output_filters	= s.find_int("output"			, 1);
	int hidden_filters	= s.find_int("hidden"			, 1);
	int groups			= s.find_int("groups"			, 1);
	int batch_normalize	= s.find_int("batch_normalize"	, 0);
	int xnor			= s.find_int("xnor"				, 0);

	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "logistic")));

	if (pad)
	{
		padding = size / 2;
	}

	Darknet::Layer l = make_crnn_layer(parms.batch, parms.h, parms.w, parms.c, hidden_filters, output_filters, groups, parms.time_steps, size, stride, dilation, padding, activation, batch_normalize, xnor, parms.train);

	l.shortcut = s.find_int("shortcut", 0);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_rnn_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int output			= s.find_int("output"			, 1);
	int hidden			= s.find_int("hidden"			, 1);
	int batch_normalize	= s.find_int("batch_normalize"	, 0);
	int logistic		= s.find_int("logistic"			, 0);

	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "logistic")));

	Darknet::Layer l = make_rnn_layer(parms.batch, parms.inputs, hidden, output, parms.time_steps, activation, batch_normalize, logistic);

	l.shortcut = s.find_int("shortcut", 0);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_local_avgpool_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int stride			= s.find_int("stride"	, 1			);
	int stride_x		= s.find_int("stride_x"	, stride	);
	int stride_y		= s.find_int("stride_y"	, stride	);
	int size			= s.find_int("size"		, stride	);
	int padding			= s.find_int("padding"	, size - 1	);
	int maxpool_depth	= 0;
	int out_channels	= 1;
	int antialiasing	= 0;
	int avgpool			= 1;
	int w				= parms.w;
	int h				= parms.h;
	int c				= parms.c;
	int batch			= parms.batch;

	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before [local_avgpool] on line %ld must output image", s.line_number);
	}

	Darknet::Layer l = make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, parms.train);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_lstm_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int output			= s.find_int("output"			, 1);
	int batch_normalize	= s.find_int("batch_normalize"	, 0);

	Darknet::Layer l = make_lstm_layer(parms.batch, parms.inputs, output, parms.time_steps, batch_normalize);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_reorg_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int stride	= s.find_int("stride"	, 1);
	int reverse	= s.find_int("reverse"	, 0);

	int w		= parms.w;
	int h		= parms.h;
	int c		= parms.c;
	int batch	= parms.batch;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before reorg layer on line %ld must output image", s.line_number);
	}

	Darknet::Layer l = make_reorg_layer(batch, w, h, c, stride, reverse);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_avgpool_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int w		= parms.w;
	int h		= parms.h;
	int c		= parms.c;
	int batch	= parms.batch;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before avgpool layer on line %ld must output image", s.line_number);
	}

	Darknet::Layer l = make_avgpool_layer(batch, w, h, c);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_cost_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	COST_TYPE type = static_cast<COST_TYPE>(get_cost_types_from_name(s.find_str("type", "sse")));

	float scale = s.find_float("scale", 1);

	Darknet::Layer l = make_cost_layer(parms.batch, parms.inputs, type, scale);

	l.ratio = s.find_float("ratio", 0);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_region_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int coords		= s.find_int("coords"	, 4		);
	int classes		= s.find_int("classes"	, 20	);
	int num			= s.find_int("num"		, 1		);
	int max_boxes	= s.find_int("max"		, 200	);

	Darknet::Layer l = make_region_layer(parms.batch, parms.w, parms.h, num, classes, coords, max_boxes);

	if (l.outputs != parms.inputs)
	{
		darknet_fatal_error(DARKNET_LOC, "filters in [convolutional] layer does not match classes or mask in [region] layer on line %ld (%d vs %d)", s.line_number, l.outputs, parms.inputs);
	}

	l.log = s.find_int("log", 0);
	l.sqrt = s.find_int("sqrt", 0);

	l.softmax			= s.find_int("softmax"			, 0);
	l.focal_loss		= s.find_int("focal_loss"		, 0);
//	l.max_boxes			= s.find_int("max"				, 30);
	l.rescore			= s.find_int("rescore"			, 0);
	l.classfix			= s.find_int("classfix"			, 0);
	l.absolute			= s.find_int("absolute"			, 0);
	l.bias_match		= s.find_int("bias_match"		, 0);
	l.jitter			= s.find_float("jitter"			, 0.2f);
	l.resize			= s.find_float("resize"			, 1.0f);
	l.thresh			= s.find_float("thresh"			, 0.5f);
	l.random			= s.find_float("random"			, 0.0f);
	l.coord_scale		= s.find_float("coord_scale"	, 1.0f);
	l.object_scale		= s.find_float("object_scale"	, 1.0f);
	l.noobject_scale	= s.find_float("noobject_scale"	, 1.0f);
	l.mask_scale		= s.find_float("mask_scale"		, 1.0f);
	l.class_scale		= s.find_float("class_scale"	, 1.0f);

	const auto tree_file = s.find_str("tree");
	if (tree_file.empty() == false)
	{
		l.softmax_tree = read_tree(tree_file.c_str());
	}

	const auto map_file = s.find_str("map");
	if (map_file.empty() == false)
	{
		l.map = read_map(map_file.c_str());
	}

	const VFloat vf = s.find_float_array("anchors");
	for (size_t i = 0; i < vf.size() && i < num * 2; i ++)
	{
		l.biases[i] = vf[i];
	}

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_gaussian_yolo_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int classes		= s.find_int("classes", 20);
	int max_boxes	= s.find_int("max", 200);
	int total		= s.find_int("num", 1);
	int num			= total;


	const auto v	= s.find_int_array("mask"); // e.g., [0, 1, 2]
	int * mask		= nullptr;
	if (not v.empty())
	{
		mask = (int*)xcalloc(v.size(), sizeof(int));
		for (size_t i = 0; i < v.size(); i ++)
		{
			mask[i] = v[i];
		}
		num = v.size();
	}

	Darknet::Layer l = make_gaussian_yolo_layer(parms.batch, parms.w, parms.h, num, total, mask, classes, max_boxes);

	if (l.outputs != parms.inputs)
	{
		darknet_fatal_error(DARKNET_LOC, "filters in [convolutional] layer does not match classes or mask in [Gaussian_yolo] layer on line %ld", s.line_number);
	}

	l.objectness_smooth	= s.find_int("objectness_smooth"	, 0		);
	l.label_smooth_eps	= s.find_float("label_smooth_eps"	, 0.0f	);
	l.scale_x_y			= s.find_float("scale_x_y"			, 1.0f	);
	l.uc_normalizer		= s.find_float("uc_normalizer"		, 1.0f	);
	l.iou_normalizer	= s.find_float("iou_normalizer"		, 0.75f	);
	l.obj_normalizer	= s.find_float("obj_normalizer"		, 1.0f	);
	l.cls_normalizer	= s.find_float("cls_normalizer"		, 1.0f	);
	l.delta_normalizer	= s.find_float("delta_normalizer"	, 1.0f	);
	l.max_delta			= s.find_float("max_delta"			, FLT_MAX);   // set 10

	VInt vi = s.find_int_array("counters_per_class");
	l.classes_multipliers = get_classes_multipliers(vi, classes, l.max_delta);

	const std::string iou_loss = s.find_str("iou_loss", "mse");
	l.iou_loss = static_cast<IOU_LOSS>(get_IoU_loss_from_name(iou_loss)); // "iou"

	const std::string iou_thresh_kind = s.find_str("iou_thresh_kind", "iou");
	l.iou_thresh_kind = static_cast<IOU_LOSS>(get_IoU_loss_from_name(iou_thresh_kind));

	l.beta_nms = s.find_float("beta_nms", 0.6f);
	const std::string nms_kind = s.find_str("nms_kind", "default");
	l.nms_kind = static_cast<NMS_KIND>(get_NMS_kind_from_name(nms_kind));

	*cfg_and_state.output << "nms_kind: " << nms_kind << "(" << l.nms_kind << "), beta=" << l.beta_nms << std::endl;

	const std::string yolo_point = s.find_str("yolo_point", "center");
	l.yolo_point = static_cast<YOLO_POINT>(get_yolo_point_types_from_name(yolo_point));

	*cfg_and_state.output
		<< "[Gaussian_yolo] iou loss: " << iou_loss << " (" << l.iou_loss << ")"
		<< ", iou_norm: " << l.iou_normalizer
		<< ", obj_norm: " << l.obj_normalizer
		<< ", cls_norm: " << l.cls_normalizer
		<< ", delta_norm: " << l.delta_normalizer
		<< ", scale: " << l.scale_x_y
		<< ", point: " << l.yolo_point
		<< std::endl;

	l.jitter		= s.find_float("jitter"			, 0.2f);
	l.resize		= s.find_float("resize"			, 1.0f);
	l.ignore_thresh	= s.find_float("ignore_thresh"	, 0.5f);
	l.truth_thresh	= s.find_float("truth_thresh"	, 1.0f);
	l.iou_thresh	= s.find_float("iou_thresh"		, 1.0f); // recommended to use iou_thresh=0.213 in [yolo]
	l.random		= s.find_float("random"			, 0.0f);

	const std::string map_file = s.find_str("map");
	if (not map_file.empty())
	{
		l.map = read_map(map_file.c_str());
	}

	const VFloat vf = s.find_float_array("anchors");
	for (size_t i = 0; i < vf.size(); i ++)
	{
		l.biases[i] = vf[i];
	}

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_contrastive_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int classes = s.find_int("classes", 1000);
	Darknet::Layer *yolo_layer = nullptr;
	int yolo_layer_id = s.find_int("yolo_layer", 0);
	if (yolo_layer_id < 0)
	{
		yolo_layer_id = parms.index + yolo_layer_id;
	}
	if (yolo_layer_id != 0)
	{
		yolo_layer = net.layers + yolo_layer_id;
	}
	if (yolo_layer->type != Darknet::ELayerType::YOLO)
	{
		darknet_fatal_error(DARKNET_LOC, "[contrastive] layer at line %ld does not point to [yolo] layer", s.line_number);
	}

	Darknet::Layer l = make_contrastive_layer(parms.batch, parms.w, parms.h, parms.c, classes, parms.inputs, yolo_layer);

	l.temperature			= s.find_float("temperature"		, 1.0f);
	l.cls_normalizer		= s.find_float("cls_normalizer"		, 1.0f);
	l.max_delta				= s.find_float("max_delta"			, FLT_MAX);   // set 10
	l.contrastive_neg_max	= s.find_int("contrastive_neg_max"	, 3);
	l.steps					= parms.time_steps;

	return l;
}



Darknet::Layer Darknet::CfgFile::parse_softmax_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int groups = s.find_int("groups", 1);

	Darknet::Layer l = make_softmax_layer(parms.batch, parms.inputs, groups);

	l.temperature = s.find_float("temperature", 1);

	const std::string tree_file = s.find_str("tree");
	if (not tree_file.empty())
	{
		l.softmax_tree = read_tree(tree_file.c_str());
	}

	l.w			= parms.w;
	l.h			= parms.h;
	l.c			= parms.c;
	l.spatial	= s.find_float("spatial", 0.0f);
	l.noloss	= s.find_int("noloss", 0);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_scale_channels_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int index = s.find_int("from");
	if (index < 0)
	{
		index = parms.index + index;
	}
	int scale_wh = s.find_int("scale_wh", 0);

	int batch = parms.batch;
	const Darknet::Layer & from = net.layers[index];

	Darknet::Layer l = make_scale_channels_layer(batch, index, parms.w, parms.h, parms.c, from.out_w, from.out_h, from.out_c, scale_wh);

	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "linear")));
	l.activation = activation;
	if (activation == SWISH || activation == MISH)
	{
		darknet_fatal_error(DARKNET_LOC, "[scale_channels] layer on line #%ld does not support SWISH or MISH activation", s.line_number);
	}

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_sam_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int index = s.find_int("from");
	if (index < 0)
	{
		index = parms.index + index;
	}

	int batch = parms.batch;
	const Darknet::Layer & from = net.layers[index];

	Darknet::Layer l = make_sam_layer(batch, index, parms.w, parms.h, parms.c, from.out_w, from.out_h, from.out_c);

	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "linear")));
	l.activation = activation;
	if (activation == SWISH || activation == MISH)
	{
		darknet_fatal_error(DARKNET_LOC, "[sam] layer on line #%ld does not support SWISH or MISH activation", s.line_number);
	}

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_dropout_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int dropblock				= s.find_int(	"dropblock"			, 0		);
	float probability			= s.find_float(	"probability"		, 0.2f	);
	float dropblock_size_rel	= s.find_float(	"dropblock_size_rel", 0.0f	);
	int dropblock_size_abs		= s.find_float(	"dropblock_size_abs", 0.0f	); ///< @todo why read in a float and then store it in an int?

	if (dropblock_size_abs > parms.w || dropblock_size_abs > parms.h)
	{
		*cfg_and_state.output << "[dropout] - dropblock_size_abs=" << dropblock_size_abs << " that is bigger than layer size " << parms.w << " x " << parms.h << std::endl;
		dropblock_size_abs = std::min(parms.w, parms.h);
	}
	if (dropblock && !dropblock_size_rel && !dropblock_size_abs)
	{
		*cfg_and_state.output << "[dropout] - None of the parameters (dropblock_size_rel or dropblock_size_abs) are set, will use dropblock_size_abs=7" << std::endl;
		dropblock_size_abs = 7;
	}
	if (dropblock_size_rel && dropblock_size_abs)
	{
		*cfg_and_state.output << "[dropout] - Both parameters are set, only this parameter will be used: dropblock_size_abs=" << dropblock_size_abs << std::endl;
		dropblock_size_rel = 0;
	}

	Darknet::Layer l = make_dropout_layer(parms.batch, parms.inputs, probability, dropblock, dropblock_size_rel, dropblock_size_abs, parms.w, parms.h, parms.c);

	l.out_w = parms.w;
	l.out_h = parms.h;
	l.out_c = parms.c;

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_deform_conv_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int n = s.find_int("filters", 1);
	int groups = s.find_int("groups", 1);
	int size = s.find_int("size", 1);
	int stride = -1;
	//int stride = s.find_int("stride",1);
	int stride_x = s.find_int("stride_x", -1);
	int stride_y = s.find_int("stride_y", -1);
	if (stride_x < 1 || stride_y < 1)
	{
		stride = s.find_int("stride", 1);
		if (stride_x < 1) stride_x = stride;
		if (stride_y < 1) stride_y = stride;
	}
	else
	{
		stride = s.find_int("stride", 1);
	}
	int dilation = s.find_int("dilation", 1);
	int antialiasing = s.find_int("antialiasing", 0);
	if (size == 1)
	{
		dilation = 1;
	}
	int pad = s.find_int("pad", 0);
	int padding = s.find_int("padding", 0);
	if (pad)
	{
		padding = size / 2;
	}

	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "logistic")));

	int assisted_excitation = s.find_float("assisted_excitation", 0);

	int share_index = s.find_int("share_index", -1000000000);
	Darknet::Layer *share_layer = nullptr;
	if (share_index >= 0)
	{
		share_layer = &net.layers[share_index];
	}
	else if (share_index != -1000000000)
	{
		share_layer = &net.layers[parms.index + share_index];
	}

	int h = parms.h;
	int w = parms.w;
	int c = parms.c;
	int batch = parms.batch;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before deform_conv layer must output image");
	}
	int batch_normalize = s.find_int("batch_normalize", 0);
	int cbn = s.find_int("cbn", 0);
	if (cbn)
	{
		batch_normalize = 2;
	}
	int binary = s.find_int("binary", 0);
	int xnor = s.find_int("xnor", 0);
	int use_bin_output = s.find_int("bin_output", 0);

		int use_mask = s.find_int("use_mask", s.find_int("mask", 1));

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, groups, size, stride_x, stride_y, dilation, padding, activation, batch_normalize, binary, xnor, net.adam, use_bin_output, parms.index, antialiasing, share_layer, assisted_excitation, parms.train, use_mask);

	l.flipped = s.find_int("flipped", 0);
	l.dot = s.find_float("dot", 0);
	l.angle = s.find_float("angle", 15);
	l.grad_centr = s.find_int("grad_centr", 0);
	l.reverse = s.find_float("reverse", 0);
	l.coordconv = s.find_int("coordconv", 0);

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	if (net.adam)
	{
		l.B1 = net.B1;
		l.B2 = net.B2;
		l.eps = net.eps;
	}

	return l;
}

Darknet::Layer Darknet::CfgFile::parse_graph_conv_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int n = s.find_int("filters", 1);
	int groups = s.find_int("groups", 1);
	int size = s.find_int("size", 1);
	int stride = -1;
	int stride_x = s.find_int("stride_x", -1);
	int stride_y = s.find_int("stride_y", -1);
	if (stride_x < 1 || stride_y < 1)
	{
		stride = s.find_int("stride", 1);
		if (stride_x < 1) stride_x = stride;
		if (stride_y < 1) stride_y = stride;
	}
	else
	{
		stride = s.find_int("stride", 1);
	}
	int dilation = s.find_int("dilation", 1);
	int antialiasing = s.find_int("antialiasing", 0);
	if (size == 1)
	{
		dilation = 1;
	}
	int pad = s.find_int("pad", 0);
	int padding = s.find_int("padding", 0);
	if (pad)
	{
		padding = size / 2;
	}

	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "logistic")));

	int assisted_excitation = s.find_float("assisted_excitation", 0);

	int share_index = s.find_int("share_index", -1000000000);
	Darknet::Layer *share_layer = nullptr;
	if (share_index >= 0)
	{
		share_layer = &net.layers[share_index];
	}
	else if (share_index != -1000000000)
	{
		share_layer = &net.layers[parms.index + share_index];
	}

	int h = parms.h;
	int w = parms.w;
	int c = parms.c;
	int batch = parms.batch;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before graph_conv layer must output image");
	}

	int batch_normalize = s.find_int("batch_normalize", 0);
	int cbn = s.find_int("cbn", 0);
	if (cbn)
	{
		batch_normalize = 2;
	}
	int binary = s.find_int("binary", 0);
	int xnor = s.find_int("xnor", 0);
	int use_bin_output = s.find_int("bin_output", 0);

	int graph_edge_mode = s.find_int("graph_edge_mode", s.find_int("edge_mode", 1));
	int graph_use_self = s.find_int("graph_use_self", s.find_int("use_self", 1));
	int graph_valid_mask_zero = s.find_int("graph_valid_mask_zero", s.find_int("valid_mask_zero", 1));

	Darknet::Layer l = make_graph_conv_layer(batch, 1, h, w, c, n, groups, size, stride_x, stride_y,
		dilation, padding, activation, batch_normalize, binary, xnor, net.adam, use_bin_output,
		parms.index, antialiasing, share_layer, assisted_excitation, parms.train, graph_edge_mode,
		graph_use_self, graph_valid_mask_zero);

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	return l;
}

Darknet::Layer Darknet::CfgFile::parse_vit_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const bool has_filters = s.exists("filters");
	const bool has_dim = s.exists("dim");
	int filters = 128;
	if (has_filters)
	{
		filters = s.find_int("filters");
	}
	if (has_dim)
	{
		const int dim = s.find_int("dim");
		if (has_filters && dim != filters)
		{
			darknet_fatal_error(DARKNET_LOC,
				"[vit] at line #%ld has conflicting filters=%d and dim=%d; use one embedding width",
				s.line_number, filters, dim);
		}
		filters = dim;
	}

	int patch_size = s.find_int("patch_size", 1);
	int patch_stride = s.find_int("patch_stride", patch_size);
	int patch_pad = s.find_int("patch_pad", 0);
	int heads = s.find_int("heads", 4);
	const bool has_ffn_ratio = s.exists("ffn_ratio");
	const bool has_mlp_dim = s.exists("mlp_dim");
	int ffn_ratio = has_ffn_ratio ? s.find_int("ffn_ratio") : 4;
	int mlp_dim = has_mlp_dim ? s.find_int("mlp_dim") : filters * ffn_ratio;
	if (has_ffn_ratio && has_mlp_dim && mlp_dim != filters * ffn_ratio)
	{
		darknet_fatal_error(DARKNET_LOC,
			"[vit] at line #%ld has conflicting ffn_ratio=%d and mlp_dim=%d for dim=%d",
			s.line_number, ffn_ratio, mlp_dim, filters);
	}
	if (mlp_dim < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "[vit] at line #%ld has invalid mlp_dim=%d", s.line_number, mlp_dim);
	}

	const std::string pos_embed = s.find_str("pos_embed", "learned");
	int pos_embed_type = 0;
	if (pos_embed == "learned" || pos_embed == "absolute")
	{
		pos_embed_type = 0;
	}
	else if (pos_embed == "sinusoidal" || pos_embed == "2d_sinusoidal" || pos_embed == "simple")
	{
		pos_embed_type = 1;
	}
	else
	{
		darknet_fatal_error(DARKNET_LOC,
			"[vit] at line #%ld has unsupported pos_embed=%s; expected learned or sinusoidal",
			s.line_number, pos_embed.c_str());
	}

	const std::string pos_init = s.find_str("pos_init", "random");
	int pos_init_type = 0;
	if (pos_init == "random" || pos_init == "small_random")
	{
		pos_init_type = 0;
	}
	else if (pos_init == "zero" || pos_init == "zeros")
	{
		pos_init_type = 1;
	}
	else
	{
		darknet_fatal_error(DARKNET_LOC,
			"[vit] at line #%ld has unsupported pos_init=%s; expected random or zero",
			s.line_number, pos_init.c_str());
	}
	auto activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "gelu")));

	Darknet::Layer l = make_vit_layer(
		parms.batch, parms.h, parms.w, parms.c, filters,
		patch_size, patch_stride, patch_pad, heads, ffn_ratio, mlp_dim, pos_embed_type, pos_init_type,
		activation, parms.index, parms.train);

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	return l;
}

Darknet::Layer Darknet::CfgFile::parse_mambavision_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int h = parms.h;
	int w = parms.w;
	int c = parms.c;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before mambavision layer must output image");
	}

	int filters = s.find_int("filters", c);
	int d_state = s.find_int("state", s.find_int("d_state", 8));
	int conv_size = s.find_int("conv_size", s.find_int("size", 3));
	int dt_rank = s.find_int("dt_rank", 0);
	int ffn_ratio = s.find_int("ffn_ratio", 4);
	auto activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "gelu")));

	Darknet::Layer l = make_mambavision_layer(
		parms.batch, h, w, c, filters,
		d_state, conv_size, dt_rank, ffn_ratio, activation, parms.index, parms.train);

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	return l;
}

Darknet::Layer Darknet::CfgFile::parse_dcnv4_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int n = s.find_int("filters", 1);
	int groups = s.find_int("groups", 1);
	int size = s.find_int("size", 1);
	int stride = -1;
	int stride_x = s.find_int("stride_x", -1);
	int stride_y = s.find_int("stride_y", -1);
	if (stride_x < 1 || stride_y < 1)
	{
		stride = s.find_int("stride", 1);
		if (stride_x < 1) stride_x = stride;
		if (stride_y < 1) stride_y = stride;
	}
	int dilation = s.find_int("dilation", 1);
	int pad = s.find_int("pad", 0);
	int padding = s.find_int("padding", 0);
	if (pad)
	{
		padding = size / 2;
	}

	ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "logistic")));
	int batch_normalize = s.find_int("batch_normalize", 0);

	float offset_scale = s.find_float("offset_scale", 1.0f);
	int remove_center = s.find_int("remove_center", 0);
	int d_stride = s.find_int("d_stride", 8);
	int block_thread = s.find_int("block_thread", 256);
	int use_softmax = s.find_int("softmax", 0);

	int h = parms.h;
	int w = parms.w;
	int c = parms.c;
	int batch = parms.batch;

	Darknet::Layer l = make_dcnv4_layer(batch, 1, h, w, c, n, groups, size, stride_x, stride_y, dilation, padding, activation, batch_normalize, offset_scale, remove_center, d_stride, block_thread, use_softmax, parms.index, parms.train);
	l.batch_normalize = batch_normalize;

	return l;
}

Darknet::Layer Darknet::CfgFile::parse_detr_decoder_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int queries     = s.find_int("queries", 100);
	const int classes     = s.find_int("classes", 80);
	const int heads       = s.find_int("heads", 8);
	const int ffn         = s.find_int("ffn", std::max(1, parms.c));
	const int max_boxes   = s.find_int("max_boxes", 90);
	const float cls_weight  = s.find_float("cls_weight", 1.0f);
	const float l1_weight   = s.find_float("l1_weight", 5.0f);
	const float giou_weight = s.find_float("giou_weight", 2.0f);
	const float noobj_weight = s.find_float("noobj_weight", 1.0f);

	Darknet::Layer l = make_detr_decoder_layer(parms.batch, parms.h, parms.w, parms.c,
			queries, classes, heads, ffn, max_boxes,
			cls_weight, l1_weight, giou_weight, noobj_weight,
			parms.index, parms.train);
	l.jitter = s.find_float("jitter", 0.2f);
	l.resize = s.find_float("resize", 1.0f);
	return l;
}

Darknet::Layer Darknet::CfgFile::parse_wmhf_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int filters = s.find_int("filters", parms.c);
	const float identity_ratio = s.find_float("identity_ratio", 0.25f);
	const float local_ratio = s.find_float("local_ratio", 0.375f);
	const float freq_scale = s.find_float("freq_scale", 0.25f);
	const int shortcut = s.find_int("shortcut", 1);
	const int batch_normalize = s.find_int("batch_normalize", 1);
	const ACTIVATION activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "leaky")));

	return Darknet::make_wmhf_layer(
		parms.batch,
		parms.h,
		parms.w,
		parms.c,
		filters,
		identity_ratio,
		local_ratio,
		freq_scale,
		shortcut,
		activation,
		batch_normalize,
		net.adam,
		parms.index,
		parms.train);
}

Darknet::Layer Darknet::CfgFile::parse_transformer_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int h = parms.h;
	int w = parms.w;
	int c = parms.c;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before transformer layer must output image");
	}

	int filters = s.find_int("filters", 128);
	int size = s.find_int("size", 7);
	int heads = s.find_int("heads", 4);
	int shift = s.find_int("shift", 0);
	int ffn_ratio = s.find_int("ffn_ratio", 4);
	auto activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "gelu")));

	Darknet::Layer l = make_transformer_layer(
		parms.batch, h, w, c, filters,
		size, heads, shift, ffn_ratio, activation, parms.index, parms.train);

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	return l;
}

Darknet::Layer Darknet::CfgFile::parse_tucker_attention_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	int h = parms.h;
	int w = parms.w;
	int c = parms.c;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before tucker_attention layer must output image");
	}

	int filters = s.find_int("filters", c);
	int size = s.find_int("size", 7);
	int heads = s.find_int("heads", 4);
	int rank = s.find_int("rank", 0);
	int default_rank = c / 8;
	if (default_rank < 1) default_rank = 1;
	int rank_q = s.find_int("rank_q", rank > 0 ? rank : default_rank);
	int rank_k = s.find_int("rank_k", rank > 0 ? rank : rank_q);
	int rank_v = s.find_int("rank_v", rank > 0 ? rank : rank_q);
	int rank_o = s.find_int("rank_o", rank > 0 ? rank : rank_q);
	auto activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "linear")));

	Darknet::Layer l = make_tucker_attention_layer(
		parms.batch, h, w, c, filters,
		size, heads, rank_q, rank_k, rank_v, rank_o,
		activation, parms.index, parms.train);

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	return l;
}

Darknet::Layer Darknet::CfgFile::parse_clifford_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int h = parms.h;
	const int w = parms.w;
	const int c = parms.c;
	if (!(h && w && c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before clifford layer must output image");
	}

	const int filters = s.find_int("filters", c);
	auto shifts = s.find_int_array("shifts");
	auto shifts_wedge = s.find_int_array("shifts_wedge");
	auto shifts_inner = s.find_int_array("shifts_inner");
	if (shifts.empty() && shifts_wedge.empty() && shifts_inner.empty())
	{
		shifts = {1, 2};
	}
	if (shifts_wedge.empty())
	{
		shifts_wedge = shifts.empty() ? shifts_inner : shifts;
	}
	if (shifts_inner.empty())
	{
		if (shifts.empty())
		{
			shifts_inner = shifts_wedge;
		}
		else if (shifts_wedge != shifts)
		{
			shifts_inner = shifts;
		}
	}

		const std::string ctx_mode_str = s.find_str("ctx_mode", "diff");
		const std::string cli_mode_str = s.find_str("cli_mode", "full");
		const std::string gffn_mode_str = s.find_str("gffn_mode", "local");
		const std::string cli_higher_mode_str = s.find_str("cli_higher_mode", "off");
		const int dwconv_size = s.find_int("dwconv_size", 3);
	const int num_dwconv = s.find_int("num_dwconv", 1);
	const auto activation = static_cast<ACTIVATION>(get_activation_from_name(s.find_str("activation", "swish")));
	const float drop_path = s.find_float("drop_path", 0.0f);
	const float layerscale = s.find_float("layerscale", 0.0001f);

	int ctx_mode = -1;
	if (ctx_mode_str == "abs")
	{
		ctx_mode = 0;
	}
	else if (ctx_mode_str == "diff")
	{
		ctx_mode = 1;
	}
	else
	{
		darknet_fatal_error(DARKNET_LOC, "unknown clifford ctx_mode \"%s\"", ctx_mode_str.c_str());
	}

	int cli_mode = -1;
	if (cli_mode_str == "inner")
	{
		cli_mode = 0;
	}
	else if (cli_mode_str == "wedge")
	{
		cli_mode = 1;
	}
	else if (cli_mode_str == "full")
	{
		cli_mode = 2;
	}
	else
	{
		darknet_fatal_error(DARKNET_LOC, "unknown clifford cli_mode \"%s\"", cli_mode_str.c_str());
	}

	int gffn_mode = -1;
	if (gffn_mode_str == "local")
	{
		gffn_mode = 0;
	}
	else if (gffn_mode_str == "global")
	{
		gffn_mode = 1;
	}
	else if (gffn_mode_str == "hybrid")
	{
		gffn_mode = 2;
	}
		else
		{
			darknet_fatal_error(DARKNET_LOC, "unknown clifford gffn_mode \"%s\"", gffn_mode_str.c_str());
		}

		int higher_mode = -1;
		if (cli_higher_mode_str == "off")
		{
			higher_mode = 0;
		}
		else if (cli_higher_mode_str == "vb")
		{
			higher_mode = 1;
		}
		else
		{
			darknet_fatal_error(DARKNET_LOC, "unknown clifford cli_higher_mode \"%s\"", cli_higher_mode_str.c_str());
		}

		Darknet::Layer l = make_clifford_layer(
			parms.batch, h, w, c, filters,
			shifts_wedge.data(), static_cast<int>(shifts_wedge.size()),
			shifts_inner.empty() ? nullptr : shifts_inner.data(), static_cast<int>(shifts_inner.size()),
			ctx_mode, cli_mode, gffn_mode, higher_mode,
			dwconv_size, num_dwconv,
			activation, drop_path, layerscale,
			parms.index, parms.train);

	l.stream = s.find_int("stream", -1);
	l.wait_stream_id = s.find_int("wait_stream", -1);

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_one_body_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);
	Darknet::Layer l = {(Darknet::ELayerType)0};

	switch (s.type)
	{
		case ELayerType::CONVOLUTIONAL:    l = parse_convolutional_section(section_idx);  break;
		case ELayerType::MAXPOOL:          l = parse_maxpool_section(section_idx);         break;
		case ELayerType::AVGPOOL:          l = parse_avgpool_section(section_idx);         break;
		case ELayerType::LOCAL_AVGPOOL:    l = parse_local_avgpool_section(section_idx);   break;
		case ELayerType::UPSAMPLE:         l = parse_upsample_section(section_idx);        break;
		case ELayerType::CHANNEL_SHUFFLE:  l = parse_channel_shuffle_section(section_idx); break;
		case ELayerType::CONNECTED:        l = parse_connected_section(section_idx);       break;
		case ELayerType::DROPOUT:          l = parse_dropout_section(section_idx);         break;
		default:
		{
			darknet_fatal_error(DARKNET_LOC,
				"layer type \"%s\" on line #%ld is not supported inside [recursive_block] body",
				s.name.c_str(), s.line_number);
		}
	}

	// Advance parms to reflect this body layer's output dimensions
	parms.h      = l.out_h;
	parms.w      = l.out_w;
	parms.c      = l.out_c;
	parms.inputs = l.outputs;

	return l;
}


Darknet::Layer Darknet::CfgFile::parse_recursive_block_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	if (!(parms.h && parms.w && parms.c))
	{
		darknet_fatal_error(DARKNET_LOC, "layer before [recursive_block] on line #%ld must output image", s.line_number);
	}

	const int loops     = s.find_int("loops",     4);
	const int injection = s.find_int("injection", 1);
	const int body_n    = s.find_int("body",      1);
	const float default_body_scale      = injection ? (1.0f / 3.0f) : 0.5f;
	const float default_residual_scale  = injection ? (1.0f / 3.0f) : 0.5f;
	const float default_injection_scale = injection ? (1.0f / 3.0f) : 0.0f;
	const float body_scale      = s.find_float("body_scale",      default_body_scale);
	const float residual_scale  = s.find_float("residual_scale",  default_residual_scale);
	const float injection_scale = s.find_float("injection_scale", default_injection_scale);

	if (loops < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "[recursive_block] on line #%ld: loops must be >= 1", s.line_number);
	}
	if (body_n < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "[recursive_block] on line #%ld: body must be >= 1", s.line_number);
	}
	if (section_idx + (size_t)body_n >= sections.size())
	{
		darknet_fatal_error(DARKNET_LOC, "[recursive_block] on line #%ld: body=%d but not enough sections follow", s.line_number, body_n);
	}

	const int in_h = parms.h;
	const int in_w = parms.w;
	const int in_c = parms.c;

	Darknet::Layer l = make_recursive_block_layer(parms.batch, in_h, in_w, in_c, loops, injection,
		body_scale, residual_scale, injection_scale, parms.train);

	// Stage 1/2 Ouroboros options
	l.rb_ouroboros        = s.find_int  ("ouroboros",        0);
	l.rb_gate_bias        = s.find_float("gate_bias",       -2.0f);
	l.rb_gamma_clip       = s.find_float("gamma_clip",       4.0f);
	l.rb_gate_clip        = s.find_float("gate_clip",       30.0f);
	// Stage 2 Conv-LoRA (ignored unless ouroboros=2)
	l.rb_lora_rank        = s.find_int  ("lora_rank",        0);
	l.rb_lora_alpha       = s.find_float("lora_alpha",       1.0f);
	l.rb_lora_freeze_base = s.find_int  ("lora_freeze_base", 0);
	l.rb_lora_diag_clip   = s.find_float("lora_diag_clip",   4.0f);
	l.rb_lora_diag_init   = s.find_float("lora_diag_init",   1.0f);
	// LDT-style lattice projection (disabled when lattice_mix=0)
	{
		const int   lattice_group       = s.find_int  ("lattice_group",       0);
		const float lattice_threshold   = s.find_float("lattice_threshold",   0.10f);
		const float lattice_temperature = s.find_float("lattice_temperature", 1.0f);
		const float lattice_mix         = s.find_float("lattice_mix",         0.0f);
		if (lattice_mix > 0.0f || lattice_group > 0)
		{
			set_recursive_block_lattice_projection(&l, lattice_group, lattice_threshold, lattice_temperature, lattice_mix);
		}
	}

	l.rb_body       = (Darknet::Layer*)xcalloc(body_n, sizeof(Darknet::Layer));
	l.rb_body_count = body_n;
	l.extra         = body_n;
	l.index         = parms.index;

	const int saved_index = parms.index;
	for (int j = 0; j < body_n; ++j)
	{
		parms.index = saved_index;
		l.rb_body[j] = parse_one_body_section(section_idx + 1 + j);
		l.workspace_size = std::max(l.workspace_size, l.rb_body[j].workspace_size);
	}
	parms.index = saved_index;

	if (parms.h != in_h || parms.w != in_w || parms.c != in_c)
	{
		darknet_fatal_error(DARKNET_LOC,
			"[recursive_block] on line #%ld: body changes dims from %dx%dx%d to %dx%dx%d; "
			"body must be size-preserving (same h, w, c) for loops to work",
			s.line_number, in_h, in_w, in_c, parms.h, parms.w, parms.c);
	}

	parms.h      = in_h;
	parms.w      = in_w;
	parms.c      = in_c;
	parms.inputs = l.outputs;

	return l;
}




static void parse_modern_yolo_common(Darknet::CfgSection & s, Darknet::Layer & l, const float vfl_gamma_default, const int score_mode_default)
{
	// Training schedule keys:
	// - atss_warmup_iters: PP-YOLOE/YOLO-NAS use IoU-only ATSS-lite assignment warmup; 0 disables.
	// - l1_final_iters: YOLOX adds L1 box loss for the final no-aug fine-tune iterations; 0 disables.
	l.jitter = s.find_float("jitter", 0.2f);
	l.resize = s.find_float("resize", 1.0f);
	l.ignore_thresh = s.find_float("ignore_thresh", 0.7f);
	l.iou_normalizer = s.find_float("iou_normalizer", 1.0f);
	l.obj_normalizer = s.find_float("obj_normalizer", 1.0f);
	l.cls_normalizer = s.find_float("cls_normalizer", 1.0f);
	l.max_delta = s.find_float("max_delta", FLT_MAX);
	l.label_smooth_eps = s.find_float("label_smooth_eps", 0.0f);
	l.vfl_gamma = s.find_float("vfl_gamma", vfl_gamma_default);
	l.score_mode = s.find_int("score_mode", score_mode_default);
	l.atss_warmup_iters = s.find_int("atss_warmup_iters", 0);
	l.l1_final_iters = s.find_int("l1_final_iters", 0);
	const std::string iou_loss = s.find_str("iou_loss", "iou");
	l.iou_loss = static_cast<IOU_LOSS>(Darknet::get_IoU_loss_from_name(iou_loss));
}


Darknet::Layer Darknet::CfgFile::parse_yolox_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int classes   = s.find_int("classes", 20);
	const int max_boxes = s.find_int("max", 200);

	Darknet::Layer l = Darknet::make_yolox_layer(parms.batch, parms.w, parms.h, classes, max_boxes);
	parse_modern_yolo_common(s, l, 0.0f, 0);
	// Released YOLOX reference code uses 1.5; keep this repo default configurable at 2.5.
	l.center_radius = s.find_float("center_radius", 2.5f);
	l.assign_topk = s.find_int("assign_topk", 10);
	l.box_loss_weight = s.find_float("box_loss_weight", 5.0f);
	l.yolox_soft_label = s.find_int("yolox_soft_label", 0);
	l.yolox_ignore_neg = s.find_int("yolox_ignore_neg", 0);
	return l;
}


Darknet::Layer Darknet::CfgFile::parse_ppyoloe_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int classes   = s.find_int("classes", 20);
	const int max_boxes = s.find_int("max", 200);
	const int reg_max   = s.find_int("reg_max", 16);

	Darknet::Layer l = Darknet::make_ppyoloe_layer(parms.batch, parms.w, parms.h, classes, max_boxes, reg_max);
	parse_modern_yolo_common(s, l, 2.0f, 1);
	l.center_radius = s.find_float("center_radius", 0.0f);
	l.assign_topk = s.find_int("assign_topk", 13);
	l.tal_alpha = s.find_float("tal_alpha", 1.0f);
	l.tal_beta = s.find_float("tal_beta", 6.0f);
	l.box_loss_weight = s.find_float("box_loss_weight", 2.5f);
	l.dfl_loss_weight = s.find_float("dfl_loss_weight", 0.5f);
	l.nas_duplicate_decay = 0.0f;
	return l;
}


Darknet::Layer Darknet::CfgFile::parse_yolonas_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int classes   = s.find_int("classes", 20);
	const int max_boxes = s.find_int("max", 200);
	const int reg_max   = s.find_int("reg_max", 16);

	Darknet::Layer l = Darknet::make_yolonas_layer(parms.batch, parms.w, parms.h, classes, max_boxes, reg_max);
	parse_modern_yolo_common(s, l, 2.0f, 1);
	l.center_radius = s.find_float("center_radius", 0.0f);
	l.assign_topk = s.find_int("assign_topk", 13);
	l.tal_alpha = s.find_float("tal_alpha", 1.0f);
	l.tal_beta = s.find_float("tal_beta", 6.0f);
	l.box_loss_weight = s.find_float("box_loss_weight", 2.5f);
	l.dfl_loss_weight = s.find_float("dfl_loss_weight", 0.25f);
	l.nas_duplicate_decay = s.find_float("nas_duplicate_decay", 0.35f);
	return l;
}


Darknet::Layer Darknet::CfgFile::parse_centernet_section(const size_t section_idx)
{
	TAT(TATPARMS);

	auto & s = sections.at(section_idx);

	const int classes   = s.find_int("classes", 20);
	const int max_boxes = s.find_int("max", 200);

	Darknet::Layer l = Darknet::make_centernet_layer(parms.batch, parms.w, parms.h, classes, max_boxes);
	l.jitter = s.find_float("jitter", 0.2f);
	l.resize = s.find_float("resize", 1.0f);
	l.ct_min_radius = std::max(0, s.find_int("center_min_radius", l.ct_min_radius));
	l.ct_peak_nms = s.find_int("peak_nms", l.ct_peak_nms);
	l.ct_anisotropic_gaussian = s.find_int("anisotropic_gaussian", l.ct_anisotropic_gaussian);
	l.ct_small_threshold = s.find_float("small_threshold", l.ct_small_threshold);
	l.ct_small_ref_size = s.find_float("small_ref_size", l.ct_small_ref_size);
	l.ct_small_boost = s.find_float("small_boost", l.ct_small_boost);
	l.ct_scale_min_px = s.find_float("scale_min_px", l.ct_scale_min_px);
	l.ct_scale_max_px = s.find_float("scale_max_px", l.ct_scale_max_px);
	l.ct_gaussian_iou = s.find_float("gaussian_iou", l.ct_gaussian_iou);
	l.ct_focal_alpha = s.find_float("focal_alpha", l.ct_focal_alpha);
	l.ct_focal_beta = s.find_float("focal_beta", l.ct_focal_beta);
	l.ct_hm_normalizer = s.find_float("hm_normalizer", l.ct_hm_normalizer);
	l.ct_wh_normalizer = s.find_float("wh_normalizer", l.ct_wh_normalizer);
	l.ct_off_normalizer = s.find_float("off_normalizer", l.ct_off_normalizer);
	l.max_delta = s.find_float("max_delta", l.max_delta);
	l.delta_normalizer = s.find_float("delta_normalizer", l.delta_normalizer);
	return l;
}
