#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace Darknet
{
	/// Shared by fp8_read_calibration_scales() and fp4_read_calibration_scales() -- the
	/// on-disk calibration sidecar format is identical between the two precisions, only
	/// the entry/table type differs.
	template <typename Table>
	bool read_calibration_scales_impl(const std::filesystem::path & path, Table & table)
	{
		table.clear();

		std::ifstream input(path);
		if (!input.good())
		{
			return false;
		}

		std::string line;
		while (std::getline(input, line))
		{
			if (line.empty() || line[0] == '#')
			{
				continue;
			}

			std::istringstream stream(line);
			typename Table::mapped_type entry;
			if (!(stream >> entry.layer_index >> entry.amax >> entry.scale))
			{
				return false;
			}
			if (entry.layer_index < 0 || !std::isfinite(entry.amax) || !std::isfinite(entry.scale) || entry.scale <= 0.0f)
			{
				return false;
			}
			table[entry.layer_index] = entry;
		}

		return input.eof();
	}

	/// Shared by fp8_write_calibration_scales() and fp4_write_calibration_scales().
	template <typename Entry>
	bool write_calibration_scales_impl(const std::filesystem::path & path, const std::vector<Entry> & entries)
	{
		std::ofstream output(path);
		if (!output.good())
		{
			return false;
		}

		output << "# layer_index amax scale\n";
		output << std::setprecision(9);
		for (const auto & entry : entries)
		{
			if (entry.layer_index < 0 || !std::isfinite(entry.amax) || !std::isfinite(entry.scale) || entry.scale <= 0.0f)
			{
				return false;
			}
			output << entry.layer_index << ' ' << entry.amax << ' ' << entry.scale << '\n';
		}

		return output.good();
	}
}
