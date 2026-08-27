#include <gtest/gtest.h>

#include <algorithm>

#include "darknet_internal.hpp"
#include "dcnv4_layer.hpp"

namespace
{
	size_t expected_dcnv4_workspace_floats(const Darknet::Layer & l)
	{
		int K = l.size * l.size;
		if (l.remove_center)
		{
			K -= 1;
		}
		const int offset_filters_raw = l.groups * K * 3;
		const int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
		const int d_stride = std::max(1, l.d_stride);
		const int H_c = (l.out_h + d_stride - 1) / d_stride;
		const int W_c = (l.out_w + d_stride - 1) / d_stride;

		const size_t input_nhwc = static_cast<size_t>(l.batch) * l.c * l.h * l.w;
		const size_t output_nhwc = static_cast<size_t>(l.batch) * l.n * l.out_h * l.out_w;
		const size_t offsets_nhwc = static_cast<size_t>(l.batch) * l.out_h * l.out_w * padded_offset_dim;
		const size_t coarse_offsets = static_cast<size_t>(l.batch) * H_c * W_c * padded_offset_dim;
		const size_t im2col = static_cast<size_t>(l.c) * l.size * l.size * l.out_h * l.out_w;
		const size_t coarse_im2col = static_cast<size_t>(l.c) * l.size * l.size * H_c * W_c;
		const size_t offset_tail = std::max(im2col, coarse_im2col + coarse_offsets);

		return input_nhwc + output_nhwc + offsets_nhwc + std::max(offset_tail, 2 * input_nhwc + offset_tail);
	}
}

TEST(DCNv4Layer, WorkspaceCoversBackwardAliasesBeforeAndAfterResize)
{
	Darknet::Layer l = make_dcnv4_layer(
		2, 1, 13, 11, 8, 8, 2, 3, 1, 1, 1, 1,
		LINEAR, 0, 1.0f, 0, 4, 256, 1, 0, 1);

	EXPECT_EQ(l.steps, 1);
	EXPECT_EQ(l.out_h, 13);
	EXPECT_EQ(l.out_w, 11);
	EXPECT_GE(l.workspace_size / sizeof(float), expected_dcnv4_workspace_floats(l));

	resize_dcnv4_layer(&l, 19, 17);

	EXPECT_EQ(l.h, 17);
	EXPECT_EQ(l.w, 19);
	EXPECT_EQ(l.out_h, 17);
	EXPECT_EQ(l.out_w, 19);
	EXPECT_EQ(l.outputs, l.out_h * l.out_w * l.out_c);
	EXPECT_EQ(l.inputs, l.h * l.w * l.c);
	EXPECT_GE(l.workspace_size / sizeof(float), expected_dcnv4_workspace_floats(l));

	free_layer(l);
}
