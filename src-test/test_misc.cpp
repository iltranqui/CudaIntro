#include <gtest/gtest.h>
#include "darknet_internal.hpp"


TEST(NN, FreeNullPtr)
{
	Darknet::NetworkPtr ptr = nullptr;

	ASSERT_TRUE(ptr == nullptr);
	Darknet::free_neural_network(ptr);
	ASSERT_TRUE(ptr == nullptr);
}


TEST(Darknet, Trim)
{
	ASSERT_STREQ("testing"		, Darknet::trim(" testing "			).c_str());
	ASSERT_STREQ("whitespace"	, Darknet::trim("\n whitespace\r\n"	).c_str());
	ASSERT_STREQ("whitespace"	, Darknet::trim(" \twhitespace\n"	).c_str());
	ASSERT_STREQ("whitespace"	, Darknet::trim("\t whitespace\r"	).c_str());
	ASSERT_STREQ(""				, Darknet::trim("   "				).c_str());
}


TEST(Darknet, Lowercase)
{
	ASSERT_STREQ("012345"		, Darknet::lowercase("012345"		).c_str());
	ASSERT_STREQ("  testing  "	, Darknet::lowercase("  TeStInG  "	).c_str());
	ASSERT_STREQ(""				, Darknet::lowercase(""				).c_str());
}


TEST(Darknet, LayerDiagnosticLabels)
{
	EXPECT_EQ(Darknet::layer_type_diagnostic_label(Darknet::ELayerType::CONVOLUTIONAL), "CONV");
	EXPECT_EQ(Darknet::layer_type_diagnostic_label(Darknet::ELayerType::YOLO), "YOLO");
	EXPECT_EQ(Darknet::layer_type_diagnostic_label(Darknet::ELayerType::TRANSFORMER), "TRANSFORMER");
	EXPECT_EQ(Darknet::layer_type_diagnostic_label(static_cast<Darknet::ELayerType>(-1)), "UNKNOWN");

	EXPECT_EQ(Darknet::layer_diagnostic_location(0, Darknet::ELayerType::CONVOLUTIONAL), "layer 0 [CONV layer]");
}
