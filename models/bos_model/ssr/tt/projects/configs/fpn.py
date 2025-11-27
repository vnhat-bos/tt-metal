import ttnn

module_config = {
	"lateral_convs": {
		"0": {
			"conv": {
				"config": {
					"dtype": ttnn.bfloat16,
					"weights_dtype": ttnn.bfloat8_b,
					"activation": None,
					"act_block_h_override": 0,
					"deallocate_activation": True,
					"reallocate_halo_output": False,
					"in_place": False,
					"reshard_if_not_optimal": False,
					"override_sharding_config": True,
					"shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 5)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    "full_inner_dim": True,
				},
			}
		}
	},
	"fpn_convs": {
		"0": {
			"conv": {
				"config": {
					"dtype": ttnn.bfloat16,
					"weights_dtype": ttnn.bfloat8_b,
					"activation": None,
					"act_block_h_override": 0,
					"deallocate_activation": True,
					"reallocate_halo_output": False,
					"in_place": False,
					"reshard_if_not_optimal": False,
					"override_sharding_config": True,
					"shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 5)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    "full_inner_dim": True,
				},
			}
		}
	}
}