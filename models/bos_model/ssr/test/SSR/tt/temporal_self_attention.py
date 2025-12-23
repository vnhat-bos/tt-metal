from test.utils import masked_fill, compare_tensors, tt2pt

import tracy
from bos_metal import op
from mmcv.cnn.bricks.registry import ATTENTION

import torch
import ttnn
import tracy

from tt.projects.configs.ops_config import MyDict

@ATTENTION.register_module(name="TemporalSelfAttention_tt", force=True)
class TemporalSelfAttention(op.BaseModule):
    counter = 0
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_bev_queue=2,
        batch_first=True,
        device=None,
        **kwargs,
    ):
        super(TemporalSelfAttention, self).__init__(device=device, **kwargs)
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue

        self.sampling_offsets = op.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = op.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points,
        )
        self.value_proj = op.Linear(embed_dims, embed_dims)
        self.output_proj = op.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        padding_attn_mask=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        bilinear_weight_hash=None,
        use_prev_bev=True,
        memory_config=MyDict(),
        program_config=MyDict(),
        **kwargs,
    ):
        tmp = ttnn.sharded_to_interleaved(query, memory_config=ttnn.L1_MEMORY_CONFIG)
        # compare_tensors(tt2pt(tmp)[:, :10_000], "/tmp/bev_query.pt", "temporal/shard2interleave query")
        if value is None:
            assert self.batch_first, "batch_first should be True if value is None"
            value = ttnn.concat([tmp, tmp], 0, memory_config=ttnn.L1_MEMORY_CONFIG)
            use_prev_bev = False

        if identity is None:
            identity = query
        if query_pos is not None:
            query_ = ttnn.add_(tmp, query_pos)

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape

        query = ttnn.concat([value[:bs], query_], -1, memory_config=memory_config["query"].value)
        # compare_tensors(tt2pt(value)[:, :10_000], f"/tmp/value.{TemporalSelfAttention.counter}.pt", "initial value")
        # compare_tensors(tt2pt(query)[:, :10_000], f"/tmp/query.{TemporalSelfAttention.counter}.pt", "initial query")
        ttnn.deallocate(query_)

        # value = ttnn.reallocate(value)
        # NOTE: Using `ttnn.reallocate` here causes a pcc drop
        # TODO: Investigate why
        # value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        # value = ttnn.to_memory_config(value, memory_config["value"].value)
        value = (ttnn.to_memory_config(value, memory_config["value"].value)
                 if use_prev_bev else
                 ttnn.reallocate(value, memory_config=memory_config["value"].value))
        value_proj = self.value_proj(
            value,
            dtype=ttnn.bfloat16,
            memory_config=value.memory_config(),
            # memory_config=memory_config["value_proj"].value,
            program_config=program_config["value_proj"].value,
        )
        # compare_tensors(tt2pt(value_proj)[:, :10_000], f"/tmp/value_proj.{TemporalSelfAttention.counter}.pt", "value proj")
        # breakpoint()
        ttnn.deallocate(value)
        if key_padding_mask is not None:
            value_proj = masked_fill(value_proj, key_padding_mask[..., None], 0.0)
        value = ttnn.to_layout(value_proj, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(value_proj)
        # value = ttnn.to_layout(value_proj, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.reshape(value, (bs * self.num_bev_queue, num_value, self.num_heads, -1))
        # value = ttnn.reallocate(value)  # NOTE: move data takes 958us
        # compare_tensors(tt2pt(value)[:, :10_000], f"/tmp/value_proj.{TemporalSelfAttention.counter}.pt", "value reshape reallocate")

        attention_weights = self.attention_weights(
            query,
            dtype=ttnn.bfloat16,
            # memory_config=memory_config["attention_weights"].value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=program_config["attention_weights"].value,
        )
        # breakpoint()
        # compare_tensors(tt2pt(attention_weights)[:, :10_000], f"/tmp/attention_weights.{TemporalSelfAttention.counter}.pt", "attention weights")
        # TODO: attention weights need to be multiplied with attn_mask to mask out 752 padded values
        # attention_weights = ttnn.multiply_(attention_weights, attn_mask)
        # if padding_attn_mask is not None:
        #     attention_weights = ttnn.add_(attention_weights, padding_attn_mask)
        attention_weights = ttnn.reshape(
            attention_weights, (num_query * self.num_heads * self.num_bev_queue, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, -1)
        # compare_tensors(tt2pt(attention_weights).reshape(num_query, self.num_heads, -1)[:10_000], f"/tmp/attention_weights_softmax.{TemporalSelfAttention.counter}.pt")
        attention_weights = ttnn.reshape(
            attention_weights, (num_query * self.num_heads, self.num_bev_queue * self.num_levels * self.num_points)
        )

        sampling_offsets = self.sampling_offsets(
            query,
            dtype=ttnn.bfloat16,
            # memory_config=memory_config["sampling_offsets"].value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=program_config["sampling_offsets"].value,
        )
        # compare_tensors(tt2pt(sampling_offsets)[:, :10_000], f"/tmp/sampling_offsets.{TemporalSelfAttention.counter}.0.pt", "sampling_offsets")
        sampling_locations = ttnn.add_(sampling_offsets, reference_points)
        # tmp = torch.load(f"/tmp/sampling_locations.{TemporalSelfAttention.counter}.pt")
        # compare_tensors(
        #     tt2pt(sampling_locations)[:, :10_000], 
        #     tmp.permute(1, 2, 0, 3, 4, 5),
        #     "sampling locations"
        # )
        sampling_locations = ttnn.reshape(
            sampling_locations, (num_query * self.num_heads, self.num_bev_queue * self.num_levels * self.num_points * 2)
        )
        # compare_tensors(
        #     tt2pt(sampling_locations)[:, :10_000], 
        #     tmp.permute(1, 2, 0, 3, 4, 5),
        #     "sampling locations reshape"
        # )

        if spatial_shapes.layout != ttnn.TILE_LAYOUT:
            spatial_shapes = ttnn.to_layout(spatial_shapes, ttnn.TILE_LAYOUT)
        output = ttnn.bos_ssr_deformable_attention(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            is_denormed_grid=True,
            bilinear_weight_hash=bilinear_weight_hash,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_queries=num_query,
            num_levels=self.num_levels,
            num_points=self.num_points,
            is_QHB=True
        )
        ttnn.deallocate(value)
        # compare_tensors(tt2pt(output)[:, :10_000], f"/tmp/output_mean.{TemporalSelfAttention.counter}.pt", "output mean")

        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        output = ttnn.to_memory_config(output, identity.memory_config())
        # TODO: Inspect why output here is sharded onto [(0, 0) - (7, 4)] & [(0, 5) - (4, 5)]
        # instead of [(0, 0) - (7, 5)] like input
        output = self.output_proj(
            output,
            # memory_config=memory_config["output_proj"].value,
            memory_config=identity.memory_config(),
            program_config=program_config["output_proj"].value,
        )
        # compare_tensors(tt2pt(output)[:, :10_000], f"/tmp/output_proj.{TemporalSelfAttention.counter}.pt", "output proj")
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))
        output = ttnn.reshard(output, identity.memory_config())
        TemporalSelfAttention.counter += 1
        return ttnn.add_(identity, output)
