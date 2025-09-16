
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'ks4': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A7D185CFC7E5BAAEAD4D34B59C10552455DF860BCF0A270A0DF4AA2561AD3A4F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = ks0
    tmp3 = tmp1 >= tmp2
    tmp4 = ks1
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = ks2
    tmp8 = tmp1 >= tmp7
    tmp9 = ks3
    tmp10 = tmp1 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp6 | tmp11
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tmp6.to(tl.int64)
    tmp15 = tmp14 * tmp2
    tmp16 = tmp11.to(tl.int64)
    tmp17 = ks0 + ks2 + ((-1)*ks1) + ((-1)*ks4)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp1 - tmp19
    tmp21 = tmp13 * tmp20
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp21, None)
