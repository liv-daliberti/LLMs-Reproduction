# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /n/fs/similarity/open-r1/.torchinductor/nx/cnxiucwmiqr2rkuifj6kx2miarjclrklwnhmipjlc5d3koqjwlf4.py
# Topologically Sorted Source Nodes: [ge, lt, org_vocab_mask, ge_1, lt_1, added_vocab_mask, vocab_mask, mul, mul_1, valid_offset, sub_3, input_, invert], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.mul, aten.add, aten.sub, aten.bitwise_not]
# Source node to ATen node mapping:
#   added_vocab_mask => bitwise_and_1
#   ge => ge
#   ge_1 => ge_1
#   input_ => mul_6
#   invert => bitwise_not
#   lt => lt
#   lt_1 => lt_1
#   mul => mul
#   mul_1 => mul_2
#   org_vocab_mask => bitwise_and
#   sub_3 => sub_13
#   valid_offset => add_16
#   vocab_mask => bitwise_or
# Graph fragment:
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg1_1, %arg2_1), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg1_1, %arg3_1), kwargs = {})
#   %bitwise_and : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %lt), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg1_1, %arg4_1), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg1_1, %arg5_1), kwargs = {})
#   %bitwise_and_1 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_1, %lt_1), kwargs = {})
#   %bitwise_or : [num_users=2] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and, %bitwise_and_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_and, %arg2_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_and_1, %sub_8), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_2), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %add_16), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_or, %sub_13), kwargs = {})
#   %bitwise_not : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_or,), kwargs = {})
triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0 = async_compile.triton('triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'out_ptr1': '*i1', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'ks4': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A7D185CFC7E5BAAEAD4D34B59C10552455DF860BCF0A270A0DF4AA2561AD3A4F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = ks0
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = ks2
    tmp7 = tmp0 >= tmp6
    tmp8 = ks3
    tmp9 = tmp0 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 | tmp10
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp5.to(tl.int64)
    tmp14 = tmp13 * tmp1
    tmp15 = tmp10.to(tl.int64)
    tmp16 = ks0 + ks2 + ((-1)*ks1) + ((-1)*ks4)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp12 * tmp19
    tmp21 = tmp11 == 0
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp21, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg2_1
    s2 = arg3_1
    s3 = arg4_1
    s4 = arg5_1
    s5 = arg6_1
    assert_size_stride(arg1_1, (s0, ), (1, ))
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        buf0 = empty_strided_cuda((s0, ), (1, ), torch.int64)
        buf1 = empty_strided_cuda((s0, ), (1, ), torch.bool)
        # Topologically Sorted Source Nodes: [ge, lt, org_vocab_mask, ge_1, lt_1, added_vocab_mask, vocab_mask, mul, mul_1, valid_offset, sub_3, input_, invert], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.mul, aten.add, aten.sub, aten.bitwise_not]
        stream1 = get_raw_stream(1)
        triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0.run(arg1_1, buf0, buf1, s1, s2, s3, s4, s5, s0, grid=grid(s0), stream=stream1)
        del arg1_1
    return (buf0, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2048
    arg1_1 = rand_strided((2048, ), (1, ), device='cuda:1', dtype=torch.int64)
    arg2_1 = 76032
    arg3_1 = 152064
    arg4_1 = 152064
    arg5_1 = 152064
    arg6_1 = 0
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
