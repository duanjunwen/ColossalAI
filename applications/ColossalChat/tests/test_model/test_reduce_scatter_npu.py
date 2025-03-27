import torch
import torch.distributed as dist

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.cluster import DistCoordinator, ProcessGroupMesh


def test_reduce_scatter_tensor():
    acclerate = get_accelerator()
    DistCoordinator()
    device = acclerate.get_current_device()
    dist.get_rank()
    dtype = torch.bfloat16  # fail case: torch.int32, torch.int64
    dp_axis, pp_axis, tp_axis, sp_axis = 0, 1, 2, 3
    dp_size, pp_size, tp_size, sp_size = 2, 1, 8, 1
    dp_size * pp_size * tp_size * sp_size
    pg_mesh = ProcessGroupMesh(dp_size, pp_size, tp_size, sp_size)  # dp8tp2

    dp_group = pg_mesh.get_group_along_axis(dp_axis)
    pg_mesh.get_group_along_axis(pp_axis)
    tp_group = pg_mesh.get_group_along_axis(tp_axis)
    pg_mesh.get_group_along_axis(sp_axis)

    # init dp input output
    data_parallel_input = torch.arange(dp_size * 2, dtype=dtype, device=device)
    data_parallel_output = torch.empty(2, dtype=dtype).to(device)

    # On dp group reduce_scatter_tensor
    # In: rank 0 [0, 1, 2, 3]; rank 1 [0, 1, 2, 3];
    # Out: rank 0 [0, 2]; rank 1 [4, 6];
    print(f"Rank {device}: Data Parallel Input tensor {data_parallel_input.shape} = {data_parallel_input}")
    dist.reduce_scatter_tensor(data_parallel_output, data_parallel_input, group=dp_group)
    print(f"Rank {device}: Data Parallel Output tensor {data_parallel_output.shape} = {data_parallel_output}")

    # init tp input output
    tensor_parallel_input = torch.arange(tp_size * 2, dtype=dtype, device=device)
    tensor_parallel_output = torch.empty(2, dtype=dtype).to(device)

    # On tp group reduce_scatter_tensor
    print(f"Rank {device}: Tensor Parallel Input tensor {tensor_parallel_input.shape} = {tensor_parallel_input}")
    dist.reduce_scatter_tensor(tensor_parallel_output, tensor_parallel_input, group=tp_group)
    print(f"Rank {device}: Tensor Parallel Output tensor {tensor_parallel_output.shape} = {tensor_parallel_output}")

    pg_mesh.destroy_mesh_process_groups()


def test_reduce_scatter():
    acclerate = get_accelerator()
    DistCoordinator()
    device = acclerate.get_current_device()
    rank = dist.get_rank()
    dtype = torch.bfloat16  # fail case: torch.int32, torch.int64
    dp_axis, pp_axis, tp_axis, sp_axis = 0, 1, 2, 3
    dp_size, pp_size, tp_size, sp_size = 8, 1, 2, 1
    dp_size * pp_size * tp_size * sp_size
    pg_mesh = ProcessGroupMesh(dp_size, pp_size, tp_size, sp_size)  # dp8tp2

    dp_group = pg_mesh.get_group_along_axis(dp_axis)
    pg_mesh.get_group_along_axis(pp_axis)
    pg_mesh.get_group_along_axis(tp_axis)
    pg_mesh.get_group_along_axis(sp_axis)

    # init input tensor list
    input_tensor_list = [torch.ones(2, dtype=dtype).to(device) * (rank + i) for i in range(dp_size)]
    output = torch.empty(2, dtype=dtype).to(device)

    print(
        f"Rank {device}: Data Parallel Input tensor {len(input_tensor_list), input_tensor_list[0].shape} = {input_tensor_list}"
    )
    dist.reduce_scatter(output, input_tensor_list, group=dp_group)
    print(f"Rank {device}: Tensor Parallel Output tensor {output.shape} = {output}")

    pg_mesh.destroy_mesh_process_groups()


def test_all_reduce():
    acclerate = get_accelerator()
    DistCoordinator()
    device = acclerate.get_current_device()
    rank = dist.get_rank()
    dtype = torch.bfloat16  # fail case: torch.int32, torch.int64
    dp_axis, pp_axis, tp_axis, sp_axis = 0, 1, 2, 3
    dp_size, pp_size, tp_size, sp_size = 8, 1, 2, 1
    dp_size * pp_size * tp_size * sp_size
    pg_mesh = ProcessGroupMesh(dp_size, pp_size, tp_size, sp_size)  # dp8tp2

    dp_group = pg_mesh.get_group_along_axis(dp_axis)
    pg_mesh.get_group_along_axis(pp_axis)
    pg_mesh.get_group_along_axis(tp_axis)
    pg_mesh.get_group_along_axis(sp_axis)

    input_tensor = torch.tensor([rank], dtype=dtype, device=device)
    dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM, group=dp_group)


def test_all_gather():
    acclerate = get_accelerator()
    DistCoordinator()
    device = acclerate.get_current_device()
    rank = dist.get_rank()
    dtype = torch.bfloat16  # fail case: torch.int32, torch.int64
    dp_axis, pp_axis, tp_axis, sp_axis = 0, 1, 2, 3
    dp_size, pp_size, tp_size, sp_size = 8, 1, 2, 1
    dp_size * pp_size * tp_size * sp_size
    pg_mesh = ProcessGroupMesh(dp_size, pp_size, tp_size, sp_size)  # dp8tp2

    dp_group = pg_mesh.get_group_along_axis(dp_axis)
    pg_mesh.get_group_along_axis(pp_axis)
    pg_mesh.get_group_along_axis(tp_axis)
    pg_mesh.get_group_along_axis(sp_axis)

    input_tensor = torch.tensor([rank], dtype=dtype, device=device)
    # 创建一个用于存储 allgather 结果的列表
    output_tensors = [torch.empty(1, dtype=dtype, device=device) for _ in range(dp_size)]
    # 执行 allgather 操作
    dist.all_gather(output_tensors, input_tensor, group=dp_group)
    print(f"input_tensor {input_tensor} output_tensors {output_tensors}")


def test_broadcast():
    acclerate = get_accelerator()
    DistCoordinator()
    device = acclerate.get_current_device()
    rank = dist.get_rank()
    dtype = torch.bfloat16  # fail case: torch.int32, torch.int64
    dp_axis, pp_axis, tp_axis, sp_axis = 0, 1, 2, 3
    dp_size, pp_size, tp_size, sp_size = 16, 1, 1, 1
    dp_size * pp_size * tp_size * sp_size
    pg_mesh = ProcessGroupMesh(dp_size, pp_size, tp_size, sp_size)  # dp8tp2

    dp_group = pg_mesh.get_group_along_axis(dp_axis)
    pg_mesh.get_group_along_axis(pp_axis)
    pg_mesh.get_group_along_axis(tp_axis)
    pg_mesh.get_group_along_axis(sp_axis)

    if rank == 0:
        # 只有 rank 为 0 的进程初始化要广播的张量
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    else:
        # 其他进程创建一个空的张量用于接收广播的数据
        tensor = torch.empty(3, dtype=torch.float32, device=device)

    # 执行 broadcast 操作，将 rank 0 的张量广播到所有进程
    dist.broadcast(tensor, src=0, group=dp_group)

    # 所有进程都应该接收到与 rank 0 相同的张量
    expected_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    print(f"tensor {tensor}")


if __name__ == "__main__":
    colossalai.launch_from_torch()
    # test_reduce_scatter_tensor()
    # test_reduce_scatter()
    # test_all_reduce()
    # test_all_gather()
    test_broadcast()
