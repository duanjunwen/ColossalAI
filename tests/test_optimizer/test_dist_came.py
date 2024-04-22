import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer import CAME, DistributedCAME
from colossalai.tensor.d_tensor import is_distributed_tensor
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.tensor.d_tensor.sharding_spec import DimSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from colossalai.zero import LowLevelZeroOptimizer
from tests.kit.model_zoo import model_zoo
from tests.test_optimizer._utils import check_optim_states, check_dist_optim_state, check_dist_param
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    build_model_from_low_level_zero_plugin,
    check_weight,
    run_forward_backward_with_hybrid_plugin,
    run_forward_backward_with_low_level_zero_plugin,
    unwrap_model,
)

_ALLOWED_P_G_TYPES = [
    (torch.float, torch.float),  # pure fp32
    (torch.float, torch.half),  # fp16 amp
    (torch.float, torch.bfloat16),  # bfloat16 amp
]

# Identifiers for Tensor Parallel linear layers
_SHARD_DIM = DimSpec([0])
_IN_DIM = 32
_HID_DIM = 128
_N_STEP = 3
_SEED = 1024
_COORD = None

Net, data_gen, *_ = next(iter(model_zoo.get_sub_registry("simple_mlp").values()))
TPNet, *_ = next(iter(model_zoo.get_sub_registry("simple_tp_mlp").values()))


def get_split_dim(p):
    if not is_distributed_tensor(p):
        raise ValueError("p is not a distributed tensor")
    sharding = p.dist_layout.sharding_spec.sharding_sequence
    return sharding.index(_SHARD_DIM)


def assert_distributed_close(tp_model, torch_model, rtol, atol, tp_group):
    rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)

    for (name, p), torch_p in zip(tp_model.named_parameters(), torch_model.parameters()):
        # if overflow, the weight won't be updated. so there will be no nan in p
        assert not torch.isnan(p).any()
        try:
            if is_distributed_tensor(p):
                split_dim = get_split_dim(p)
                torch_p = torch_p.chunk(tp_size, dim=split_dim)[rank]

            assert_close(p.float(), torch_p, rtol=rtol, atol=atol)
        except AssertionError as e:
            print(f"grad mismatch in {name}")
            raise e


def setup_param_groups(bert_model: nn.Module) -> list:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def force_assign_grad(p, g_dtype, grad=None):
    """avoid inconsistent grad and param dtype error"""
    orig_p = p.data
    p.data = torch.randn_like(p, device=orig_p.device, dtype=g_dtype) if grad == None else grad
    p.grad = p.data
    p.data = orig_p


def set_dist_grad(
    dist_module: nn.Module,
    torch_model: nn.Module,
    g_dtype: torch.dtype,
    group: dist.ProcessGroup,
) -> None:
    """
    Set grads chunks for Tensor Parallel or ZeRO DP.
    We do not need a separate treatment for ZeRO,
    as the LowLevelOptimizer takes care of reduce-scattering grads.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    for p, torch_p in zip(dist_module.parameters(), torch_model.parameters()):
        if torch_p.grad is None:
            # avoid inconsistent grad and param dtype error
            force_assign_grad(torch_p, g_dtype)
        else:
            torch_p.grad += torch.randn_like(torch_p, device=torch_p.device, dtype=g_dtype)

        if p.grad is None:
            force_assign_grad(p, g_dtype)

        if is_distributed_tensor(p):
            split_dim = get_split_dim(p)
            # Add grads only to the correctly split chunk
            force_assign_grad(p, g_dtype, torch_p.grad.chunk(world_size, dim=split_dim)[rank])
            # assert_close(p.grad, torch_p.grad.chunk(world_size, dim=split_dim)[rank])
        else:
            force_assign_grad(p, g_dtype, torch_p.grad)


@parameterize("p_g_dtype", _ALLOWED_P_G_TYPES)
@parameterize("bias_correction", [False, True])
@parameterize("tp_zero_size", [(1, 4), (4, 1), (2, 2)])
def run_dist_lamb_basic(
    bias_correction: bool, p_g_dtype: tuple[torch.dtype, torch.dtype], tp_zero_size: tuple[int, int]
) -> None:
    """Test without forward"""
    p_dtype, g_dtype = p_g_dtype
    tp_size, zero_size = tp_zero_size

    # Set distributed groups
    rank = dist.get_rank()
    clear_layout_converter()  # Ensure correct sharding
    proc_mesh = ProcessGroupMesh(tp_size, zero_size)
    tp_group = proc_mesh.get_group_along_axis(0)
    dp_group = proc_mesh.get_group_along_axis(1)

    tp_rank = dist.get_rank(tp_group)
    seed_all(_SEED)  # Fix model init
    torch_model = Net(in_dim=_IN_DIM, hid_dim=_HID_DIM, identity=True).to(rank)
    tp_model = TPNet(torch_model.fc0, torch_model.fc1, torch_model.fc2, tp_group).to(rank)
    # Ensure equal weight init
    assert_close(
        torch_model.fc1.weight[tp_rank * _HID_DIM // tp_size : (tp_rank + 1) * _HID_DIM // tp_size],
        tp_model.fc1.weight,
    )
    assert_close(
        torch_model.fc2.weight[:, tp_rank * _HID_DIM // tp_size : (tp_rank + 1) * _HID_DIM // tp_size],
        tp_model.fc2.weight,
    )

    # Set up optimizers
    lr = 1e-3
    beta1, beta2, beta3 = 0.9, 0.999, 0.9999
    eps = (1e-30, 1e-16)
    torch_optim = CAME(setup_param_groups(torch_model), lr=lr, betas=(beta1, beta2, beta3), eps=eps)
    optim = DistributedCAME(
        setup_param_groups(tp_model),
        lr=lr,
        betas=(beta1, beta2, beta3),
        eps=eps,
    )
    optim.setup_distributed(tp_group, dp_group, None, zero_flag=False)

    rtol, atol = 1e-4, 1e-4
    if p_dtype is torch.float16 or g_dtype is torch.float16:
        rtol, atol = 1e-3, 1e-3
    if p_dtype is torch.bfloat16 or g_dtype is torch.bfloat16:
        rtol, atol = 2e-3, 2e-3

    for i in range(_N_STEP):
        seed_all(_SEED + i)  # NOTE: having only one manual_seed above doesn't work?
        set_dist_grad(tp_model, torch_model, g_dtype, tp_group)

        torch_optim.step()
        optim.step()
        torch_optim.zero_grad()
        optim.zero_grad()
        try:
            assert_distributed_close(tp_model, torch_model, rtol, atol, tp_group)
        except Exception as e:
            _COORD.print_on_master(
                f"step {i + 1}: bias_correction: {bias_correction}, p_g_dtype: {p_g_dtype}, tp_zero_size: {tp_zero_size}"
            )
            raise e


@parameterize("p_g_dtype", _ALLOWED_P_G_TYPES)
@parameterize("bias_correction", [False, True])
@parameterize("tp_zero_size", [(2, 2), (4, 1), (1, 4)])
def run_dist_lamb_fwd_bwd(
    bias_correction: bool, p_g_dtype: tuple[torch.dtype, torch.dtype], tp_zero_size: tuple[int, int]
) -> None:
    p_dtype, g_dtype = p_g_dtype
    tp_size, zero_size = tp_zero_size

    # Set distributed groups
    rank = dist.get_rank()
    proc_mesh = ProcessGroupMesh(tp_size, zero_size)
    tp_group = proc_mesh.get_group_along_axis(0)
    dp_group = proc_mesh.get_group_along_axis(1)
    tp_rank = dist.get_rank(tp_group)

    seed_all(_SEED)
    clear_layout_converter()  # Ensure correct sharding
    torch_model = Net(_IN_DIM, _HID_DIM).to(rank)
    tp_model = TPNet(torch_model.fc0, torch_model.fc1, torch_model.fc2, tp_group).to(rank)

    assert_close(
        torch_model.fc1.weight[tp_rank * _HID_DIM // tp_size : (tp_rank + 1) * _HID_DIM // tp_size],
        tp_model.fc1.weight,
    )
    assert_close(
        torch_model.fc2.weight[:, tp_rank * _HID_DIM // tp_size : (tp_rank + 1) * _HID_DIM // tp_size],
        tp_model.fc2.weight,
    )

    # Set up optimizers
    lr = 1e-3
    beta1, beta2, beta3 = 0.9, 0.999, 0.9999
    eps = (1e-30, 1e-16)
    torch_optim = CAME(setup_param_groups(torch_model), lr=lr, betas=(beta1, beta2, beta3), eps=eps)
    optim = DistributedCAME(
        setup_param_groups(tp_model),
        lr=lr,
        betas=(beta1, beta2, beta3),
        eps=eps,
    )

    # Setup distributed optimizer
    if zero_size > 1:
        optim = LowLevelZeroOptimizer(
            optim,
            overlap_communication=True,
            initial_scale=128,
            partition_grad=True,
            dp_process_group=dp_group,
            verbose=True,
        )
        shard_to_param = optim._param_store.master_to_working_param
        optim.optim.setup_distributed(tp_group, dp_group, shard_to_param, zero_flag=True)
    else:
        optim.setup_distributed(tp_group, dp_group, None, zero_flag=False)

    rtol, atol = 1e-4, 1e-4
    if p_dtype is torch.float16 or g_dtype is torch.float16:
        rtol, atol = 1e-3, 1e-3
    if p_dtype is torch.bfloat16 or g_dtype is torch.bfloat16:
        rtol, atol = 2e-3, 2e-3

    seed_all(_SEED)  # NOTE: having only one manual_seed above doesn't work?
    x = data_gen()
    x = x.cuda().to(dtype=p_dtype)

    out_tp = tp_model(x)
    out = torch_model(x)
    try:
        assert_close(out, out_tp, rtol=rtol, atol=atol)
    except Exception as e:
        _COORD.print_on_master(
            f"bias_correction: {bias_correction}, p_g_dtype: {p_g_dtype}, tp_zero_size: {tp_zero_size}"
        )
        raise e

    if zero_size > 1:
        optim.backward(out_tp.sum())
        out.sum().backward()
    else:
        out_tp.sum().backward()
        out.sum().backward()

    torch_optim.step()
    optim.step()
    dist.barrier()
    torch_optim.zero_grad()
    optim.zero_grad()
    try:
        assert_distributed_close(tp_model, torch_model, rtol, atol, tp_group)
        check_optim_states(getattr(torch_optim, "optim", torch_optim), getattr(optim, "optim", optim))
    except Exception as e:
        _COORD.print_on_master(
            f"bias_correction: {bias_correction}, p_g_dtype: {p_g_dtype}, tp_zero_size: {tp_zero_size}"
        )
        raise e


@parameterize(
    "test_config",
    [
        {
            "stage": 1,
            "precision": "bf16",
        },
        {
            "stage": 2,
            "precision": "bf16",
        },
    ],
)
def exam_bert_test_on_lowlevelzero_plugin(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")
    model_list = [
        "transformers_bert",
        "transformers_bert_for_pretraining",
        "transformers_bert_lm_head_model",
        "transformers_bert_for_masked_lm",
        "transformers_bert_for_sequence_classification",
        "transformers_bert_for_token_classification",
        "transformers_bert_for_next_sentence",
        "transformers_bert_for_mcq",
        "transformers_bert_for_question_answering",
    ]
    clear_layout_converter()
    torch.set_default_dtype(torch.bfloat16)
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name in model_list:
            (
                org_model,
                org_optimizer,
                sharded_model,
                sharded_optimizer,
                criterion,
                booster,
            ) = build_model_from_low_level_zero_plugin(model_fn, loss_fn, test_config, CAME, DistributedCAME)

            org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_low_level_zero_plugin(
                org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
            )

            weight_layer_for_check = [
                "bert.encoder.layer.0.output.dense.weight",
                "bert.encoder.layer.0.output.dense.weight",
            ]

            org_optimizer.step()
            sharded_optimizer.step()

            # check weights
            if test_config["precision"] == "bf16":
                atol, rtol = 5e-4, 5e-4
            else:
                atol, rtol = 5e-4, 5e-4

            check_dist_param(org_model, sharded_model, weight_layer_for_check, atol, rtol)
            check_optim_states(org_optimizer, sharded_optimizer.optim)

    Randomizer.reset_index()
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {
            "tp_size": 1,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "bf16",
        },
        {
            "tp_size": 2,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "bf16",
        },
        {
            "tp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "bf16",
        },
        {
            "tp_size": 2,
            "num_microbatches": 4,
            "zero_stage": 1,
            "precision": "bf16",
        },
        {
            "tp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 0,
            "precision": "bf16",
        },
    ],
)
def exam_bert_test_on_hybrid_plugin(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")
    test_config["use_lazy_init"] = False
    test_config["pp_size"] = 1  # Do NOT test Pipeline Parallel
    test_config["initial_scale"] = 2**16  # avoid overflow
    model_list = [
        "transformers_bert",
        # "transformers_bert_for_pretraining",
        # "transformers_bert_lm_head_model",
        # "transformers_bert_for_masked_lm",
        # "transformers_bert_for_sequence_classification",
        # "transformers_bert_for_token_classification",
        # "transformers_bert_for_next_sentence",
        # "transformers_bert_for_mcq",
        # "transformers_bert_for_question_answering",
    ]
    clear_layout_converter()
    torch.set_default_dtype(torch.bfloat16)
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name in model_list:
            (
                org_model,
                org_optimizer,
                sharded_model,
                sharded_optimizer,
                criterion,
                booster,
            ) = build_model_from_hybrid_plugin(model_fn, loss_fn, test_config, CAME, DistributedCAME)

            org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
                org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
            )

            stage_manager = booster.plugin.stage_manager
            tp_group = booster.plugin.tp_group

            bert = unwrap_model(org_model, "BertModel", "bert")
            sharded_bert = unwrap_model(sharded_model, "BertModel", "bert")
            # weight_layer_for_check = ["encoder.layer[0].output.dense", "encoder.layer[1].output.dense"]
            weight_layer_for_check = ["encoder.layer.0.output.dense", "encoder.layer.1.output.dense"]

            
            org_optimizer.step()
            sharded_optimizer.step()

            # check weights
            if test_config["precision"] == "bf16":
                atol, rtol = 5e-4, 5e-4
            else:
                atol, rtol = 5e-4, 5e-4
            if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
                # for (org_name, org_param), (shard_name, shard_param) in zip(bert.named_parameters(), sharded_bert.named_parameters()):
                    # print(org_name, shard_name)
                    # if org_name in weight_layer_for_check:
                        # print(f"org_name {org_name} shape {org_param.shape} {org_param}\n sharded_name {shard_name} shape {shard_param.shape} {shard_param}\n")
                        # assert_close(org_param, shard_param, atol=atol, rtol=rtol)
                 check_weight(bert, sharded_bert, weight_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1)
                # check optim states
                # check_dist_optim_state(org_optimizer, sharded_optimizer.optim)

    Randomizer.reset_index()
    torch.cuda.empty_cache()


def check_dist_lamb(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    global _COORD
    _COORD = DistCoordinator()

    # run_dist_lamb_basic()
    # _COORD.print_on_master("Basic tests passed")

    # run_dist_lamb_fwd_bwd()
    # _COORD.print_on_master("Forward-backward tests passed")

    exam_bert_test_on_lowlevelzero_plugin()
    _COORD.print_on_master("LowLevelZeroPlugin + Bert Model Zoo tests passed")

    # exam_bert_test_on_hybrid_plugin()
    # _COORD.print_on_master("HybridParallelPlugin + Bert Model Zoo tests passed")


    # run_bert_test(optim_class=CAME, sharded_optim_class=DistributedCAME)
    print(f"rank {rank} tests passed :)")


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_lamb():
    spawn(check_dist_lamb, nprocs=4)


if __name__ == "__main__":
    test_dist_lamb()
