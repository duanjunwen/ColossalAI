import json

import torch
import torch.distributed as dist
from coati.dataset.loader import RawConversationDataset
from coati.utils.compare_tool import open_module_tracker
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, Plugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing.random import seed_all

BATCH_SIZE = 1
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 1
DATA_PATH = "/home/duanjunwen/datasets/math_dataset_profile.jsonl"  # math_dataset_profile.jsonl math_dataset.jsonl
MODEL_PATH = "/home/duanjunwen/models/Qwen/Qwen2.5-3B"


class RandomDataset(Dataset):
    def __init__(self, num_samples, sequence_length, vocab_size=10000):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.input_idx = torch.randint(0, vocab_size, (num_samples, sequence_length))
        self.attention_mask = torch.randint(0, 2, (num_samples, sequence_length), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"input_ids": self.input_idx[idx], "attention_mask": self.attention_mask[idx]}


def load_model_and_tokenizer():
    attn_impl = "eager"
    # if get_accelerator().name == "cuda" else "flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model = Qwen2ForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return tokenizer, model


def all_reduce_mean(loss: torch.Tensor, plugin: Plugin) -> torch.Tensor:
    loss = loss.data
    group = getattr(plugin, "dp_group", None)
    dist.all_reduce(loss, group=group)
    return loss / dist.get_world_size(group)


def test_hybrid_qwen(device: str = "cpu"):
    colossalai.launch_from_torch()
    get_accelerator()
    coordinator = DistCoordinator()
    tokenizer, model = load_model_and_tokenizer()
    seed_all(42)
    # dataset = RandomDataset(num_samples=100, sequence_length=2304)
    dataset = RawConversationDataset(tokenizer, DATA_PATH, 1024)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        precision="bf16",
        zero_stage=2,
        cpu_offload=True,
    )
    # plugin = HybridParallelPlugin(tp_size=2, pp_size=2, precision="bf16", zero_stage=1, num_microbatches=4, enable_flash_attention=True)

    dataloader = plugin.prepare_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )

    booster = Booster(plugin=plugin)
    open_module_tracker(model)
    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, None, dataloader)

    def is_master():
        if isinstance(plugin, HybridParallelPlugin) and plugin.pp_size > 1:
            return coordinator.rank == coordinator.world_size - 1
        return coordinator.is_master()

    #####
    # train
    #####
    model.train()
    loss_dict = {}
    # if not os.path.exists("./tests/tensor_log/"):
    #     os.makedirs("./tests/tensor_log/")
    for epoch in range(NUM_EPOCHS):
        if booster.plugin.pp_size > 1:
            data_iter = iter(dataloader)
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Step",
                disable=not is_master(),
            )
            for step in step_bar:
                print(f"data_iter {data_iter}")
                outputs = booster.execute_pipeline(
                    data_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs[0],
                    optimizer=optimizer,
                    return_loss=True,
                )
                loss = outputs["loss"]
                if booster.plugin.stage_manager.is_last_stage():
                    global_loss = all_reduce_mean(loss, plugin)

                optimizer.step()

                if booster.plugin.stage_manager.is_last_stage():
                    grad_norm = optimizer.get_grad_norm()
                    step_bar.set_postfix({"loss": global_loss.item(), "grad_norm": grad_norm})

                optimizer.step()
                optimizer.zero_grad()
        else:
            total_loss = 0
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device=model.module.device)
                attention_mask = batch["attention_mask"].to(device=model.module.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                # tensor_pt = {
                #     "input_ids": input_ids.to("cpu"),
                #     "attention_mask": attention_mask.to("cpu"),
                #     "logits": outputs["logits"].to("cpu"),
                # }
                # torch.save(tensor_pt, f"./tests/tensor_log/{device}/tensor_rank{dist.get_rank()}_step{step}.pt")
                # print(f"step {step} rank {dist.get_rank()} : loss {loss}")
                loss_value = loss.detach().cpu().item()
                loss_dict[step] = loss_value
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                booster.backward(loss, optimizer)
                # print(f"finish backward")
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    # print(f"finish optimizer step")

                total_loss += loss.item()
            # 将字典保存为 JSON 文件
            if dist.get_rank() == 0:
                with open("loss_dict.json", "w") as f:
                    json.dump(loss_dict, f)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


def test_hybrid_qwen_fwd(device: str = "cpu"):
    colossalai.launch_from_torch()
    get_accelerator()
    coordinator = DistCoordinator()
    tokenizer, model = load_model_and_tokenizer()
    seed_all(42)
    # dataset = RandomDataset(num_samples=100, sequence_length=2304)
    dataset = RawConversationDataset(tokenizer, DATA_PATH, 1024)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        precision="bf16",
        zero_stage=2,
        cpu_offload=True,
    )
    # plugin = HybridParallelPlugin(tp_size=2, pp_size=2, precision="bf16", zero_stage=1, num_microbatches=4, enable_flash_attention=True)

    dataloader = plugin.prepare_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )

    booster = Booster(plugin=plugin)
    open_module_tracker(model)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, None, dataloader)
    model.eval()

    def is_master():
        if isinstance(plugin, HybridParallelPlugin) and plugin.pp_size > 1:
            return coordinator.rank == coordinator.world_size - 1
        return coordinator.is_master()

    #####
    # train
    #####
    model.eval()
    loss_dict = {}
    with torch.no_grad():
        for epoch in range(NUM_EPOCHS):
            if booster.plugin.pp_size > 1:
                data_iter = iter(dataloader)
                step_bar = tqdm(
                    range(len(dataloader)),
                    desc="Step",
                    disable=not is_master(),
                )
                for step in step_bar:
                    print(f"data_iter {data_iter}")
                    outputs = booster.execute_pipeline(
                        data_iter,
                        model,
                        criterion=lambda outputs, inputs: outputs[0],
                        optimizer=optimizer,
                        return_loss=True,
                    )
                    loss = outputs["loss"]
            else:
                total_loss = 0
                for step, batch in enumerate(dataloader):
                    input_ids = batch["input_ids"].to(device=model.module.device)
                    attention_mask = batch["attention_mask"].to(device=model.module.device)
                    # input_ids = torch.arange(0, 1024).unsqueeze(0).to(device=model.module.device)
                    # attention_mask =  torch.zeros((1, 1024)).to(dtype=torch.int64, device=model.module.device)
                    # print(f"input_ids {input_ids} attention_mask {attention_mask}")
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    # tensor_pt = {
                    #     "input_ids": input_ids.to("cpu"),
                    #     "attention_mask": attention_mask.to("cpu"),
                    #     "logits": outputs["logits"].to("cpu"),
                    # }
                    # torch.save(tensor_pt, f"./tests/tensor_log/{device}_tensor_rank{dist.get_rank()}_step{step}.pt")
                    # print(f"step {step} rank {dist.get_rank()} : loss {loss}")
                    total_loss += loss.item()
                # 将字典保存为 JSON 文件
                if dist.get_rank() == 0:
                    with open("loss_dict.json", "w") as f:
                        json.dump(loss_dict, f)
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    dtype = torch.bfloat16
    test_hybrid_qwen(device)
    # test_hybrid_qwen_fwd(device)
