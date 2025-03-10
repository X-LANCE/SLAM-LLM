import torch
import torch_npu
import deepspeed
import torch.distributed as dist
import os
# dist.init_process_group(
#         backend='hccl',    # 使用NCCL后端（GPU场景）
#     )
# deepspeed.init_distributed(
#     dist_backend='hccl',    # 使用NCCL后端（GPU场景）
# )
local_rank = os.environ["LOCAL_RANK"]
torch.npu.set_device(f"npu:{local_rank}")  # 绑定当前NPU
deepspeed.init_distributed(
dist_backend='hccl',    # 使用NCCL后端（GPU场景）
)
tensor = torch.tensor([1.0]).npu()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"Rank {dist.get_rank()}: {tensor.item()}")