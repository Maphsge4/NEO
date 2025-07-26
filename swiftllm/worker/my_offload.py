# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from builtins import isinstance
import functools
import time
import logging
from typing import Any, List, Tuple, Optional, Union

import torch
from torch import nn
import logging
import torch.cuda.nvtx as nvtx
import copy
import gc

def _conditional_amp_fwd_decorator(orig_func):  # type: ignore

    if hasattr(torch.cuda.amp, "custom_fwd"):
        return torch.cuda.amp.custom_fwd(orig_func)  # type: ignore

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) -> Any:
        return orig_func(*args, **kwargs)

    return inner_decorator


def _conditional_amp_bwd_decorator(orig_func):  # type: ignore
    if hasattr(torch.cuda.amp, "custom_bwd"):
        return torch.cuda.amp.custom_bwd(orig_func)  # type: ignore

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) -> Any:
        return orig_func(*args, **kwargs)

    return inner_decorator


def _split(modules: nn.Sequential, number_splits: int) -> List[List[nn.Module]]:
    number_splits = min(len(modules), number_splits)
    splits: List[List[nn.Module]] = [[] for _ in range(number_splits)]

    # Count the number of parameters per exposed layer, use that as a proxy for memory footprint
    total_number_params = sum([sum(p.numel() for p in m.parameters()) for m in modules])
    number_parameters_per_shard = total_number_params // number_splits

    current_shard = -1

    # maphsge4 add
    # print(f"This model has {total_number_params/1e6:.2f}M parameters, aiming for {number_parameters_per_shard/1e6:.2f}M parameters per shard")
    logging.info(
        f"This model has {total_number_params / 1e6:.2f}M parameters, aiming for {number_parameters_per_shard / 1e6:.2f}M parameters per shard"
    )

    for m in modules:
        for p in m.parameters():
            p.data = p.data.pin_memory()
        # Number of parameters in the current shard
        current_shard_params = sum(p.numel() for sm in splits[current_shard] for p in sm.parameters())

        # # This shard is big enough, point to the next one
        # if (
        #         current_shard_params > 0
        #         and current_shard_params + sum(p.numel() for p in m.parameters()) > number_parameters_per_shard
        #         and current_shard < number_splits - 1
        # ):
        #     current_shard += 1
        current_shard += 1

        splits[current_shard].append(m)

    for i, split in enumerate(splits):
        current_shard_params = sum(p.numel() for sm in split for p in sm.parameters())
        logging.info(f"Shard {i} holds {current_shard_params / 1e6:.2f}M parameters")

    return splits

model_shard_backup : List[nn.Module] = []


class ModelShard(nn.Module):
    """
    Wrap one shard of the model, make it possible to load parameters on the
    fly for the FW and BW pass on the given device.
    """

    def __init__(
            self,
            cpu_model_shard: nn.Module,
            device: torch.device,
            offload_device: torch.device,
            index: int,
            g_stream,
            c_stream
    ):
        super().__init__()
        self.model_shard = cpu_model_shard
        self.kv_cache = None
        self.index = index

        # Save all the parameter sizes to be able to restore them
        self.device = device
        torch.cuda.device(self.device)

        self.offload_device = offload_device

        # self.model_shard.to(offload_device)
        # self._cpu_to_gpu_stream = torch.cuda.Stream(device=self.device)  # 两个stream流
        # self._gpu_to_cpu_stream = torch.cuda.Stream(device=self.device)  # 两个stream流
        self._cpu_to_gpu_stream = c_stream
        self._gpu_to_cpu_stream = g_stream

    def forward(self, *inputs, **_):  # type: ignore  # maphsge4 add args
        # print(f"model shard forward max:", torch.cuda.max_memory_allocated(device=torch.device("cuda")))  # 显存量
        if isinstance(inputs, tuple):
            return self.model_shard.forward(
                *inputs,
                **_
            )
        else:
            return self.model_shard(inputs)

    def to(self, device: torch.device) -> "ModelShard":  # type: ignore
        # Make sure that the lookahead and lookback shards are not captured by this call
        self.model_shard.to(device)
        return self

    def train(self, mode: bool = True) -> "ModelShard":
        # Make sure that the lookahead and lookback shards are not captured by this call
        self.model_shard.train(mode)
        return self

    def to_device(self) -> None:
        self.model_shard.to(device=self.device, non_blocking=True)

    def get_or_create_cpu_backup(self, tensor, attr_name):
        """获取或创建CPU备份"""
        cpu_backup_attr = f"_cpu_backup_{attr_name}"
        
        if not hasattr(self.model_shard.weight, cpu_backup_attr):
            # 第一次访问，创建CPU备份
            assert tensor.device.type == 'cpu'
            cpu_backup = tensor.clone()
            cpu_backup = cpu_backup.pin_memory() 
            setattr(self.model_shard.weight, cpu_backup_attr, cpu_backup)
        
        return getattr(self.model_shard.weight, cpu_backup_attr)

    def create_cpu_backup(self, tensor, attr_name):
        """创建CPU备份"""
        cpu_backup_attr = f"_cpu_backup_{attr_name}"
        
        if not hasattr(self.model_shard.weight, cpu_backup_attr):
            # 第一次访问，创建CPU备份
            assert tensor.device.type == 'cpu'
            cpu_backup = tensor.clone()
            cpu_backup = cpu_backup.pin_memory() 
            setattr(self.model_shard.weight, cpu_backup_attr, cpu_backup)
        
        return getattr(self.model_shard.weight, cpu_backup_attr)
    
    def get_cpu_backup(self, attr_name):
        """获取CPU备份"""
        cpu_backup_attr = f"_cpu_backup_{attr_name}"
        
        return getattr(self.model_shard.weight, cpu_backup_attr)
    
    def copy_from_cpu_to_gpu(self, cpu_tensor, target_device, non_blocking=False):
        """从CPU复制到GPU"""
        if target_device.type == 'cpu':
            return cpu_tensor
        
        gpu_tensor = torch.empty_like(cpu_tensor, device=target_device)
        gpu_tensor.copy_(cpu_tensor, non_blocking=non_blocking)
        return gpu_tensor

    def init_percentage_load(self, percentage: float = 0.7, non_blocking: bool = True) -> None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            nvtx.range_push("forward_load")
            # self.model_shard.to(self.device, non_blocking=non_blocking)
            flag = True
            for item in self.model_shard.weight.registered_weights:
                if (item.attr_name == "up_proj" or item.attr_name == "gate_proj"):
                    if not flag:
                        continue

                    tensor = getattr(self.model_shard.weight, "up_gate_proj")
                    total_size = tensor.shape[0]
                    split_point = int(total_size * percentage)  # 70%的位置
                    part1, part2 = torch.split(tensor, [split_point, total_size - split_point], dim=0)
                    
                    # 获取或创建CPU备份
                    cpu_backup = self.get_or_create_cpu_backup(part2, "up_gate_proj")
                    assert cpu_backup.is_pinned(), "CPU backup tensor must be pinned memory"
                    
                    # 从CPU复制到目标设备
                    new_tensor = [part1.cuda(), part2]
                    setattr(self.model_shard.weight, "up_gate_proj", new_tensor)
                    
                    flag = False  # 只处理一次up_gate_proj
                else:
                    tensor = getattr(self.model_shard.weight, item.attr_name)
                    total_size = tensor.shape[0]
                    split_point = int(total_size * percentage)  # 70%的位置
                    part1, part2 = torch.split(tensor, [split_point, total_size - split_point], dim=0)

                    # 获取或创建CPU备份
                    cpu_backup = self.get_or_create_cpu_backup(part2, item.attr_name)
                    assert cpu_backup.is_pinned(), "CPU backup tensor must be pinned memory"

                    # 从CPU复制到目标设备
                    new_tensor = [part1.cuda(), part2]
                    setattr(self.model_shard.weight, item.attr_name, new_tensor)

            nvtx.range_pop()

    def forward_percentage_load(self, non_blocking: bool = True) -> None:
        # print("forward load start", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            # time1 = time.time()
            nvtx.range_push("forward_load")
            flag = True
            for item in self.model_shard.weight.registered_weights:
                if item.attr_name == "up_proj" or item.attr_name == "gate_proj":
                    if not flag:
                        continue

                    tensor_list = getattr(self.model_shard.weight, "up_gate_proj")

                    # 获取CPU备份
                    cpu_backup = self.get_cpu_backup("up_gate_proj")
                    assert cpu_backup.is_pinned(), "CPU backup tensor must be pinned memory"

                    # 从CPU复制到目标设备
                    part2 = self.copy_from_cpu_to_gpu(cpu_backup, self.device, non_blocking)
                    new_tensor = [tensor_list[0], part2]
                    new_tensor = torch.cat(new_tensor, dim=0)
                    setattr(self.model_shard.weight, "up_gate_proj", new_tensor)

                    flag = False  # 只处理一次up_gate_proj
                else:
                    tensor_list = getattr(self.model_shard.weight, item.attr_name)

                    # 获取CPU备份
                    cpu_backup = self.get_cpu_backup(item.attr_name)
                    assert cpu_backup.is_pinned(), "CPU backup tensor must be pinned memory"

                    # 从CPU复制到目标设备
                    part2 = self.copy_from_cpu_to_gpu(cpu_backup, self.device, non_blocking)
                    new_tensor = [tensor_list[0], part2]
                    new_tensor = torch.cat(new_tensor, dim=0)
                    setattr(self.model_shard.weight, item.attr_name, new_tensor)
            if self.kv_cache is not None:
                self.kv_cache = self.kv_cache.to(self.device, non_blocking=non_blocking)
            nvtx.range_pop()
            # time2 = time.time()
            # print(f"forward load time: {time2 - time1}")
        # print("forward load end", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量

    def forward_load(self, non_blocking: bool = True) -> None:
        # print("forward load start", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            # time1 = time.time()
            nvtx.range_push("forward_load")
            # self.model_shard.to(self.device, non_blocking=non_blocking)
            for item in self.model_shard.weight.registered_weights:
                if item.attr_name == "up_proj" or item.attr_name == "gate_proj":
                    tensor = getattr(self.model_shard.weight, "up_gate_proj")
                    # 获取或创建CPU备份
                    cpu_backup = self.get_or_create_cpu_backup(tensor, "up_gate_proj")
                    assert cpu_backup.is_pinned(), "CPU backup tensor must be pinned memory"
                    # 从CPU复制到目标设备
                    new_tensor = self.copy_from_cpu_to_gpu(cpu_backup, self.device, non_blocking)
                    setattr(self.model_shard.weight, "up_gate_proj", new_tensor)
                else:
                    tensor = getattr(self.model_shard.weight, item.attr_name)
                    # 获取或创建CPU备份
                    cpu_backup = self.get_or_create_cpu_backup(tensor, item.attr_name)
                    assert cpu_backup.is_pinned(), "CPU backup tensor must be pinned memory"
                    # 从CPU复制到目标设备
                    new_tensor = self.copy_from_cpu_to_gpu(cpu_backup, self.device, non_blocking)
                    setattr(self.model_shard.weight, item.attr_name, new_tensor)
            if self.kv_cache is not None:
                self.kv_cache = self.kv_cache.to(self.device, non_blocking=non_blocking)
            nvtx.range_pop()
            # time2 = time.time()
            # print(f"forward load time: {time2 - time1}")
        # print("forward load end", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量

    # Ignore the following function for code coverage since the backward pass
    # is triggered by C++ code and cannot be calculated when overriding
    # autograd.Function
    def backward_load(self, non_blocking: bool = True) -> None:  # pragma: no cover
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard.to(self.device, non_blocking=non_blocking)

    def forward_percentage_drop(self, percentage: float = 0.7, non_blocking: bool = True) -> None:
        # print("forward drop start", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量
        nvtx.range_push("forward_drop")
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            # self.model_shard.to(self.offload_device, non_blocking=non_blocking)
            flag = True
            for item in self.model_shard.weight.registered_weights:
                if item.attr_name == "up_proj" or item.attr_name == "gate_proj":
                    if not flag:
                        continue

                    tensor = getattr(self.model_shard.weight, "up_gate_proj")
                    total_size = tensor.shape[0]
                    split_point = int(total_size * percentage)  # 70%的位置
                    part1, part2 = torch.split(tensor, [split_point, total_size - split_point], dim=0)
                    del part2

                    setattr(self.model_shard.weight, "up_gate_proj", [part1, None])

                    flag = False  # 只处理一次up_gate_proj
                else:
                    tensor = getattr(self.model_shard.weight, item.attr_name)
                    total_size = tensor.shape[0]
                    split_point = int(total_size * percentage)  # 70%的位置
                    part1, part2 = torch.split(tensor, [split_point, total_size - split_point], dim=0)
                    del part2

                    setattr(self.model_shard.weight, item.attr_name, [part1, None])

            # # 强制垃圾回收
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            if self.kv_cache is not None:
                self.kv_cache = self.kv_cache.to(self.offload_device, non_blocking=non_blocking)
        nvtx.range_pop()
        # print("forward drop end", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量

    def forward_drop(self, non_blocking: bool = True) -> None:
        # print("forward drop start", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量
        nvtx.range_push("forward_drop")
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            # self.model_shard.to(self.offload_device, non_blocking=non_blocking)
            for item in self.model_shard.weight.registered_weights:
                if item.attr_name == "up_proj" or item.attr_name == "gate_proj":
                    # 保留属性名，但设置张量为None
                    if hasattr(self.model_shard.weight, "up_gate_proj"):
                        setattr(self.model_shard.weight, "up_gate_proj", None)
                else:
                    # 保留属性名，但设置张量为None
                    if hasattr(self.model_shard.weight, item.attr_name):
                        setattr(self.model_shard.weight, item.attr_name, None)

            # # 强制垃圾回收
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            if self.kv_cache is not None:
                self.kv_cache = self.kv_cache.to(self.offload_device, non_blocking=non_blocking)
        nvtx.range_pop()
        # print("forward drop end", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量

    # Ignore the following function for code coverage since the backward pass
    # is triggered by C++ code and cannot be calculated when overriding
    # autograd.Function
    def backward_drop(self, non_blocking: bool = True) -> None:  # pragma: no cover
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard.to(self.offload_device, non_blocking=non_blocking)


class OffloadFunction(torch.autograd.Function):  # torch.autograd.Function 要求 forward传了几个参数，backward就得传回几个
    """
    This Function enables checkpointing of intermediate activations at
    shard boundaries by overriding the forward and backward pass of the nn.Module.

    - In the FW pass, it drops parameters in the previous shard and
    loads parameters for the next shard. No graph is constructed in the FW pass.
    This enables us to offload intermediate activations present at the shard
    boundaries.

    - In the BW pass, it does the reverse. We run the forward pass using the
    saved intermediate activations and calculate gradients as needed.
    The trade-off is latency vs memory when using activation checkpointing.

    - Follows heavily from https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html#checkpoint.

    NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    @_conditional_amp_fwd_decorator  # type: ignore
    def forward(ctx: Any, inputs: Any, dummy_input: Any, model_instance: Any, **_: Any) -> Any:

        inputs = inputs if isinstance(inputs, tuple) else (inputs,)

        ctx.inputs = inputs
        ctx.model_instance = model_instance
        # TODO(anj-s): We might need to store this for each boundary activation.
        # Currently we assume all boundary activation inputs require
        ctx.grad_requirements = tuple(x.requires_grad for x in inputs)
        ctx.fwd_rng_state = torch.get_rng_state()

        # List of input activations starting with the given input.
        model_instance._activations = [inputs]
        # Enumerate through layer shards and apply activations from the previous shard.
        for index, layer_shard in enumerate(model_instance.model_slices):
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:forward_load"):
                # Bring in the current activations onto the device.
                model_instance._activations[index] = tuple([a.cuda() for a in list(model_instance._activations[index])])
                # Bring in the current layer shard onto the device.
                layer_shard.forward_load()

            # Apply the FP and store the activations on the CPU.
            inputs = model_instance._activations[index]
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:no_grad_forward_pass"):
                with torch.no_grad():
                    output_list: List[Any] = []
                    for given_input in inputs:
                        given_input_list = torch.chunk(given_input, model_instance._num_microbatches)
                        given_output_list = []
                        for inputs in given_input_list:
                            # output = layer_shard(inputs)
                            output = layer_shard(
                                inputs,
                                # layer_past=layer_past,
                                # attention_mask=attention_mask,
                                # head_mask=head_mask[i],
                                # encoder_hidden_states=encoder_hidden_states,
                                # encoder_attention_mask=encoder_attention_mask,
                                # use_cache=use_cache,
                                # output_attentions=output_attentions,
                            )
                            given_output_list.append(output)
                        given_output = torch.cat(given_output_list[0]).squeeze(-1)  # [0]
                        output_list.append(given_output)
                    output = tuple(output_list)

            output = output if isinstance(output, tuple) else (output,)
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:forward_drop"):
                # Move the activation used back for the curent shard back to the CPU.
                model_instance._activations[index] = tuple([a.cpu() for a in list(model_instance._activations[index])])
                # The newly computed activations remain on the GPU ready for the next shard computation.
                model_instance._activations.append(output)
                # Move the layer shard back to the CPU.
                layer_shard.forward_drop()

        # The last instance will lose the gradient function if we move it to the CPU.
        # This is because all grad function are present on the device that ran the FW pass.
        # The last activation remains on the GPU and is the return value of this function.
        # Note that this assumes that the target is also on the GPU which is required for calculating
        # the loss.
        result = model_instance._activations[-1]
        result = [r.cuda() for r in result]
        for r in result:
            r.requires_grad = True
        return result[0] if len(result) == 1 else result

    # Ignore the following function for code coverage since the backward pass
    # is triggered by C++ code and cannot be calculated when overriding
    # autograd.Function
    @staticmethod
    @_conditional_amp_bwd_decorator
    def backward(ctx, *grad_outputs):  # type: ignore # pragma: no cover
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.inputs
        model_instance = ctx.model_instance

        for i, need_grad in enumerate(ctx.grad_requirements):
            inputs[i].requires_grad = need_grad

        all_grads = [grad_outputs]

        for model_shard, activation in zip(
                reversed(model_instance.model_slices), reversed(model_instance._activations[:-1])
        ):
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:backward_load"):
                # Move the activation to the GPU.
                activation = tuple([a.cuda() for a in list(activation)])

                # Move the model shard to the GPU.
                model_shard.backward_load()

            # Store the BW pass state.
            bwd_rng_state = torch.get_rng_state()

            # TODO(anj-s): Why detach inputs?
            activation = torch.utils.checkpoint.detach_variable(activation)
            # Get the last gradient calculation.
            final_grads = all_grads[-1]

            if isinstance(activation, torch.Tensor):
                activation = (activation,)
            if isinstance(final_grads, torch.Tensor):
                final_grads = (final_grads,)
            # Iterate through all the inputs/outputs of a shard (there could be multiple).
            chunked_grad_list: List[Any] = []
            # Chunk the activation and grad based on the number of microbatches that are set.
            for chunked_activation, chunked_grad in zip(
                    torch.chunk(*activation, model_instance._num_microbatches),  # type: ignore
                    torch.chunk(*final_grads, model_instance._num_microbatches),  # type: ignore
            ):
                # Set the states to what it used to be before the forward pass.
                torch.set_rng_state(ctx.fwd_rng_state)

                if isinstance(chunked_activation, torch.Tensor):
                    chunked_activation = (chunked_activation,)  # type: ignore
                if isinstance(chunked_grad, torch.Tensor):
                    chunked_grad = (chunked_grad,)  # type: ignore

                # Since we need a grad value of a non leaf element we need to set these properties.
                for a in chunked_activation:
                    if a.dtype == torch.long:
                        continue
                    a.requires_grad = True
                    a.retain_grad()

                with torch.autograd.profiler.record_function(
                        "fairscale.experimental.nn.offload:forward_pass_with_enable_grad"
                ):
                    with torch.enable_grad():
                        # calculate the output of the last shard wrt to the stored activation at the slice boundary.
                        outputs = model_shard(*chunked_activation)

                # Set the states back to what it was at the start of this function.
                torch.set_rng_state(bwd_rng_state)
                with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:backward_pass"):
                    torch.autograd.backward(outputs, chunked_grad)
                intermediate_grads = []
                for a in chunked_activation:
                    if a.grad is not None:
                        intermediate_grads.append(a.grad)
                if None not in intermediate_grads:
                    chunked_grad_list += intermediate_grads
            if chunked_grad_list:
                # Append the list of grads to the all_grads list and this should be on the GPU.
                all_grads.append(torch.cat(chunked_grad_list).squeeze(-1))  # type: ignore
            with torch.autograd.profiler.record_function("fairscale.experimental.nn.offload:backward_drop"):
                # Move the shard back to the CPU. This should move all the grad tensors to CPU as well.
                # We don't need to move activations since we are using a copy of the tensors on the GPU.
                model_shard.backward_drop()
        detached_inputs = model_instance._activations[0]
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads


class ShardSyncLayer(torch.autograd.Function):
    """
    The shard sync layer is a synchronization point between model shards.
    - In the forward pass, it drops parameters in the previous shard and
    loads parameters for the next shard.
    - In the backward pass, it does the reverse.
    It does not change or create any outputs at all, instead it just
    forwards the input as the output.
    NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    @_conditional_amp_fwd_decorator  # type: ignore
    def forward(ctx: Any, inputs: Any, index: int, model_slices: Any, device_list, 
                model_instance: Any, mode: Optional[str] = None,
                percentage: float = 0.7,
                decoder: Optional[bool] = False
                ) -> Any:  # ctx到底是什么东西？
        drop_index = index
        load_index = index + 1 if not decoder else index + 2
        max_slices = len(model_slices)

        if drop_index > 0 and drop_index < max_slices - 1:
            if mode == "percentage":
                model_slices[drop_index].forward_percentage_drop(percentage)
            elif (mode == "select" or mode == "flexgen") and device_list[index] == 0:
                model_slices[drop_index].forward_drop()
            elif mode == "slice":
                model_slices[drop_index].forward_load()

        if load_index != 0 and load_index < max_slices - 1:
            if mode == "percentage":
                model_slices[load_index].forward_percentage_load()
            elif mode == "slice":
                model_slices[load_index].forward_load()


        # 这是什么意思？
        ctx.index = index
        ctx.model_slices = model_slices
        ctx.model_instance = model_instance

        # if mode == "percentage":
        #     torch.cuda.synchronize()

        return inputs if isinstance(inputs, tuple) else (inputs,)

    # Ignore the following function for code coverage since the backward pass
    # is triggered by C++ code and cannot be calculated when overriding
    # autograd.Function
    @staticmethod
    @_conditional_amp_bwd_decorator
    def backward(ctx, *grad_outputs):  # type: ignore # pragma: no cover

        load_index = ctx.index
        drop_index = load_index + 1
        model_slices = ctx.model_slices
        model_instance = ctx.model_instance

        # TODO(anj-s): Are these redundant in the backward pass?
        if drop_index == len(model_slices):
            # Drop the last activation since it is still on the CPU
            # after the loss.backward() call.
            model_instance._activations[-1] = tuple([a.cuda() for a in list(model_instance._activations[-1])])

        if drop_index < len(model_slices):
            # Move shard from device to offload device.
            model_slices[drop_index].backward_drop()
            model_instance._activations[drop_index] = tuple(
                [a.cpu() for a in list(model_instance._activations[drop_index])]
            )

        if load_index >= 0:
            # Load shard from offload device to device.
            model_slices[load_index].backward_load()
            model_instance._activations[load_index] = tuple(
                [a.cuda() for a in list(model_instance._activations[load_index])]
            )

        # The returned variables need to mirror the forward inputs
        # TODO(anj-s): Why do we need to do this?
        if isinstance(grad_outputs, tuple):
            return grad_outputs[0], None, None, None

        return grad_outputs, None, None, None


class OffloadModel(nn.Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train by offloading majority of the model parameters to the CPU.
    `OffloadModel` is heavily inspired by the _L2L algorithm and _Zero-Offload.
    ::

        model = get_model()
        offload_model = OffloadModel(model, device,
                                    offload_device=torch.device(“cpu”),
                                    num_slices=3,
                                    checkpoint_activation=True,
                                    num_microbatches=5)

    .. _L2L: https://arxiv.org/abs/2002.05645
    .. _Zero-Offload: https://arxiv.org/abs/2101.06840

    At each step, a layer(or series of layers) are loaded
    onto the GPU for the forward and backward pass with intermediate
    activations being copied onto the GPU as required. Once the forward
    or backward pass is completed for a given shard, it is moved back to
    the CPU again.

    `OffloadModel` supports activation checkpointing which reduces
    the memory footprint. You can also increase the number of
    microbatches which translates to more computation cycles for
    every shard load. This helps offset the cost of moving the shard
    from the CPU to GPU and vice versa.

    Note: OffloadModel currently only supports nn.Sequential models.

    Args:
        module (~torch.nn.Sequential): Module to be offloaded.

        device (torch.device):
            Device where the active model should reside.

        offload_device (torch.device):
            Device where the inactive model should reside.

        num_slices (int):
            Number of slices into which the model should be chunked.

        checkpoint_activation (bool):
            Boolean to indicate if we want to checkpoint intermediate
            activation states on the CPU. Default value is False.

        num_microbatches (int):
            Number of microbatches which should be run per model
            shard on device.
    """

    def __init__(
            self,
            model: Any,
            device: torch.device,
            offload_device: torch.device = torch.device("cpu"),
            num_slices: int = 3,
            checkpoint_activation: bool = False,
            num_microbatches: int = 1,
            device_list=None,
            percentage=0.7,
            name=None,
            mode=None
    ):
        super().__init__()
        if not model:
            raise TypeError("`model` argument to `OffloadModel` cannot be None.")

        if not device:
            raise TypeError("`device` argument to `OffloadModel` cannot be None.")

        if not (isinstance(model, nn.Sequential) or type(model) == list):
            raise TypeError("`model` argument to `OffloadModel` must be of type `nn.Sequential`.")

        if not torch.cuda.is_available():
            raise TypeError("CUDA must be available as one of the compute devices for `OffloadModel`.")

        self.device = device
        self.offload_device = offload_device
        # List of model shards that will be placed on/off the device.
        self.model_slices: List[nn.Module] = []
        self.percentage = percentage

        self.device_list = device_list
        self.compute_stream = torch.cuda.default_stream()  # 计算流，坚决不能新定义一个计算流！要让所有计算都在default stream上进行
        self.cpu_to_gpu_stream = torch.cuda.Stream(device=torch.device("cuda"))  # CPU到GPU的流
        self.gpu_to_cpu_stream = torch.cuda.Stream(device=torch.device("cuda"))

        self.name = name
        self.mode = mode

        if self.mode == "select" and self.device_list is not None:
            self.select_list = []
            self.start_list = []
            self.end_list = []
            self.start_list.append(0)  # 默认第一个在device上
            for index, value in enumerate(self.device_list):
                if value == 0:
                    self.select_list.append(index)
                elif value == 1:
                    if index != 0 and self.device_list[index - 1] == 0:
                        self.start_list.append(index)
                    if index != len(self.device_list) - 1 and self.device_list[index + 1] == 0:
                        self.end_list.append(index)
            if len(self.start_list) > len(self.select_list):
                self.start_list.pop()  # 如果最后是1且后无0，就删除最后一个start
            print("select_list: ",self.select_list)  # debug
            print("start_list: ",self.start_list)  # debug
            print("end_list: ",self.end_list)  # debug

        # TODO(anj): Add an experimental flag for using this instead of modifying the
        # arg type.
        # print(model, type(model))  # debug
        if type(model) == list:
            # for m in model:  # model的24层，总共循环24次
            #     for p in m.parameters():  # 每一层的参数
            #         p.data = p.data.pin_memory()
            # This is already sharded using the auto shard functinality.
            for i, m in enumerate(model):
                self.model_slices.append(
                    ModelShard(
                        cpu_model_shard=m,
                        device=device,
                        offload_device=offload_device,
                        index=i,
                        g_stream=self.gpu_to_cpu_stream,
                        c_stream=self.cpu_to_gpu_stream,
                    )
                )
        else:
            # print("no!!")  # debug
            # Slice the model into roughly equivalent sequential shards.
            # 基本上是按参数量划分的
            splits = _split(model, num_slices)  # type: ignore

            for i, split in enumerate(splits):
                # Add one model handling this slice
                self.model_slices.append(
                    ModelShard(
                        cpu_model_shard=nn.Sequential(*split),
                        device=device,
                        offload_device=offload_device,
                        index=i,
                    )
                )

        if self.mode == "percentage":
            self.load_percentage_model_shard(percentage)
        elif self.mode == "select" and device_list is not None:
            self.load_model_shard()

        # Expose a unified view of the slices
        self._model = torch.nn.Sequential(*self.model_slices)

        # intermediate activations at the slice boundaries.
        self._activations: List[Tuple] = []

        # Currently we only support microbatches with activation checkpointing.
        if not checkpoint_activation and num_microbatches > 1:
            raise RuntimeError("We currently only support microbatches with activation checkpointing.")

        # Bool indicating if we want to checkpoint activation on the host.
        self._checkpoint_activation = checkpoint_activation

        # Number of microbatches to run per batch on the device
        self._num_microbatches = num_microbatches

    def load_model_shard(self):
        for i, m in enumerate(self.model_slices):
            if self.device_list[i] == 1:
                self.model_slices[i].forward_load()

    def load_percentage_model_shard(self, percentage: float):
        for i, m in enumerate(self.model_slices):
            if i != 0 and i != len(self.model_slices) - 1:
                self.model_slices[i].init_percentage_load(percentage)
            else:
                self.model_slices[i].forward_load()

    def forward(self, *inputs: Any, **_: Any) -> Any:  # 需要改的
        # `apply` calls the `forward` function of the `OffloadFunction` class
        # and the `forward` function calls `inputs` on the first model shard.
        # Please see https://pytorch.org/docs/stable/autograd.html#function for more details.

        # We need the second param to be a dummy input to enable the
        # backward pass to be triggered for integer inputs.

        if self._checkpoint_activation:
            return OffloadFunction.apply(*inputs, torch.tensor([], requires_grad=True), self)  # 准备加_

        if self.mode == "select":
            select_list = self.select_list.copy()
            start_list = self.start_list.copy()
            end_list = self.end_list.copy()

        self._activations = []
        if self.name == "gpt2":
            hidden_states=_['hidden_states'],
            kv_caches=_['kv_caches'],
            attn_metadata=_['attn_metadata'],
        elif self.name == "bert":
            attention_mask=_['attention_mask']
            head_mask=_['head_mask']
            encoder_hidden_states=_['encoder_hidden_states']
            encoder_attention_mask=_['encoder_attention_mask']
            past_key_values=_['past_key_values']
            use_cache=_['use_cache']
            output_attentions=_['output_attentions']
            output_hidden_states=_['output_hidden_states']
            return_dict=_['return_dict']
            add_cross_attention = _['add_cross_attention']  # maphsge4 add
        elif self.name == "qwen":
            positions = _['positions']
            hidden_states=_['hidden_states'],
            kv_caches=_['kv_caches'],
            attn_metadata=_['attn_metadata'],
            residual=_['residual'],
        elif self.name == "yi":
            positions = _['positions']
            hidden_states=_['hidden_states'],
            kv_caches=_['kv_caches'],
            attn_metadata=_['attn_metadata'],
            residual=_['residual'],
        elif self.name == "opt":
            hidden_states=_['hidden_states'],
            kv_caches=_['kv_caches'],
            attn_metadata=_['attn_metadata'],
        elif self.name == "qwen2" or self.name == "llama":
            positions = _['positions'],
            residual=_['residual'],
            hidden_states=_['hidden_states'],
        elif self.name == "optn":
            hidden_states = _['hidden_states'],
        elif self.name == "neo":
            batch = _['batch'],
            embeddings = _['embeddings'],
            batch = batch[0] if isinstance(batch, tuple) else batch
            embeddings = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        
        
        if self.name != "neo":
            hidden_states = hidden_states[0]  # vllm之前是inputs[0]

        if self.name == "bert":
            next_decoder_cache = () if use_cache else None
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            all_cross_attentions = () if output_attentions and add_cross_attention else None
        elif self.name == "qwen" or self.name == "yi":
            attn_metadata = attn_metadata[0]
            residual = residual[0]  
            kv_caches = kv_caches[0]
        elif self.name == "opt":
            attn_metadata = attn_metadata[0]
            kv_caches = kv_caches[0]
        elif self.name == "gpt2":
            kv_caches = kv_caches[0]
            attn_metadata = attn_metadata[0]
        elif self.name == "qwen2" or self.name == "llama":
            residual = residual[0]
            positions = positions[0]
        
        for index in range(-1, len(self.model_slices)):
            if index >= 0:
                time1 = time.time()
                
                if self.name != "neo":
                    inputs = hidden_states  # 叶博改的
                nvtx.range_push(f"shard {index} forward")

                # inputs = self.model_slices[index](*inputs)[0]
                # torch.cuda.synchronize()  # 这个同步是真不能删
                
                if self.name == "gpt2":
                    inputs = self.model_slices[index](
                        hidden_states=inputs,
                        kv_cache=kv_caches[index],
                        attn_metadata=attn_metadata,
                    )
                elif self.name == "bert":
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)
                    
                    layer_head_mask = head_mask[index] if head_mask is not None else None
                    past_key_value = past_key_values[index] if past_key_values is not None else None

                    inputs = self.model_slices[index](
                        inputs,
                        attention_mask=attention_mask,
                        head_mask=layer_head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                    )
                elif self.name == "qwen" or self.name == "yi":     
                    # if self.device_list is not None and self.device_list[index] == 0:  # 如果不是device上
                    #     inputs = inputs.cpu()
                    #     residual = residual.cpu()
                    inputs , residual = self.model_slices[index](
                        positions=positions,
                        hidden_states=inputs,
                        kv_cache=self.model_slices[index].kv_cache,
                        attn_metadata=attn_metadata,
                        residual=residual,
                    )
                    # if self.device_list[index] == 0:  # 如果是device上 
                    #     residual = residual.cuda()
                    #     inputs = inputs.cuda()
                elif self.name == "qwen2" or self.name == "llama":
                    inputs , residual = self.model_slices[index](
                        positions=positions,
                        hidden_states=inputs,
                        residual=residual,
                    )
                elif self.name == "opt":
                    inputs = self.model_slices[index](
                        hidden_states=inputs,
                        kv_cache=self.model_slices[index].kv_cache,
                        attn_metadata=attn_metadata,
                    )
                elif self.name == "optn":
                    inputs = self.model_slices[index](
                        hidden_states=inputs,
                    )
                elif self.name == "neo":
                    embeddings = self.model_slices[index](
                        batch=batch,
                        embeddings=embeddings,
                    )
                else:
                    inputs = self.model_slices[index](*inputs)
                    
                time2 = time.time()
                tot_time = time2 - time1
                # print(f"shard {index} forward time: {tot_time}")

                if self.mode == "select":
                    if len(start_list) > 0 and index == start_list[0]:
                        start_list.pop(0)
                        select_id = select_list.pop(0)
                        if self.model_slices[index].kv_cache is None:
                            self.model_slices[select_id].forward_load()

                    if len(end_list) > 0 and index == end_list[0]:
                        end_list.pop(0)
                        torch.cuda.synchronize()  # 不知道为什么，现在版本中这里不能删
                
                nvtx.range_pop()
                            
            # Call the custom autograd hooks (discard/load slices FW and BW)
            inputs = ShardSyncLayer.apply(inputs, index, self.model_slices, self.device_list, self, self.mode, self.percentage)  
            if self.mode == "percentage":
                torch.cuda.current_stream().wait_stream(self.cpu_to_gpu_stream)

            if index >= 0:
                if self.name == "gpt2":
                    hidden_states = inputs[0]  # hidden_states显存变小是因为本身有值，赋的新值比原来的值小

                    # # Model Parallel: If it's the last layer for that device, put things on the next device
                    # if self.model_parallel:
                    #     for k, v in self.device_map.items():
                    #         if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    #             hidden_states = hidden_states.to("cuda:" + str(k + 1))
                elif self.name == "bert":
                    hidden_states = inputs[0]
                    if use_cache:
                        next_decoder_cache += (inputs[-1],)
                    if output_attentions:
                        all_self_attentions = all_self_attentions + (inputs[1],)
                        if add_cross_attention:
                            all_cross_attentions = all_cross_attentions + (inputs[2],)     
                elif self.name != "neo":
                    hidden_states = inputs[0]

        if self.name == "gpt2":
            return (hidden_states)
        elif self.name == "bert":
            if use_cache is False:
                return (hidden_states, all_hidden_states, None, None, None) 
            else:
                return (hidden_states, all_hidden_states, next_decoder_cache, all_self_attentions, all_cross_attentions)
        elif self.name == "qwen" or self.name == "qwen2" or self.name == "llama" or self.name == "yi":
            return (hidden_states, residual)
        elif self.name == "opt" or self.name == "optn":
            return (hidden_states)
        elif self.name == "neo":
            return embeddings
        
    def forward_pipeline(self, embeddings: Any, batches: Any) -> Any:

        inputs = ()

        if self.mode == "select":
            select_list = self.select_list.copy()
            start_list = self.start_list.copy()
            end_list = self.end_list.copy()

        for index in range(-2, len(self.model_slices)):

            if index == -1:
                q1, k1, v1 = self.model_slices[-1].model_shard.forward_first_stage(embeddings, batches)
            elif index == len(self.model_slices) - 1:
                embeddings = self.model_slices[-1].model_shard.forward_last_stage(q1, k1, v1, batches)
            elif index >= 0 and index < len(self.model_slices) - 1:
                # torch.cuda.synchronize()  # 确保前一片的计算已经完成
                with torch.cuda.stream(self.compute_stream):
                    q1, k1, v1 = self.model_slices[index].model_shard.forward_double(q1, k1, v1, batches)

                if self.mode == "select":
                    if len(start_list) > 0 and index == start_list[0]:
                        start_list.pop(0)
                        select_id = select_list.pop(0)
                        if self.model_slices[index].kv_cache is None:
                            self.model_slices[select_id].forward_load()

                    if len(end_list) > 0 and index == end_list[0]:
                        end_list.pop(0)
                        torch.cuda.current_stream().wait_stream(self.cpu_to_gpu_stream)

            inputs = ShardSyncLayer.apply(inputs, index, self.model_slices, self.device_list, self, self.mode, self.percentage, True)
            if self.mode == "percentage":
                torch.cuda.current_stream().wait_stream(self.cpu_to_gpu_stream)

        return embeddings
        

        