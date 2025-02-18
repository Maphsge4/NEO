"""
Offline example of using the NEO engine to run inference on a model.

Performance details are printed on the screen.

Note that this script is for demonstration purposes only and uses symmetric pipelining. In evaluation, we use asymmetric pipelining instead.
"""
import os
import time
import argparse
from transformers import AutoTokenizer

import swiftllm


if __name__ == '__main__':
    home = os.path.expanduser("~")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_dir = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser()
    parser.description = """
        An example script to demonstrate how to use the swiftllm model executor directly for inferencing without using the engine
    """
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        default=f"{home}/weights/Llama-3-8B"
    )
    parser.add_argument(
        "--library-path",
        help="Path to the shared library",
        type=str,
        default=f"{repo_dir}/pacpu/build/libpacpu-llama3_8b-tp1.so"
    )
    parser.add_argument(
        "--profile-result-path",
        help="Path to folder of profiling results",
        type=str,
        default=f"{repo_dir}/profile_results/"
    )
    parser.add_argument(
        "--prompt-path",
        help="Path to the prompt file",
        type=str,
        default=f"{script_dir}/example.txt"
    )
    parser.add_argument(
        "--num-gpu-requests",
        help="Number of GPU requests",
        type=int,
        default=10
    )
    parser.add_argument(
        "--num-cpu-requests",
        help="Number of CPU requests",
        type=int,
        default=10
    )
    parser.add_argument(
        "--monitor-performace",
        help="Performance monitoring switch",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    
    # 1. Create the engine
    engine_config = swiftllm.EngineConfig(
        model_path = args.model_path,
        use_dummy = False,

        block_size = 16,
        gpu_mem_utilization = 0.99,
        num_gpu_blocks_override = 1300,
        swap_space = 2,
        max_seqs_in_block_table = 1024,
        max_blocks_per_seq = 512,

        max_batch_size = 512,
        max_tokens_in_batch = 20000,

        library_path=args.library_path,
        profile_result_path=args.profile_result_path,

        extra_layer_for_cprf=True,
        tensor_parallel_degree=1
    )

    start_time = time.perf_counter()
    engine = swiftllm.Engine(engine_config)
    engine.initialize()
    print(f"Engine creation time: {time.perf_counter() - start_time:.2f} seconds")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)


    # 2. Load the prompt and tokenize
    ngpu_prompts = args.num_gpu_requests
    ncpu_prompts = args.num_cpu_requests
    nprompts = ncpu_prompts + ngpu_prompts
    with open(args.prompt_path, "r") as f:
        prompt = ''.join(f.readlines())

    input_ids = tokenizer(prompt)['input_ids']
    print("Prompt token length: ", len(input_ids))
    
    
    # 3. Prefill the prompts
    reqs = [None] * nprompts
    gpu_req_ids = list(range(ngpu_prompts // 2)) + list(range(nprompts // 2, nprompts // 2 + ngpu_prompts // 2))
    gpu_reqs = []
    if ngpu_prompts:
        batch = swiftllm.SubBatch()
        for i in gpu_req_ids:
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=True)
        gpu_reqs = [reqs[i] for i in gpu_req_ids]
        engine.step([batch])

    if ncpu_prompts:
        batch = swiftllm.SubBatch()
        for i in range(ngpu_prompts // 2, nprompts // 2):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=False)
        engine.step([batch])

        batch = swiftllm.SubBatch()
        for i in range(nprompts // 2 + ngpu_prompts // 2, nprompts):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=False)
        engine.step([batch])

    print("Prefilling phase done")


    # 4. Run the inference
    if args.monitor_performace:
        engine.executor.turn_on_perf_monitor()
    
    for iteration in range(16):
        batches = [swiftllm.SubBatch() for _ in range(2)]
        for i in range(ngpu_prompts // 2):
            batches[0].add_gdec(reqs[i])
        for i in range(ngpu_prompts // 2, nprompts // 2):
            batches[1].add_cdec(reqs[i])
        for i in range(nprompts // 2, nprompts // 2 + ngpu_prompts // 2):
            batches[1].add_gdec(reqs[i])
        for i in range(nprompts // 2 + ngpu_prompts // 2, nprompts):
            batches[0].add_cdec(reqs[i])
            
        # Comment out the following 4 lines to run pure decoding iterations
        reqs.append(swiftllm.create_request(input_ids, len(reqs)))
        reqs.append(swiftllm.create_request(input_ids, len(reqs)))
        batches[0].add_pref(reqs[-2], is_gpu=False)
        batches[1].add_pref(reqs[-1], is_gpu=False)

        start = time.perf_counter()
        engine.step(batches)
        end = time.perf_counter()
        print(f"Iteration {iteration:3} E2E time: {(end - start) * 1000:.4f} ms")
    
    for i in range(nprompts):
        if i in (0, nprompts // 2 - 1, nprompts - 1):
            output_text = tokenizer.decode(reqs[i].output_token_ids, skip_special_tokens=True)
            print(f"{prompt}|{output_text}")

    if args.monitor_performace:
        res = engine.executor.turn_off_perf_monitor_and_flush_results()
        print(res)
