import torch
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from verl import DataProto
from verl.utils.reward_score import _default_compute_score

class CodeRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        lock = threading.Lock()
        print_lock = threading.Lock()

        def process_item(i, data_item):
            nonlocal already_print_data_sources, reward_tensor

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
    
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
    
            score_result = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            #with print_lock:
            #    print(compute_score_debug_str)
            # Handle both scalar and dictionary returns
            if isinstance(score_result, dict):
                score = score_result['score']
                if 'extra_info' in score_result:
                    for key, value in score_result['extra_info'].items():
                        if key not in extra_info_dict:
                            extra_info_dict[key] = [0.0] * len(data)
                        extra_info_dict[key][i] = value
            else:
                score = float(score_result)
    
            reward_tensor[i, valid_response_length - 1] = score

            with lock:
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                current_count = already_print_data_sources[data_source]
    
                if current_count < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    need_print = True
                else:
                    need_print = False
    
            if need_print:
                with print_lock:
                    print(sequences_str)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_item, i, data[i]) for i in range(len(data))]
            for future in futures:
                future.result()
        return reward_tensor