# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
import logging

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)

def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 data_source_key=None,
                 run_diagnostics=False):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.data_source_key = data_source_key

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
        
        # Run diagnostics if requested
        if run_diagnostics:
            print(f"Running diagnostics on dataset with {len(self.dataframe)} samples")
            self._check_and_print_sample(num_samples=10)

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        logger.info(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]

        logger.info(f'filter dataset len: {len(self.dataframe)}')
        
        # Log data source information if data_source_key is specified
        if self.data_source_key and self.data_source_key in self.dataframe.columns:
            sources = self.dataframe[self.data_source_key].unique()
            print(f"Dataset contains {len(sources)} data sources: {sources}")
            logger.info(f"Dataset contains {len(sources)} data sources: {sources}")
            # Log source distribution
            source_counts = self.dataframe[self.data_source_key].value_counts().to_dict()
            logger.info(f"Data source distribution: {source_counts}")
            print(f"Data source distribution: {source_counts}")

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            logger.warning('old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict.pop(self.prompt_key)
        
        # Ensure data_source is included in output if data_source_key is specified
        if self.data_source_key and self.data_source_key in row_dict:
            # Use 'data_source' as a standardized key for the curriculum sampler
            row_dict['data_source'] = row_dict[self.data_source_key]

        apply_chat_template = row_dict.get("extra_info", {}).get("apply_chat_template", True)

        if apply_chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            assert isinstance(chat, str), "If not applying chat_template, the prompt inside the dataentry should be a string"
            prompt_with_chat_template = chat

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
    
    def create_curriculum_sampler(self, batch_size, seed=None):
        """Create a curriculum learning sampler for this dataset.
        
        Args:
            batch_size: Batch size for sampling
            seed: Random seed for reproducibility
            
        Returns:
            A CurriculumSampler instance if data_source_key is set,
            otherwise None (indicating curriculum learning not applicable)
        """
        if not self.data_source_key:
            logger.warning("Cannot create curriculum sampler: data_source_key not specified")
            return None
            
        # Import here to avoid circular import issues
        from verl.utils.curriculum_learning import CurriculumSampler
        return CurriculumSampler(
            dataset=self,
            data_source_key='data_source',  # Use standardized key
            batch_size=batch_size,
            seed=seed
        )

    def _check_and_print_sample(self, num_samples=5):
        """
        Check and print details about a few samples to help diagnose format issues.
        
        Args:
            num_samples: Number of samples to check and print.
        """
        import random
        import json
        
        print("\n===== DIAGNOSTICS: SAMPLE CHECKS =====")
        
        # Get indices for random samples
        if len(self.dataframe) < num_samples:
            print(f"Warning: Requested {num_samples} samples but only {len(self.dataframe)} available")
            indices = list(range(len(self.dataframe)))
        else:
            indices = random.sample(range(len(self.dataframe)), num_samples)
        
        problem_indices = []
        
        for i, idx in enumerate(indices):
            try:
                print(f"\n----- Sample {i+1}/{len(indices)} (index {idx}) -----")
                
                # Get the raw sample
                row_dict = self.dataframe.iloc[idx].to_dict()
                
                # Extract the chat/prompt
                chat = row_dict[self.prompt_key]
                
                # Print type and truncated content
                print(f"Chat type: {type(chat)}")
                if isinstance(chat, str):
                    print(f"Chat content (truncated): {chat[:100]}...")
                elif isinstance(chat, dict):
                    print(f"Chat keys: {list(chat.keys())}")
                    for k, v in chat.items():
                        print(f"  {k}: {type(v)} - {str(v)[:50]}...")
                elif isinstance(chat, list):
                    print(f"Chat length: {len(chat)}")
                    for j, item in enumerate(chat[:3]):  # Show first 3 items
                        print(f"  Item {j}: {type(item)} - {str(item)[:50]}...")
                    if len(chat) > 3:
                        print(f"  ... plus {len(chat)-3} more items")
                else:
                    print(f"Chat content: {str(chat)[:100]}...")
                
                # Try to apply chat template
                print("Attempting to apply chat template...")
                
                # For diagnostic purposes, try both direct and transformed approaches
                try:
                    # Method 1: Direct application
                    print("Method 1: Direct application")
                    result1 = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                    print(f"Success! Result length: {len(result1)}")
                    print(f"Result preview: {result1[:50]}...")
                except Exception as e1:
                    print(f"Error with direct application: {e1}")
                    
                    # Method 2: With formatting transformation
                    try:
                        print("Method 2: With formatting transformation")
                        # Apply same transformation as in __getitem__
                        if isinstance(chat, str):
                            formatted_chat = [{"role": "user", "content": chat}]
                        elif isinstance(chat, dict) and "role" in chat and "content" in chat:
                            formatted_chat = [chat]
                        elif isinstance(chat, dict):
                            if "prompt" in chat:
                                formatted_chat = [{"role": "user", "content": chat["prompt"]}]
                            else:
                                formatted_chat = [{"role": "user", "content": json.dumps(chat)}]
                        elif isinstance(chat, list):
                            if all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in chat):
                                formatted_chat = chat
                            else:
                                formatted_chat = [{"role": "user", "content": str(chat)}]
                        else:
                            formatted_chat = [{"role": "user", "content": str(chat)}]
                            
                        # Print the transformed format
                        print(f"Transformed format: {json.dumps(formatted_chat)[:100]}...")
                        
                        # Try with transformed format
                        result2 = self.tokenizer.apply_chat_template(formatted_chat, add_generation_prompt=True, tokenize=False)
                        print(f"Success with transformation! Result length: {len(result2)}")
                        print(f"Result preview: {result2[:50]}...")
                    except Exception as e2:
                        print(f"Error even with transformation: {e2}")
                        problem_indices.append(idx)
            
            except Exception as e:
                print(f"Error analyzing sample: {e}")
                problem_indices.append(idx)
        
        print("\n===== DIAGNOSTICS SUMMARY =====")
        print(f"Checked {len(indices)} samples")
        if problem_indices:
            print(f"Found {len(problem_indices)} problematic samples at indices: {problem_indices}")
        else:
            print("All samples checked successfully")
        print("===== END DIAGNOSTICS =====\n")
        
        return problem_indices
