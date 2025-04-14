"""
Combine multiple logic datasets into a single parquet file
"""

import os
import re
import random
import numpy as np
import math
import datasets
from datasets import concatenate_datasets
import argparse
import json
import tiktoken
from transformers import AutoTokenizer
import importlib.util
import warnings

from verl.utils.hdfs_io import copy, makedirs

# ========================================================================
# GLOBAL CONFIGURATION - All sampling parameters and settings in one place
# ========================================================================

CONFIG = {
    # General settings
    "random_seed": 42,            # Seed for all random operations
    "local_dir": "~/data/logic_datasets",  # Default local directory for saving datasets
    "final_test_samples": 2000,    # Final number of test samples to save
    
    # Default sample limits per dataset
    "max_samples_per_dataset": None,  # Default max samples per dataset (None = no limit)
    
    # Dataset-specific settings
    "datasets": {
        "kk": {
            "train_samples": 900,     # Number of samples for training per depth
            "test_samples": 100,      # Number of samples for testing per depth
            "depth_values": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # People counts (depths) to include
            "data_paths": {
                3: "./combined_logic_dataset/kk/people3_num1000.jsonl",
                4: "./combined_logic_dataset/kk/people4_num1000.jsonl",
                5: "./combined_logic_dataset/kk/people5_num1000.jsonl",
                6: "./combined_logic_dataset/kk/people6_num1000.jsonl",
                7: "./combined_logic_dataset/kk/people7_num1000.jsonl",
                8: "./combined_logic_dataset/kk/people8_num1000.jsonl",
                9: "./combined_logic_dataset/kk/people9_num1000.jsonl",
                10: "./combined_logic_dataset/kk/people10_num1000.jsonl",
                11: "./combined_logic_dataset/kk/people11_num1000.jsonl",
                12: "./combined_logic_dataset/kk/people12_num1000.jsonl",
                13: "./combined_logic_dataset/kk/people13_num1000.jsonl",
                14: "./combined_logic_dataset/kk/people14_num1000.jsonl",
            },
            "template_type": "qwen-instruct"  # Template format to use
        }
    },
    
    # Datasets that track depth attribute
    "datasets_with_depth": [
        "kk_logic"
    ]
}

QWEN_TOKENIZER = None

def format_prompt(content):
    """Format content in the standardized prompt format"""
    return f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a reasoning problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{content}\n"""

def process_kk_logic(example, idx, split, depth):
    """Process Knights and Knaves (KK) logic dataset examples"""
    quiz = example.get('quiz', '')
    
    # Format the prompt based on template type
    template_type = CONFIG["datasets"]["kk"]["template_type"]
    if template_type == 'base':
        content = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        content = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n"""
    
    # Format the final prompt
    prompt = content
    
    # Get the solution and convert it to JSON string to make it compatible with other datasets
    solution_dict = {
        "solution_text_format": example.get('solution_text_format', ''),
        "statements": example.get('statements', [])
    }
    
    solution = json.dumps(solution_dict)  # Convert to JSON string
    
    # Store the original solution in extra_info for reference
    return {
        "data_source": "kk_logic_"+str(depth),
        "prompt": [{
            "role": "user",
            "content": prompt,
        }],
        "ability": "logic",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution  # Now a string, compatible with other datasets
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'quiz': quiz,
            'depth': depth,  # Store the depth (number of people)
        }
    }

def count_tokens(text):
    """Count the number of tokens in the text using Qwen tokenizer"""
    global QWEN_TOKENIZER
    
    # Handle case where text is a list of prompt dictionaries
    if isinstance(text, list):
        # Extract content from prompts list
        text_content = ""
        for item in text:
            if isinstance(item, dict) and "content" in item:
                text_content += item["content"] + " "
        text = text_content.strip()
    
    # Initialize tokenizer (if not already loaded)
    if QWEN_TOKENIZER is None:
        try:
            # Check if transformers is available
            if importlib.util.find_spec("transformers") is None:
                warnings.warn("transformers package not installed. Using fallback token counter.")
                return len(text.split())
                
            # Load Qwen tokenizer
            QWEN_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
            print("Successfully loaded Qwen tokenizer for token counting")
        except Exception as e:
            warnings.warn(f"Failed to load Qwen tokenizer: {e}. Using fallback token counter.")
            return len(text.split())
    
    try:
        # Use Qwen tokenizer for token counting
        tokens = QWEN_TOKENIZER.encode(text)
        return len(tokens)
    except Exception as e:
        warnings.warn(f"Error during tokenization: {e}. Using fallback token counter.")
        return len(text.split())

def print_dataset_stats(dataset, name):
    """Print detailed statistics about the dataset"""
    print(f"\n{'='*80}")
    print(f"  {name} Dataset Statistics  ")
    print(f"{'='*80}")
    print(f"Total samples: {len(dataset)}")
    
    # Count by data source
    source_counts = {}
    token_counts = {}
    
    for example in dataset:
        source = example['data_source']
        if source not in source_counts:
            source_counts[source] = 0
            token_counts[source] = []
        
        source_counts[source] += 1
        # Calculate prompt tokens and add to list
        prompt_text = example['prompt']
        token_count = count_tokens(prompt_text)
        token_counts[source].append(token_count)
    
    print("\nData source distribution:")
    print("-" * 50)
    for source, count in sorted(source_counts.items()):
        avg_tokens = sum(token_counts[source]) / len(token_counts[source]) if token_counts[source] else 0
        print(f"{source}: {count} samples ({count/len(dataset)*100:.2f}%) - Avg input tokens: {avg_tokens:.1f}")
    
    # Statistics for datasets with depth field
    datasets_with_depth = CONFIG["datasets_with_depth"]
    
    for source in datasets_with_depth:
        if source in source_counts:
            print(f"\n{source} depth distribution:")
            print("-" * 50)
            
            depth_counts = {}
            depth_token_counts = {}
            
            for example in dataset:
                if example['data_source'] == source and 'extra_info' in example:
                    depth = example['extra_info'].get('depth')
                    if depth is not None:
                        if depth not in depth_counts:
                            depth_counts[depth] = 0
                            depth_token_counts[depth] = []
                        
                        depth_counts[depth] += 1
                        prompt_text = example['prompt']
                        token_count = count_tokens(prompt_text)
                        depth_token_counts[depth].append(token_count)
            
            if depth_counts:
                for depth, count in sorted(depth_counts.items()):
                    source_total = source_counts[source]
                    avg_tokens = sum(depth_token_counts[depth]) / len(depth_token_counts[depth]) if depth_token_counts[depth] else 0
                    print(f"  Depth {depth}: {count} samples ({count/source_total*100:.2f}%) - Avg input tokens: {avg_tokens:.1f}")
            else:
                print("  No valid depth information found")
    
    # Calculate overall average token count
    all_token_counts = []
    for source_tokens in token_counts.values():
        all_token_counts.extend(source_tokens)
    
    avg_all_tokens = sum(all_token_counts) / len(all_token_counts) if all_token_counts else 0
    print(f"\nOverall average input token count: {avg_all_tokens:.1f}")
    
    # Add examples from each dataset
    print(f"\nExample prompts and ground truths by source:")
    print("-" * 80)
    
    # Create a dictionary to store one example from each source
    source_examples = {}
    
    # Get the first example from each source
    for example in dataset:
        source = example['data_source']
        if source not in source_examples:
            source_examples[source] = example
    
    # Print example from each source
    for source, example in sorted(source_examples.items()):
        print(f"\n{'='*40}")
        print(f"--- {source} Example ---")
        print(f"{'='*40}")
        
        # Get the ground truth
        if 'reward_model' in example and 'ground_truth' in example['reward_model']:
            ground_truth = example['reward_model']['ground_truth']
        else:
            ground_truth = "No ground truth found"
            
        # For JSON ground truths, provide a more readable format
        if isinstance(ground_truth, str) and (ground_truth.startswith('{') or ground_truth.startswith('[')):
            try:
                # Try to parse and pretty print the JSON
                parsed_json = json.loads(ground_truth)
                ground_truth = json.dumps(parsed_json, indent=2)
            except:
                # If parsing fails, just show the original string
                pass
        
        print(f"\nPROMPT:")
        print("-" * 40)
        print(example['prompt'])
        print("-" * 40)
        
        print(f"\nGROUND TRUTH:")
        print("-" * 40)
        print(ground_truth)
        print("-" * 40)
    
    print(f"{'='*80}\n")

def get_human_readable_size(size_bytes):
    """Convert bytes to human readable format (KB, MB, GB etc.)"""
    if size_bytes == 0:
        return "0 B"
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = int(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def load_jsonl_file(file_path):
    """Load a JSONL file and return a dataset"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
        
    try:
        # Function to load JSONL
        def gen_from_jsonl(path):
            with open(path) as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line in {path}: {e}")
                        continue
        
        # Load the dataset
        dataset = datasets.Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': file_path})
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=CONFIG["local_dir"])
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    
    # Update CONFIG with command line arguments
    CONFIG["local_dir"] = args.local_dir
    
    # Set all random seeds for reproducibility
    SEED = CONFIG["random_seed"]
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"Using random seed: {SEED} for reproducibility")
    
    # Create separate lists for training and test datasets
    processed_train_datasets = []
    processed_test_datasets = []
    
    # Process Knights and Knaves (KK) dataset
    try:
        print("Processing Knights and Knaves (KK) logic dataset")
        kk_train_datasets = []
        kk_test_datasets = []
        
        for depth in CONFIG["datasets"]["kk"]["depth_values"]:
            file_path = CONFIG["datasets"]["kk"]["data_paths"][depth]
            print(f"Loading KK dataset from {file_path} (depth={depth})")
            
            # Load the dataset using the helper function
            kk_dataset = load_jsonl_file(file_path)
            
            if kk_dataset is None:
                print(f"Skipping depth {depth} due to loading error")
                continue
                
            print(f"Loaded KK dataset with {len(kk_dataset)} examples for depth {depth}")
            
            # Check if we have enough samples
            required_samples = CONFIG["datasets"]["kk"]["train_samples"] + CONFIG["datasets"]["kk"]["test_samples"]
            if len(kk_dataset) < required_samples:
                print(f"Warning: KK dataset for depth {depth} has only {len(kk_dataset)} examples, less than the requested {required_samples}")
                # Adjust sample counts to available data
                train_count = int(len(kk_dataset) * 0.8)
                test_count = len(kk_dataset) - train_count
            else:
                train_count = CONFIG["datasets"]["kk"]["train_samples"]
                test_count = CONFIG["datasets"]["kk"]["test_samples"]
            
            # Shuffle the dataset for random selection
            indices = list(range(len(kk_dataset)))
            random.Random(SEED).shuffle(indices)
            
            # Select train and test samples
            train_indices = indices[:train_count]
            test_indices = indices[train_count:train_count+test_count]
            
            train_data = kk_dataset.select(train_indices)
            test_data = kk_dataset.select(test_indices)
            
            # Process train dataset
            print(f"Processing {len(train_data)} KK examples for depth {depth} (train)")
            process_fn = lambda example, idx, d=depth: process_kk_logic(example, idx, 'train', d)
            processed_train = train_data.map(function=process_fn, with_indices=True)
            kk_train_datasets.append(processed_train)
            
            # Process test dataset
            print(f"Processing {len(test_data)} KK examples for depth {depth} (test)")
            process_fn = lambda example, idx, d=depth: process_kk_logic(example, idx, 'test', d)
            processed_test = test_data.map(function=process_fn, with_indices=True)
            kk_test_datasets.append(processed_test)
        
        # Combine all KK datasets and add to the main datasets
        if kk_train_datasets:
            combined_kk_train = concatenate_datasets(kk_train_datasets)
            processed_train_datasets.append(combined_kk_train)
            print(f"Added {len(combined_kk_train)} KK examples to training dataset")
        
        if kk_test_datasets:
            combined_kk_test = concatenate_datasets(kk_test_datasets)
            processed_test_datasets.append(combined_kk_test)
            print(f"Added {len(combined_kk_test)} KK examples to test dataset")
            
    except Exception as e:
        print(f"Error processing Knights and Knaves dataset: {e}")
    
    # Check if we have any datasets to concatenate
    if not processed_train_datasets:
        print("No training datasets were successfully processed. Exiting.")
        exit(1)
        
    if not processed_test_datasets:
        print("No test datasets were successfully processed. Exiting.")
        exit(1)
    
    # Then merge the datasets
    try:
        combined_train_dataset = concatenate_datasets(processed_train_datasets)
        combined_test_dataset = concatenate_datasets(processed_test_datasets)

        combined_train_dataset = combined_train_dataset.filter(
            lambda example: example.get('prompt') is not None
        )
        combined_test_dataset = combined_test_dataset.filter(
            lambda example: example.get('prompt') is not None
        )
    except Exception as e:
        print(f"Error concatenating datasets: {e}")
        exit(1)
    
    # Print training and test dataset statistics
    if processed_train_datasets:
        print_dataset_stats(combined_train_dataset, "Training")
    
    if processed_test_datasets:
        print_dataset_stats(combined_test_dataset, "Test")
    
    # Create output directory if needed
    local_dir = os.path.expanduser(CONFIG["local_dir"])
    os.makedirs(local_dir, exist_ok=True)
    
    # Save the training dataset
    if processed_train_datasets:
        print(f"Combined training dataset contains {len(combined_train_dataset)} examples")
        
        train_output_path = os.path.join(local_dir, 'combined_logic_datasets_train.parquet')
        combined_train_dataset.to_parquet(train_output_path)
        
        # Get and print file size
        file_size = os.path.getsize(train_output_path)
        human_readable_size = get_human_readable_size(file_size)
        print(f"Saved combined training dataset to {train_output_path} (File size: {human_readable_size})")
        
        # Copy training dataset to HDFS if specified
        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            hdfs_train_path = os.path.join(args.hdfs_dir, 'combined_logic_datasets_train.parquet')
            copy(src=train_output_path, dst=hdfs_train_path)
            print(f"Copied training dataset to HDFS: {hdfs_train_path}")
    else:
        print("No training datasets were successfully processed.")
    
    # Save the test dataset
    if processed_test_datasets:
        print(f"Combined test dataset contains {len(combined_test_dataset)} examples")
        
        # Randomly sample test dataset before saving
        if len(combined_test_dataset) > CONFIG["final_test_samples"]:
            # Set random seed for reproducibility
            local_random = random.Random(CONFIG["random_seed"])
            
            # Create a list of all indices and shuffle it
            indices = list(range(len(combined_test_dataset)))
            local_random.shuffle(indices)
            
            # Select the specified number of examples
            selected_indices = indices[:CONFIG["final_test_samples"]]
            sampled_test_dataset = combined_test_dataset.select(selected_indices)
            print(f"Randomly sampled {len(sampled_test_dataset)} examples from the test dataset (from original {len(combined_test_dataset)} examples)")
            combined_test_dataset = sampled_test_dataset
        
        test_output_path = os.path.join(local_dir, 'combined_logic_datasets_test.parquet')
        combined_test_dataset.to_parquet(test_output_path)
        
        # Get and print file size
        file_size = os.path.getsize(test_output_path)
        human_readable_size = get_human_readable_size(file_size)
        print(f"Saved combined test dataset to {test_output_path} (File size: {human_readable_size})")
        
        # Copy test dataset to HDFS if specified
        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            hdfs_test_path = os.path.join(args.hdfs_dir, 'combined_logic_datasets_test.parquet')
            copy(src=test_output_path, dst=hdfs_test_path)
            print(f"Copied test dataset to HDFS: {hdfs_test_path}")
    else:
        print("No test datasets were successfully processed.")
