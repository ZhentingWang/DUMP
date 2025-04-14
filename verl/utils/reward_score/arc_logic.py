import re
from typing import Dict, Tuple, Optional

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    #print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        #print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    # Check that nothing follows </answer> except possibly <|im_end|>
    if positions['answer_end'] >= 0:
        remaining_text = processed_str[positions['answer_end'] + len('</answer>'):].strip()
        if remaining_text and remaining_text != '<|im_end|>':
            print(f"  [Error] Unexpected content after </answer>: '{remaining_text}'")
            validation_passed = False
        else:
            print("  Post-answer content validation passed")

    return validation_passed

def compute_score(solution_str: str, 
                 ground_truth: str,
                 format_reward: int = 1,
                 answer_reward: float = 1.0):
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Expected answer string (e.g., a grid in JSON format)
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print(" Processing New Sample ".center(80, '='))
    
    # Print ground truth
    print(f"ground_truth: {ground_truth}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)

    # Validate answer content
    answer_score = 0
  
    if answer_text is None:
        answer_score = -1.5
        print("  Content validation: NO VALID ANSWER FOUND")
    else:
        # Convert model's grid answer to a comparable format
        try:
            # Process grid answer - convert text to 2D array
            model_grid = []
            for line in answer_text.strip().split('\n'):
                row = [int(cell) for cell in line.strip().split()]
                model_grid.append(row)
            
            # Convert ground_truth to the same format for comparison (assuming ground_truth is a grid in JSON format)
            import json
            expected_grid = json.loads(ground_truth)
            
            # Compare grids
            if model_grid == expected_grid:
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                print("  Content validation: MISMATCH")
                print(f"  Got: {model_grid}")
                print(f"  Expected: {expected_grid}")
        except Exception as e:
            answer_score = -1.5
            print(f"  Content validation: ERROR - {str(e)}")
            print(f"  Got: {answer_text}")

    total_score = format_score + answer_score
    # print("\n" + "-"*80)
    # print(f" Final Score ".center(80, '-'))
    # print(f"  Format: {format_score}")
    # print(f"  Answer: {answer_score}")
    # print(f"  Total: {total_score}")
    # print("="*80 + "\n")

    # 返回包含组件分数的字典，而不只是总分
    return {
        'score': total_score,
        'extra_info': {
            'format_score': format_score,
            'answer_score': answer_score
        }
    }
