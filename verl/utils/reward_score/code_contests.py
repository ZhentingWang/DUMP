import re
import hashlib
from typing import Dict, Tuple, Optional
import subprocess
import json
import time
import random
import signal
import sys
import os
import concurrent.futures

class OnlineJudge(object):
    def __init__(self):
        pass

    def check_language(self, code_string):
        if code_string.find("#include") != -1:
            return "c++"
        if code_string.find("raw_input") != -1:
            return "python2.7"
        if code_string.find("print ") != -1:
            return "python2.7"
        return "python3.10"
    
    def run_test_case(self, file_name, lan_bin, code_string, one_in, one_out):
        if lan_bin == "c++":
            p = subprocess.Popen("ulimit -c 0 && " + file_name,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                shell=True)
        else:
            p = subprocess.Popen([lan_bin, '-c', code_string], 
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid)
        input_string = one_in.encode()
        ce = False
        try:
            output, errors = p.communicate(input_string, timeout=1)
            if errors != b"":
                ce = True
                print(output, one_out, errors)
        except:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            output = b"" 
        score = 0
        try:
            if output.decode("utf-8").strip() == one_out.strip():
                return 1
        except:
            return 0
        return 0

    def check_ce(self, code_string):
        if code_string is None or code_string == "":
            return True
        file_name = "./tmp_code/" + str(random.randint(1, 10000000000))
        f = open(file_name + ".cpp", "w+")
        f.write(code_string)
        f.close()
        p = subprocess.Popen("c++ -o " + file_name + " " + file_name + ".cpp",
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                shell=True)
        ce = False
        try:
            output, errors = p.communicate("", timeout=10)
            if errors != b"":
                ce = True
                print(errors)
        except:
            ce = True
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            output = b"" 
        try:
            #os.remove(file_name + ".cpp")
            pass
        except:
            pass
        try:
            os.remove(file_name)
        except:
            pass
        return ce

    def compile(self, code_string):
        if code_string is None or code_string == "":
            return True
        md5_hash = hashlib.md5()
        md5_hash.update(code_string.encode("utf-8") + str(time.time()).encode("utf-8"))
        file_name = "./tmp/" + md5_hash.hexdigest()
        f = open(file_name + ".cpp", "w+")
        f.write(code_string)
        f.close()
        p = subprocess.Popen("c++ -o " + file_name + " " + file_name + ".cpp",
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                shell=True)
        ce = False
        try:
            output, errors = p.communicate("", timeout=10)
            if errors != b"":
                ce = True
        except:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            output = b"" 
        try:
            os.remove(file_name + ".cpp")
        except:
            pass
        return file_name

    def run(self, code_string, test_cases):
        lan_bin = self.check_language(code_string)
        all_tests = 0
        correct_tests = 0
        file_name = self.compile(code_string)
        input_case = test_cases["input"]
        output_case = test_cases["output"]
        cases = []
        for i in range(len(input_case)):
            cases.append((str(input_case[i]), str(output_case[i])))

        all_tests = len(cases)

        for (one_in, one_out) in cases:
            res = self.run_test_case(file_name, lan_bin, code_string, one_in, one_out)
            if res != 1:
                break
            else:
                correct_tests += 1
        try:
            os.remove(file_name)
        except:
            pass
        if all_tests == 0:
            return 0
        return int(5.0 * correct_tests / all_tests)
    
    def score(self, pid, code_string):
        all_tests, correct_tests = self.run(pid, code_string)
        return int(5.0 * correct_tests / all_tests)

oj = OnlineJudge()

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
        question_str = solution_str.split("Assistant:", 1)[0]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
        question_str = solution_str.split("<|im_start|>assistant", 1)[0]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str, ""

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str, question_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str, question_str

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    debug_str = []
    debug_str.append("\n[Structure Validation]")
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
        
        debug_str.append(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            debug_str.append(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        debug_str.append("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    elif processed_str.strip()[-len("</answer><|endoftext|>"):] != "</answer><|endoftext|>":
        debug_str.append("  [Error] Incorrect end token: Expected </answer><|endoftext|>")
        validation_passed = False
    elif processed_str.strip()[0:len("<think>")] != "<think>":
        debug_str.append("  [Error] Incorrect start token: Expected <think>")
        validation_passed = False
    else:
        debug_str.append("  Tag sequence validation passed")

    return validation_passed, debug_str

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    debug_str = []
    debug_str.append("\n" + "="*80)
    debug_str.append(" Processing New Sample ".center(80, '='))
    
    # Parse ground truth data
    solution_text = ""

    # Extract model answer
    answer_text, processed_str, question_str = extract_solution(solution_str)
    debug_str.append(f"\n[Question]\n{question_str}")
    debug_str.append(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct, debug_info = validate_response_structure(processed_str)
    debug_str.extend(debug_info)
    format_score = format_reward if format_correct else -abs(format_reward)
    debug_str.append(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    debug_str.append(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    if oj.check_ce(answer_text):
        debug_str.append("Compile Error!")
        answer_score = -2
    else:
        answer_score = oj.run(answer_text, ground_truth)

    total_score = format_score + answer_score
    debug_str.append("\n" + "-"*80)
    debug_str.append(f" Final Score ".center(80, '-'))
    debug_str.append(f"  Format: {format_score}")
    debug_str.append(f"  Answer: {answer_score}")
    debug_str.append(f"  Total: {total_score}")
    debug_str.append("="*80 + "\n")

    return total_score, "\n".join(debug_str)

if __name__ == "__main__":
    oj = OnlineJudge()
    print(oj.run("#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n  string s;\n  cin >> s;\n  for(int j = 0; j < 10000000; ++j) for (int i = 0; i < s.length(); i++) {\n    if (s[i] == s[i + 1] && s[i] == s[i + 2]) {\n      cout << s[i];\n      return 0;\n    }\n  }\n  cout << -1;\n  return 0;\n}\n", {"input": [123123123129912857127437128819329319200] * 200, "output":[1] * 200}))