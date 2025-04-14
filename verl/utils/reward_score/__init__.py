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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Default reward computation function.
    
    Returns:
        Union[float, Dict[str, Any]]: Either a float score or a dictionary with 'score' and optional 'extra_info'
    """
    if 'gsm8k_r1format' in data_source:
        from .import gsm8k_r1format
        res = gsm8k_r1format.compute_score(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif 'kk' in data_source:
        from .import kk
        res = kk.compute_score(solution_str, ground_truth)
    elif data_source == 'arc_logic':
        from . import arc_logic
        res = arc_logic.compute_score(solution_str, ground_truth)
    elif data_source in ['logicnli','logiqa','longiclbench_goemotion','lsat_ar','proofwriter','ruletaker']:
        from . import multiple_choice_r1format
        res = multiple_choice_r1format.compute_score(solution_str, ground_truth)
    elif data_source in ['probabilistic_colored_1000','probabilistic_base_1000']:
        from . import number_r1format
        res = number_r1format.compute_score(solution_str, ground_truth)
    elif 'code_contests' in data_source:
        from .import code_contests
        res = code_contests.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    elif isinstance(res, dict):
        return res
    else:
        return float(res[0])
