from src.RBAMA import moral_judge
from src.RBAMA import translator
import logging
from tqdm import tqdm


def create_judge(judge_type):
    trans = translator.Translator()

    judge_map = {
        'prioR': moral_judge.JudgePrioR,
        'prioW': moral_judge.JudgePrioW,
        'onlyR': moral_judge.JudgeOnlyR,
        'onlyW': moral_judge.JudgeOnlyW
    }

    if judge_type not in judge_map:
        raise ValueError(f"Unknown judge type '{judge_type}'. Choose from {list(judge_map.keys())}.")

    return judge_map[judge_type](translator=trans)

