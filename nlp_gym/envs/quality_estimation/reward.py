from typing import List

from nlp_gym.envs.common.observation import BaseObservation
from nlp_gym.envs.common.reward import RewardFunction

from nltk.translate.bleu_score import sentence_bleu


class SimpleTranslationReward(RewardFunction):

    def __call__(self, observation: BaseObservation, action: str, targets: List[str]) -> float:
        if action == targets[-1] or action == "TER":
            reward = 1.0
        else:
            reward = 0.0
        return reward


class SimpleBLEUReward(RewardFunction):

    def __call__(self, observation: BaseObservation, action: str, targets: List[str]) -> float:
        print(f"Target: {targets} compared against: {observation.tent_translated_sentence + [action]}")
        reward = sentence_bleu(references=[targets], hypothesis=observation.tent_translated_sentence + [action])

        return reward
