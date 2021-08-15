from dataclasses import dataclass
from typing import List

from nlp_gym.envs.common.observation import BaseObservation


@dataclass(init=True)
class Observation(BaseObservation):
    src_sentence: str
    tent_translated_sentence: List[str]

    time_step: int
    total_steps: int

    @classmethod
    def build(cls, src_sentence: str, tent_translated_sentence: List,
              time_step: int, total_steps: int):
        observation = Observation(src_sentence, tent_translated_sentence, time_step, total_steps)
        return observation

    def get_updated_observation(self, action: str) -> 'Observation':
        self.tent_translated_sentence.append(action)
        return self
