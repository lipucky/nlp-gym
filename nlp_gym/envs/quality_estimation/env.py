import copy
from dataclasses import dataclass
from typing import Tuple, Union, List, Optional

import numpy as np

from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.common.base_env import BaseEnv
from nlp_gym.envs.common.observation import BaseObservation, BaseObservationFeaturizer
from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.envs.quality_estimation.observation import Observation
from nlp_gym.envs.quality_estimation.vocab import pre_process_text, filter_most_frequent_tokens
from nlp_gym.envs.quality_estimation.reward import SimpleTranslationReward
from rich import print

@dataclass(init=True)
class DataPoint:
    src_sentence: str
    tgt_sentence: List[str]
    observation: Observation


class NeuralMachineTranslationEnv(BaseEnv):
    def __init__(self,
                 corpus: List[str],
                 max_steps: int = 100,
                 reward_function: Optional[RewardFunction] = None,
                 observation_featurizer: Optional[BaseObservationFeaturizer] = None):

        self.current_sample: Optional[DataPoint] = None

        self.vocab = NeuralMachineTranslationEnv._create_vocab(corpus)
        self.action_space = NeuralMachineTranslationEnv._get_action_space(self.vocab)

        reward_function = SimpleTranslationReward() if reward_function is None else reward_function
        super().__init__(max_steps, reward_function, observation_featurizer)

        # set the counter
        self.time_step = None

        # observation time line
        self.__observation_sequence = None
        self.__current_target = None
        self.__current_observation = None

        # hold samples
        self.__samples: List[Sample] = []

    @staticmethod
    def _create_vocab(corpus: List[str], vocab_path: Optional[str] = None) -> List[str]:
        if vocab_path is None:
            raw_words = pre_process_text(corpus)
            vocab = filter_most_frequent_tokens(raw_words, min_freq=0)
        else:
            with open(vocab_path, "r") as f:
                file_content = f.read()
                vocab = file_content.splitlines()
        print("Vocabulary is of size ", len(vocab))
        return vocab

    @staticmethod
    def _get_action_space(vocab: List[str]) -> ActionSpace:
        actions = copy.deepcopy(vocab)
        actions.extend(["EOS", "TER"])  # Add special tokens to vocab
        action_space = ActionSpace(actions)
        return action_space

    def _is_terminal(self, action: str, time_step: int):
        termimal = (action == "ANSWER") or (time_step == len(self.__observation_sequence) - 1)
        return termimal

    def reset(self, sample: Sample = None) -> Union[BaseObservation, np.array]:
        if sample is None:
            sample = np.random.choice(self.__samples)

        self.time_step = 0

        observation = Observation.build(src_sentence=sample.input_text, tent_translated_sentence=[],
                                        time_step=0, total_steps=100)

        self.__current_observation = observation

        self.current_sample = DataPoint(src_sentence=sample.input_text,
                                        tgt_sentence=sample.oracle_label,
                                        observation=observation)
        return observation

    def render(self):
        print(f"[italic yellow]Step {self.time_step}[/italic yellow]")
        print(f"{self.current_sample.src_sentence} -> {self.current_sample.observation.tent_translated_sentence}")

    def close(self):
        pass

    def step(self, action: int) -> Tuple[Union[BaseObservation, np.array], int, bool, dict]:
        action_str = self.action_space.ix_to_action(action)

        tent_target_label = copy.deepcopy(self.current_sample.tgt_sentence[:self.time_step + 1])

        step_reward = self.reward_function(self.current_sample.observation, action_str, tent_target_label)

        self.time_step += 1

        done = action_str == "TER"
        # get the updated observation
        if not done:
            updated_observation = self.current_sample.observation.get_updated_observation(action_str)

            self.current_sample.observation = updated_observation
        else:
            updated_observation = self.current_sample.observation

        self.current_sample.observation = updated_observation

        return updated_observation, step_reward, done, {}

    # Methods for online learning and sampling
    def add_sample(self, sample: Sample):
        self.__samples.append(sample)

    def get_samples(self) -> List[Sample]:
        return self.__samples