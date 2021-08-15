import collections
import re
from typing import List, Optional, Counter


def pre_process_text(raw_data: List[str]) -> List[str]:
    '''Clean up raw data. Use lowercasing, (punctuation removal). Split into single tokens.
        Returns:
            List of words (and tokens) unordered and with duplicates
    '''
    lowered_data = [sentence.lower() for sentence in raw_data]
    tokens = [token for text in lowered_data for token in re.findall(r"\w+|[^\w\s]", text, re.UNICODE)]
    return tokens

def filter_most_frequent_tokens(words: List[str],
                                abs_number: Optional[int] = None,
                                min_freq: Optional[int] = None) -> List[str]:
    words = collections.Counter(words)
    if min_freq:
        words = {word: c for word, c in words.items() if c >= min_freq}

    sorted_words: List[str] = sort_words(words)
    if abs_number:
        sorted_words = sorted_words[:abs_number]

    return sorted_words


def sort_words(unsorted_words: Counter) -> List[str]:
    '''Sorts alphabetically first and then for frequency'''
    alphabet_sorted_words = sorted(unsorted_words.items(), key=lambda x: x[0], reverse=True)
    freq_sorted_words = sorted(alphabet_sorted_words, key=lambda x: x[1], reverse=True)
    return [word for word, freq in freq_sorted_words]
