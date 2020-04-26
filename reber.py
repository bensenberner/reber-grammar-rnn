# https://web.archive.org/web/20100801214816/https://cnl.salk.edu/~schraudo/teach/NNcourse/figs/reber.gif
# TODO: find better labelling, make it clear to a reader which node means what
from collections import namedtuple
import random
from enum import Enum

import pandas as pd
from typing import List, Tuple, Dict, Callable

Edge = namedtuple("Edge", ["to", "get_str_list"])
PADDING_VALUE = 0


class ReberDataType(Enum):
    VALID = "valid"  # valid embedded reber string
    PERTURBED = "perturbed"  # embedded reber string with some number of random edits (add, replace) that render it invalid
    SYMMETRY_DISTURBED = "symmetry_disturbed"  # embedded reber string in which either the second or second to last index is modified to render it invalid
    RANDOM = "random"  # string that is randomly sampled from the reber alphabet (not guaranteed invalid reber, but probably)

    def get_class_label(self):
        # all we care about is valid or invalid reber. All other classes are invalid, just in different ways
        return 1 if self == self.VALID else 0


class ReberDatatypeToPercentage:
    """
    Map of {ReberDataType: percentage} where percentage represents what proportion of data generated will be of that
    datatype. see ReberDataType for type details
    """

    def __init__(self, datatype_to_percentage: Dict[ReberDataType, int]):
        if not datatype_to_percentage:
            datatype_to_percentage = {
                ReberDataType.VALID: 50,
                ReberDataType.PERTURBED: 5,
                ReberDataType.SYMMETRY_DISTURBED: 40,  # these are the hardest to recognize
                ReberDataType.RANDOM: 5,
            }
        self._validate_datatype_to_percentage(datatype_to_percentage)
        self._map = datatype_to_percentage

    @classmethod
    def from_kwargs(cls, **kwargs):
        datatype_to_percentage = {ReberDataType(k): v for k, v in kwargs.items()}
        return cls(datatype_to_percentage)

    def get(self, datatype: ReberDataType):
        return self._map.get(datatype)

    @staticmethod
    def _validate_datatype_to_percentage(datatype_to_percentage):
        percentage_sum = sum(datatype_to_percentage.values())
        if percentage_sum != 100:
            raise ArithmeticError(
                f"Percentages must add up to exactly 100; was {percentage_sum}"
            )
        negative_percentage_exists = set(
            value for value in datatype_to_percentage.values() if value < 0
        )
        if negative_percentage_exists:
            raise ArithmeticError("A negative percentage doesn't make sense")
        for datatype in ReberDataType:
            if datatype not in datatype_to_percentage:
                raise ValueError(f"Missing {datatype} percentage.")


class ReberMetadata:
    def __init__(self, m_total: int, datatype_to_percentage: ReberDatatypeToPercentage):
        self._datatype_to_row_count = self._get_datatype_to_row_count(
            m_total, datatype_to_percentage
        )

    def _get_datatype_to_row_count(
        self, m_total: int, datatype_to_percentage: ReberDatatypeToPercentage
    ) -> Dict[ReberDataType, int]:
        """
        :return: a map of {ReberDataType: number_of_rows_for_that_type}
        """
        all_datatypes_but_valid = {
            percentage_type
            for percentage_type in ReberDataType
            if percentage_type != ReberDataType.VALID
        }
        remaining_rows = m_total
        row_counts = {}
        for datatype in all_datatypes_but_valid:
            percentage = datatype_to_percentage.get(datatype)
            num_rows_of_this_type = round(m_total * percentage / 100)
            row_counts[datatype] = num_rows_of_this_type
            remaining_rows -= num_rows_of_this_type
        num_valid_rows = remaining_rows
        row_counts[ReberDataType.VALID] = num_valid_rows

        assert m_total == sum(row_counts.values())
        return row_counts

    def get_num_rows_of(self, datatype: ReberDataType):
        return self._datatype_to_row_count[datatype]


class ReberGenerator:
    # start and end idxes are the same in both graphs for convenience
    _reber_start_node_idx = 0
    _reber_end_node_idx = 7
    _reber_letters = "BEPSTVX"
    _reber_letter_shifted_idx = {  # shifted so that 0 will represent a padding token
        char: idx + 1 for idx, char in enumerate(_reber_letters)
    }
    _reber_letters_set = set(_reber_letters)
    """
    used for performing random in-place replacements. the values represent the set of possible characters that might
    be able to replace the key character and still leave a valid reber in place (which would violate the assumption
    that all perturbations yield invalid reber strings). This lets us avoid replacing a character with a valid alternate
    """
    _reber_alternates = {
        "B": set(),
        "E": set(),
        "P": {"T", "V"},
        "T": {"P", "V"},
        "S": {"X"},
        "V": {"T", "P"},
        "X": {"S"},
    }
    """
    maps each char to a set of chars that could possibly *follow* that char in a reber string.
    Used for making invalid additions.
    """
    _reber_next_chars = {
        "": {"B"},  # represents the letter "before" the first B
        "B": {"T", "P"},
        "E": {"T", "P"},
        "P": {"E", "S", "T", "V", "X"},
        "S": {"E", "X", "S"},
        "T": {"E", "S", "X", "T", "V"},
        "V": {"P", "E"},
        "X": {"T", "V", "X", "S"},
    }

    def __init__(self, max_length: int, num_perturbations: int = 2):
        """
        # TODO: to motivate the max_length thing, describe the distribution of the strings, explain how it drops off, show a graph, say how you didn't want useless data
        :param max_length: the maximum length of any string, valid reber or otherwise, generated by `self.make_data`
        :param num_perturbations: the number of perturbations made by `self.make_perturbed_embedded_reber_string`
        """
        self.max_length = max_length
        self.num_perturbations = num_perturbations
        self._datatype_to_make_str_fn = {
            ReberDataType.VALID: self.make_valid_embedded_reber_string,
            ReberDataType.PERTURBED: self.make_perturbed_embedded_reber_string,
            ReberDataType.SYMMETRY_DISTURBED: self.make_symmetry_disturbed_reber_string,
            ReberDataType.RANDOM: self.make_random,
        }
        assert set(self._datatype_to_make_str_fn) == set(e for e in ReberDataType)
        self._reber_graph = {
            0: [Edge(to=1, get_str_list=lambda: ["B"])],
            1: [
                Edge(to=2, get_str_list=lambda: ["T"]),
                Edge(to=6, get_str_list=lambda: ["P"]),
            ],
            2: [
                Edge(to=2, get_str_list=lambda: ["S"]),
                Edge(to=3, get_str_list=lambda: ["X"]),
            ],
            3: [
                Edge(to=4, get_str_list=lambda: ["S"]),
                Edge(to=6, get_str_list=lambda: ["X"]),
            ],
            4: [Edge(to=7, get_str_list=lambda: ["E"])],
            5: [
                Edge(to=3, get_str_list=lambda: ["P"]),
                Edge(to=4, get_str_list=lambda: ["V"]),
            ],
            6: [
                Edge(to=6, get_str_list=lambda: ["T"]),
                Edge(to=5, get_str_list=lambda: ["V"]),
            ],
        }
        self._embedded_reber_graph = {
            0: [Edge(to=1, get_str_list=lambda: ["B"])],
            1: [
                Edge(to=2, get_str_list=lambda: ["T"]),
                Edge(to=6, get_str_list=lambda: ["P"]),
            ],
            2: [Edge(to=3, get_str_list=self._make_reber_list)],
            3: [Edge(to=4, get_str_list=lambda: ["T"])],
            4: [Edge(to=7, get_str_list=lambda: ["E"])],
            5: [Edge(to=4, get_str_list=lambda: ["P"])],
            6: [Edge(to=5, get_str_list=self._make_reber_list)],
        }

    def _make_reber_list(self) -> List[str]:
        return self._make_reber_str_list(is_embedded=False)

    def _randomly_inplace_edit_str_list(self, str_list: List[str]) -> List[str]:
        new_str_list = str_list[:]
        random_index = random.randrange(0, len(new_str_list))
        curr_letter = new_str_list[random_index]
        # only replace with a letter that will yield invalid reber
        possible_replacement_letters = list(
            self._reber_letters_set
            - self._reber_alternates[curr_letter]
            - {curr_letter}
        )
        replacement_letter = random.choice(possible_replacement_letters)

        new_str_list[random_index] = replacement_letter
        return new_str_list

    def _add_random_char_to_str_list(self, str_list: List[str]) -> List[str]:
        """
        Pick a random index and add that a random character before that index in the string to make the reber invalid
        TODO: explain why you're adding this (you tried adding a P at the end of a string and it was unable to determine
        that it wasn't invalid reber)
        """
        addition_idx = random.randrange(len(str_list) + 1)
        # the empty string is the char before str_list[0]
        letter_before_addition_idx = (
            str_list[addition_idx - 1] if addition_idx > 0 else ""
        )
        # only add letter that will yield invalid reber
        possible_letters_to_add = list(
            self._reber_letters_set - self._reber_next_chars[letter_before_addition_idx]
        )
        letter_to_add = random.choice(possible_letters_to_add)

        new_list = str_list[:]
        new_list.insert(addition_idx, letter_to_add)
        return new_list

    def _perturb_str_list(self, str_list: List[str]) -> None:
        # TODO: is it better to return new list or mutate old list? Faster? Easier to read?
        for _ in range(self.num_perturbations):
            possible_perturb_fns: List[Callable] = [
                self._randomly_inplace_edit_str_list
            ]
            if len(str_list) < self.max_length:
                possible_perturb_fns.append(self._add_random_char_to_str_list)
            perturb_fn = random.choice(possible_perturb_fns)
            str_list[:] = perturb_fn(str_list)

    def _make_reber_str_list(self, is_embedded) -> List[str]:
        """
        :return list of chars that represent a reber string or an embedded reber string
        """
        graph = self._embedded_reber_graph if is_embedded else self._reber_graph
        curr_node_idx = self._reber_start_node_idx
        str_list = []
        while curr_node_idx != self._reber_end_node_idx:
            edge = random.choice(graph[curr_node_idx])
            str_list.extend(edge.get_str_list())
            curr_node_idx = edge.to
        return str_list

    def _make_embedded_reber_list_of_correct_length(self) -> List[str]:
        while True:
            str_list = self._make_reber_str_list(is_embedded=True)
            if len(str_list) <= self.max_length:
                return str_list

    def make_valid_embedded_reber_string(self) -> str:
        return "".join(self._make_embedded_reber_list_of_correct_length())

    def make_perturbed_embedded_reber_string(self) -> str:
        """
        Creates a string, guaranteed not to match the reber grammar (i.e. to be invalid),
        that is `self.num_perturbations` edits different from a valid reber string
        """
        str_list = self._make_embedded_reber_list_of_correct_length()
        # TODO: you changed this up and didn't test it out dude!! Test it out!!
        self._perturb_str_list(str_list)
        return "".join(str_list)

    def make_random(self) -> str:
        """
        :return: a string whose characters are randomly sampled from the reber alphabet
        """
        min_embedded_reber_length = 8  # if you look at the grammar you see this is true
        num_chars = random.randrange(
            start=min_embedded_reber_length, stop=self.max_length
        )
        return "".join(
            random.choice(list(self._reber_letters_set)) for _ in range(num_chars)
        )

    def make_symmetry_disturbed_reber_string(self) -> str:
        """
        In the embedded reber grammar, the second and second to last chars are either both 'P' or both 'T'.
        :return: an invalid reber string, created by taking a valid reber string and changing only the second
            or second to last character
        """
        str_list = self._make_embedded_reber_list_of_correct_length()
        index_to_change = 1 if random.random() < 0.5 else -2
        str_list[index_to_change] = "P" if str_list[index_to_change] == "T" else "T"
        return "".join(str_list)

    def _create_rows_of_datatype(self, num_rows: int, datatype: ReberDataType):
        make_str: Callable = self._datatype_to_make_str_fn[datatype]
        return [
            self.encode_as_padded_ints(string=make_str(), safe=False)
            for _ in range(num_rows)
        ]

    def make_data(
        self, m_total: int, **kwargs: Dict[str, int]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :param m_total: total number of rows to generate
        :param kwargs: maps ReberDataTypes (the strings of the names of the enums) to percentages
        :return: X, y where X is a (m_total, self.max_length) matrix of strings encoded as lists of ints,
             each int representing the index of a character in `self._reber_letters`, padded with 0s at
             the end of each row
             y is a vector of the corresponding labels for each row in X where 1 means that the string
             matches the reber grammar
        """
        if m_total < 100:
            raise AssertionError(f"m_total must be at least 100; was only {m_total}")
        datatype_to_percentage = ReberDatatypeToPercentage.from_kwargs(**kwargs)
        metadata = ReberMetadata(m_total, datatype_to_percentage)
        X_raw = []
        y_raw = []
        for datatype in ReberDataType:
            num_rows = metadata.get_num_rows_of(datatype)
            X_raw.extend(self._create_rows_of_datatype(num_rows, datatype))
            y_raw.extend([datatype.get_class_label()] * num_rows)
        return pd.DataFrame(X_raw).astype("int64"), pd.Series(y_raw)

    def encode_as_padded_ints(self, string, safe=True) -> List[int]:
        """
        Used to easily create data for testing a model
        :param string: a string consisting of the reber alphabet
        :param safe: indicates whether to check the input
        :return:
        """
        if safe:
            unrecognized_letters = set(string) - self._reber_letters_set
            if unrecognized_letters:
                raise AssertionError(
                    "Must pass in a string consisting of chars from the uppercase reber alphabet."
                )
        original_length = len(string)
        if original_length > self.max_length:
            raise AssertionError(
                f"String is length {original_length}; must be at most {self.max_length}"
            )
        padding_length = self.max_length - original_length
        unpadded_ints = [self._reber_letter_shifted_idx[char] for char in string]
        return unpadded_ints + [PADDING_VALUE] * padding_length


if __name__ == "__main__":
    reber = ReberGenerator(max_length=20)
    X, y = reber.make_data(m_total=1000)
    print(X.head())
