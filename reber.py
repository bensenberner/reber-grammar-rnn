# https://web.archive.org/web/20100801214816/https://cnl.salk.edu/~schraudo/teach/NNcourse/figs/reber.gif
# TODO: find better labelling, make it clear to a reader which node means what
from collections import namedtuple
import random
from enum import Enum

import pandas as pd
from typing import List, Tuple

Edge = namedtuple("Edge", ["to", "get_str_list"])


class ReberDataType(Enum):
    VALID = "valid"  # valid embedded reber string
    PERTURBED = "perturbed"  # embedded reber string with some number of random in-place edits that render it invalid
    SYMMETRY_DISTURBED = "symmetry_disturbed"  # embedded reber string in which either the second or second to last index is modified to render it invalid
    RANDOM = (
        "random"  # string that consists of the reber alphabet but arranged randomly
    )

    def get_class_label(self):
        return 1 if self == self.VALID else 0


class ReberMetadata:
    # TODO: rename this?:
    def __init__(self, **datatype_to_percentage):
        """
        :param datatype_to_percentage:
            valid: percentage of rows that follow embedded reber grammar
            perturbed: percentage of rows that have `self.num_perturbations` in-place edits corrupting an embedded reber
            symmetry_disturbed: percentage of rows in which the symmetry of the second and second to last indexes of a reber string is violated
            random: percentage of rows consisting of strings whose characters were randomly sampled from `self._reber_letters`
        """
        if not datatype_to_percentage:
            datatype_to_percentage = {
                ReberDataType.VALID.value: 50,
                ReberDataType.PERTURBED.value: 5,
                ReberDataType.SYMMETRY_DISTURBED.value: 20,
                ReberDataType.RANDOM.value: 25,
            }
        if sum(datatype_to_percentage.values()) != 100:
            raise ArithmeticError("Percentages must add up to exactly 100")
        for datatype in ReberDataType:
            value = datatype.value
            if value not in datatype_to_percentage:
                raise ValueError(f"Missing {value} percentage.")
        self.datatype_to_percentage = datatype_to_percentage

    def get_datatype_to_row_count(self, total_num_rows):
        all_percentage_types_but_valid = {
            percentage_type
            for percentage_type in ReberDataType
            if percentage_type != ReberDataType.VALID
        }
        remaining_rows = total_num_rows
        row_counts = {}
        for percentage_type in all_percentage_types_but_valid:
            percentage = self.datatype_to_percentage[percentage_type.value]
            num_rows_of_this_type = round(total_num_rows * percentage / 100)
            row_counts[percentage_type.value] = num_rows_of_this_type
            remaining_rows -= num_rows_of_this_type
        num_valid_rows = remaining_rows
        row_counts[ReberDataType.VALID.value] = num_valid_rows

        assert total_num_rows == sum(row_counts.values())
        return row_counts


class ReberGenerator:
    _reber_start_node_idx = _embed_reber_start_node_idx = 0
    _reber_end_node_idx = _embed_reber_end_node_idx = 7
    _reber_letters = "BEPSTVX"
    _reber_letter_shifted_idx = {  # shifted so that 0 will represent a padding token
        char: idx + 1 for idx, char in enumerate(_reber_letters)
    }
    _reber_letters_set = set(_reber_letters)
    _reber_alternates = {
        "B": set(),
        "E": set(),
        "P": {"T", "V"},
        "T": {"P", "V"},
        "S": {"X"},
        "V": {"T", "P"},
        "X": {"S"},
    }

    def __init__(self, max_length: int = 25, num_perturbations: int = 1):
        self.max_length = max_length
        self.num_perturbations = num_perturbations
        self._datatype_to_string_making_fn = {  # TODO: is this the best place for it?
            ReberDataType.VALID.value: self.make_embedded_reber_string,
            ReberDataType.PERTURBED.value: self.make_perturbed_embedded_reber_string,
            ReberDataType.SYMMETRY_DISTURBED.value: self._make_symmetry_disturbed_reber_string,
            ReberDataType.RANDOM.value: self.make_random,
        }
        assert set(self._datatype_to_string_making_fn.keys()) == set(
            d.value for d in ReberDataType
        )
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

    def _make_reber_str_list(self, is_embedded) -> List[str]:
        """
        Generates a list of strings that represent a reber string or an embedded reber string
        """
        graph = self._embedded_reber_graph if is_embedded else self._reber_graph
        start_idx = (
            self._embed_reber_start_node_idx
            if is_embedded
            else self._reber_start_node_idx
        )
        end_idx = (
            self._embed_reber_end_node_idx if is_embedded else self._reber_end_node_idx
        )

        curr_node_idx = start_idx
        str_list = []
        while curr_node_idx != end_idx:
            edge = random.choice(graph[curr_node_idx])
            str_list.extend(edge.get_str_list())
            curr_node_idx = edge.to
        return str_list

    def _make_embedded_reber_list_of_correct_length(self) -> List[str]:
        """
        TODO: could exit early during list creation process
        instead of waiting for the whole thing to be created
        """
        while True:
            str_list = self._make_reber_str_list(is_embedded=True)
            if len(str_list) <= self.max_length:
                return str_list

    def _make_reber_list(self):
        return self._make_reber_str_list(is_embedded=False)

    def make_embedded_reber_string(self) -> str:
        return "".join(self._make_embedded_reber_list_of_correct_length())

    def make_perturbed_embedded_reber_string(self) -> str:
        """
        Creates a string, guaranteed not to match the reber grammar,
        that is `self.num_perturbations` edits different from a valid reber string
        """
        str_list = self._make_embedded_reber_list_of_correct_length()
        random_indices = random.sample(range(len(str_list)), k=self.num_perturbations)
        for random_index in random_indices:
            curr_letter = str_list[random_index]
            # only replace with a letter that will yield invalid reber
            possible_replacement_letters = (
                self._reber_letters_set
                - self._reber_alternates[curr_letter]
                - {curr_letter}
            )
            replacement_letter = random.sample(possible_replacement_letters, 1)[0]
            str_list[random_index] = replacement_letter
        return "".join(str_list)

    def make_random(self):
        min_embedded_reber_length = 8  # if you look at the grammar you see this is true
        num_chars = random.randrange(
            start=min_embedded_reber_length, stop=self.max_length
        )
        return "".join(
            random.sample(self._reber_letters_set, k=1)[0] for _ in range(num_chars)
        )

    def _make_symmetry_disturbed_reber_string(self) -> str:
        str_list = self._make_embedded_reber_list_of_correct_length()
        index_to_change = 1 if random.random() < 0.5 else -2
        # in embedded reber, the second and second to last characters are identical.
        # this is all we will change. these characters are either P or T.
        str_list[index_to_change] = "P" if str_list[index_to_change] == "T" else "T"
        return "".join(str_list)

    def _encode_as_unpadded_ints(self, string):
        return [self._reber_letter_shifted_idx[char] for char in string]

    def make_data(
        self, total_num_rows=10, **datatype_to_percentage,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        # TODO: to motivate the max_length thing, describe the distribution of the strings, explain how it drops off, show a graph, say how you didn't want useless data
        # TODO: should I include random reber too?
        # TODO: make this max length some aspect of the embedder
        :param total_num_rows: total number of rows to generate
        :return: X, y where X is a (num_rows, self.max_length) matrix of strings
            encoded as lists of ints, each int representing the index of a character in
            `self._reber_letters`, padded with 0s at the end of each row; y is a vector
            of the corresponding labels for each row in X where 1 means that the string
            matches the reber grammar
        """
        metadata = ReberMetadata(**datatype_to_percentage)
        datatypes_to_row_counts = metadata.get_datatype_to_row_count(total_num_rows)
        X_raw = []
        y_raw = []
        for datatype_enum in ReberDataType:
            datatype = datatype_enum.value
            string_making_fn = self._datatype_to_string_making_fn[datatype]
            num_rows_to_make = datatypes_to_row_counts[datatype]
            X_raw.extend(
                [
                    self._encode_as_unpadded_ints(string_making_fn())
                    for _ in range(num_rows_to_make)
                ]
            )
            class_label = datatype_enum.get_class_label()
            y_raw.extend([class_label] * num_rows_to_make)
        X = (
            pd.DataFrame(X_raw)
            .fillna(value=0)  # 0 is a dummy encoding for padding
            .astype("int64")
        )
        y = pd.Series(y_raw)
        if X.shape[1] != self.max_length:
            raise AssertionError(
                "No strings were generated that reached max_length. "
                "Try setting a higher num_rows or a lower max_length"
            )
        assert X.shape[0] == len(y)
        return X, y

    def encode_as_padded_ints(self, string):
        original_length = len(string)
        if original_length > self.max_length:
            raise Exception("Can't encode a string that long")
        padding_length = self.max_length - original_length
        return self._encode_as_unpadded_ints(string) + [0] * padding_length


if __name__ == "__main__":
    reber = ReberGenerator(max_length=15)
    X, y = reber.make_data(total_num_rows=100)
    print(X.head())
