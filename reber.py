# https://web.archive.org/web/20100801214816/https://cnl.salk.edu/~schraudo/teach/NNcourse/figs/reber.gif
# TODO: find better labelling, make it clear to a reader which node means what
from collections import namedtuple
import random
import pandas as pd
from typing import List, Tuple

Edge = namedtuple("Edge", ["to", "get_str_list"])


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
        self,
        num_rows=10,
        valid_percentage=50,
        perturbed_percentage=5,
        symmetry_disturbed_percentage=20,
        random_percentage=25,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        # TODO: to motivate the max_length thing, describe the distribution of the strings, explain how it drops off, show a graph, say how you didn't want useless data
        # TODO: should I include random reber too?
        # TODO: make this max length some aspect of the embedder
        :param num_rows: total number of rows to generate
        :param valid_percentage: percentage of rows that follow embedded reber grammar
        :param perturbed_percentage: percentage of rows that have `self.num_perturbations` in-place edits corrupting an embedded reber
        :param symmetry_disturbed_percentage: percentage of rows in which the symmetry of the second and second to last indexes of a reber string is violated
        :param random_percentage: percentage of rows consisting of strings whose characters were randomly sampled from `self._reber_letters`
        :return: X, y where X is a (num_rows, self.max_length) matrix of strings
            encoded as lists of ints, each int representing the index of a character in
            `self._reber_letters`, padded with 0s at the end of each row; y is a vector
            of the corresponding labels for each row in X where 1 means that the string
            matches the reber grammar
        """
        if (
            sum(
                [
                    valid_percentage,
                    perturbed_percentage,
                    symmetry_disturbed_percentage,
                    random_percentage,
                ]
            )
            != 100
        ):
            raise ArithmeticError("Percentages must add up to 100.")
        num_perturbed_rows = round(num_rows * perturbed_percentage / 100)
        num_random_rows = round(num_rows * random_percentage / 100)
        num_symmetry_disturbed_rows = round(
            num_rows * symmetry_disturbed_percentage / 100
        )
        num_valid_rows = num_rows - (
            num_perturbed_rows + num_symmetry_disturbed_rows + num_random_rows
        )

        X = (
            pd.DataFrame(
                [
                    self._encode_as_unpadded_ints(self.make_embedded_reber_string())
                    for _ in range(num_valid_rows)
                ]
                + [
                    self._encode_as_unpadded_ints(
                        self.make_perturbed_embedded_reber_string()
                    )
                    for _ in range(num_perturbed_rows)
                ]
                + [
                    self._encode_as_unpadded_ints(
                        self._make_symmetry_disturbed_reber_string()
                    )
                    for _ in range(num_symmetry_disturbed_rows)
                ]
                + [
                    self._encode_as_unpadded_ints(self.make_random())
                    for _ in range(num_random_rows)
                ]
            )
            .fillna(value=0)  # 0 is a dummy encoding for padding
            .astype("int64")
        )
        y = pd.Series(
            [1] * num_valid_rows
            + [0] * (num_perturbed_rows + num_symmetry_disturbed_rows + num_random_rows)
        )
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
    X, y = reber.make_data(num_rows=100)
    print(X.head())
