from unittest import TestCase
from unittest.mock import patch, Mock

from reber import (
    ReberGenerator,
    ReberDataType,
    ReberMetadata,
    ReberDatatypeToPercentage,
)

MAX_LENGTH = 15


class TestReberDatatypeToPercentage(TestCase):
    def test_from_kwargs(self):
        ReberDatatypeToPercentage.from_kwargs(
            valid=25, perturbed=25, symmetry_disturbed=25, random=25
        )

    def test_from_kwargs_wrong_sum(self):
        with self.assertRaisesRegex(
            ArithmeticError, "Percentages must add up to exactly 100"
        ):
            ReberDatatypeToPercentage.from_kwargs(
                valid=125, perturbed=25, symmetry_disturbed=25, random=25
            )

    def test_init_missing_percentage_type(self):
        with self.assertRaisesRegex(
            ValueError, f"Missing {ReberDataType.RANDOM} percentage."
        ):
            ReberDatatypeToPercentage.from_kwargs(
                valid=40, perturbed=30, symmetry_disturbed=30
            )

    def test_row_counts(self):
        metadata = ReberMetadata(
            m_total=1000, datatype_to_percentage=ReberDatatypeToPercentage(),
        )
        datatype_expected_row_counts = [
            (ReberDataType.VALID, 500),
            (ReberDataType.PERTURBED, 50),
            (ReberDataType.SYMMETRY_DISTURBED, 400),
            (ReberDataType.RANDOM, 50),
        ]
        for datatype, expected_row_count in datatype_expected_row_counts:
            actual_row_count = metadata.get_num_rows_of(datatype)
            self.assertEqual(expected_row_count, actual_row_count)


class TestReberGenerator(TestCase):
    def test_make_data_shape(self):
        reber = ReberGenerator(MAX_LENGTH)
        m_total = 100
        X, y = reber.make_data(m_total=m_total)
        self.assertEqual((m_total, MAX_LENGTH), X.shape)
        self.assertEqual((m_total,), y.shape)

    def test_make_data_m_total_too_low(self):
        reber = ReberGenerator(MAX_LENGTH)
        with self.assertRaisesRegex(AssertionError, "m_total must be at least 100"):
            reber.make_data(m_total=1)

    def test_encode_as_padded_ints(self):
        reber = ReberGenerator(max_length=10)
        unencoded_string = "XBEPSTVX"
        encoded_string = reber.encode_as_padded_ints(unencoded_string)
        self.assertEqual([7, 1, 2, 3, 4, 5, 6, 7, 0, 0], encoded_string)

    # TODO: maybe make separate test classes instead of breaking it up like this
    # ------------------------- string generator fns
    @patch("reber.random.random", return_value=0)  # index to change = 1
    def test_symmetry_disturbed(self, _):
        # can't actually create a str of length 5 but whatever
        reber = ReberGenerator(max_length=5)
        reber._make_embedded_reber_list_of_correct_length = Mock(
            return_value=["B", "T", "S", "T", "E"]
        )
        sym_disturbed_str = reber.make_symmetry_disturbed_reber_string()
        self.assertEqual("BPSTE", sym_disturbed_str)

    @patch("reber.random.choice")
    @patch("reber.random.randrange")
    def test_make_random(self, mock_randrange, mock_choice):
        length_of_string = 8
        mock_randrange.return_value = length_of_string
        mock_choice.side_effect = ["B", "T", "S", "S", "V", "X", "T", "E"]
        reber = ReberGenerator(MAX_LENGTH)

        actual = reber.make_random()

        self.assertEqual("BTSSVXTE", actual)

    def test_perturb_str_list(self):
        # TODO: actually do this
        reber = ReberGenerator(max_length=MAX_LENGTH, num_perturbations=2)

    def test_randomly_inplace_edit_str_list(self):
        # TODO:
        pass

    def test_make_reber_str_list(self):
        pass

    # ------------------------- encode -------------
    def test_encode_as_padded_ints_invalid_chars(self):
        reber = ReberGenerator(MAX_LENGTH)
        with self.assertRaisesRegex(
            AssertionError,
            "Must pass in a string consisting of chars from the uppercase reber alphabet.",
        ):
            reber.encode_as_padded_ints(
                "oh bother, this string won't do, look at that exclamation point!"
            )

    def test_encode_as_padded_ints_string_too_long(self):
        reber = ReberGenerator(max_length=MAX_LENGTH)
        excessive_length = MAX_LENGTH + 3
        with self.assertRaisesRegex(
            AssertionError,
            f"String is length {excessive_length}; must be at most {MAX_LENGTH}",
        ):
            reber.encode_as_padded_ints("B" * excessive_length)
