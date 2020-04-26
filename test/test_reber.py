from unittest import TestCase
from unittest.mock import patch, Mock

from reber import ReberGenerator, ReberDataType, ReberMetadata

MAX_LENGTH = 15


def make_fake_generator(max_length_and_generated_str_len=MAX_LENGTH):
    # TODO: is it right and proper to call this a 'fake'?
    reber = ReberGenerator(max_length=max_length_and_generated_str_len)
    """
    TODO: this is brittle bc it relies on being able to map the datatypes to arbitrary chars in the reber alphabet.
    if I were to add additional datatypes then I might run out. Need a better scheme. The problem is that I end up
    encoding the strings as ints and then relies on the assumption that all the chars that I'm encoding are in the
    reber alphabet. I guess I could do the integer encoding at the very end instead of when creating the dataframe.
    But a type change like that might be expensive. Hmm....
    # TODO: maybe I should mock out these functions instead of using these hacky lambdas
    """
    reber._datatype_to_make_str_fn = {
        ReberDataType.VALID.value: (
            lambda: ReberGenerator._reber_letters[0] * max_length_and_generated_str_len
        ),
        ReberDataType.PERTURBED.value: (
            lambda: ReberGenerator._reber_letters[1] * max_length_and_generated_str_len
        ),
        ReberDataType.RANDOM.value: (
            lambda: ReberGenerator._reber_letters[2] * max_length_and_generated_str_len
        ),
        ReberDataType.SYMMETRY_DISTURBED.value: (
            lambda: ReberGenerator._reber_letters[3] * max_length_and_generated_str_len
        ),
    }
    return reber


class TestReberMetadata(TestCase):
    def test_init(self):
        datatype_to_percentage = {
            ReberDataType.VALID.value: 25,
            ReberDataType.PERTURBED.value: 25,
            ReberDataType.SYMMETRY_DISTURBED.value: 25,
            ReberDataType.RANDOM.value: 25,
        }
        ReberMetadata(**datatype_to_percentage)

    def test_init_wrong_sum(self):
        datatype_to_percentage = {
            ReberDataType.VALID.value: 125,
            ReberDataType.PERTURBED.value: 125,
            ReberDataType.SYMMETRY_DISTURBED.value: 125,
            ReberDataType.RANDOM.value: 125,
        }
        with self.assertRaisesRegex(
            ArithmeticError, "Percentages must add up to exactly 100"
        ):
            ReberMetadata(**datatype_to_percentage)

    def test_init_missing_percentage_type(self):
        datatype_to_percentage = {
            ReberDataType.VALID.value: 30,
            ReberDataType.PERTURBED.value: 30,
            ReberDataType.SYMMETRY_DISTURBED.value: 40,
        }
        with self.assertRaisesRegex(
            ValueError, f"Missing {ReberDataType.RANDOM.value} percentage."
        ):
            ReberMetadata(**datatype_to_percentage)

    def test_row_counts(self):
        datatype_to_percentage = {
            ReberDataType.VALID.value: 10,
            ReberDataType.PERTURBED.value: 20,
            ReberDataType.SYMMETRY_DISTURBED.value: 30,
            ReberDataType.RANDOM.value: 40,
        }
        metadata = ReberMetadata(**datatype_to_percentage)

        actual_row_counts = metadata.get_datatype_to_row_count(total_num_rows=1000)

        expected_row_counts = {
            ReberDataType.VALID.value: 100,
            ReberDataType.PERTURBED.value: 200,
            ReberDataType.SYMMETRY_DISTURBED.value: 300,
            ReberDataType.RANDOM.value: 400,
        }
        self.assertEqual(expected_row_counts, actual_row_counts)


class TestReberGenerator(TestCase):
    def test_make_data_shape(self):
        reber = make_fake_generator()
        m_total = 100
        X, y = reber.make_data(m_total=m_total)
        self.assertEqual((m_total, MAX_LENGTH), X.shape)
        self.assertEqual((m_total,), y.shape)

    def test_make_data_no_long_enough_string_generated(self):
        """
        create a generator that (out of sheer luck) only ends up generating strings of exactly length 20
        but have a max_length of greater than 20
        """
        reber = make_fake_generator(max_length_and_generated_str_len=20)
        reber.max_length = 25
        with self.assertRaisesRegex(
            AssertionError, "No strings were generated that reached max_length"
        ):
            reber.make_data(100)

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

    @patch("reber.random.sample")
    @patch("reber.random.randrange")
    def test_make_random(self, mock_randrange, mock_sample):
        length_of_string = 8
        mock_randrange.return_value = length_of_string
        mock_sample.side_effect = ["B", "T", "S", "S", "V", "X", "T", "E"]
        reber = ReberGenerator(MAX_LENGTH)

        actual = reber.make_random()

        self.assertEqual("BTSSVXTE", actual)

    def test_perturb_str_list(self):
        # TODO: actually do this
        reber = ReberGenerator(max_length=MAX_LENGTH, num_perturbations=2)

    def test_randomly_inplace_edit_str_list(self):
        # TODO:
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
