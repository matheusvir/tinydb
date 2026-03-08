"""
Unit tests for the Bloom Filter implementation.

This module tests the standalone BloomFilter class in isolation,
covering initialization, add/test operations, false positive rates,
hash determinism, rebuild, and serialization.
"""

import pytest

from tinydb.bloom_filter import BloomFilter


# ---------------------------------------------------------------------------
# Initialization and parameter validation
# ---------------------------------------------------------------------------

class TestBloomFilterInit:
    def test_default_parameters(self):
        bf = BloomFilter()
        assert bf.count == 0
        assert bf.size > 0
        assert bf.num_hashes > 0

    def test_custom_parameters(self):
        bf = BloomFilter(expected_items=500, false_positive_rate=0.05)
        assert bf.count == 0
        assert bf.size > 0
        assert bf.num_hashes >= 1

    def test_invalid_expected_items_zero(self):
        with pytest.raises(ValueError, match='expected_items must be positive'):
            BloomFilter(expected_items=0)

    def test_invalid_expected_items_negative(self):
        with pytest.raises(ValueError, match='expected_items must be positive'):
            BloomFilter(expected_items=-10)

    def test_invalid_fp_rate_zero(self):
        with pytest.raises(ValueError, match='false_positive_rate must be between'):
            BloomFilter(false_positive_rate=0.0)

    def test_invalid_fp_rate_one(self):
        with pytest.raises(ValueError, match='false_positive_rate must be between'):
            BloomFilter(false_positive_rate=1.0)

    def test_invalid_fp_rate_negative(self):
        with pytest.raises(ValueError, match='false_positive_rate must be between'):
            BloomFilter(false_positive_rate=-0.1)

    def test_repr(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        r = repr(bf)
        assert 'BloomFilter' in r
        assert 'size=' in r
        assert 'hashes=' in r
        assert 'count=' in r


# ---------------------------------------------------------------------------
# Add and test operations
# ---------------------------------------------------------------------------

class TestBloomFilterAddAndTest:
    def test_add_and_test_single_item(self):
        bf = BloomFilter(expected_items=100)
        bf.add('hello')
        assert bf.test('hello') is True

    def test_add_and_test_multiple_items(self):
        bf = BloomFilter(expected_items=1000)
        items = [str(i) for i in range(100)]
        for item in items:
            bf.add(item)

        # All inserted items MUST be found (zero false negatives)
        for item in items:
            assert bf.test(item) is True

    def test_absent_item_returns_false(self):
        bf = BloomFilter(expected_items=100)
        bf.add('exists')
        assert bf.test('does_not_exist') is False

    def test_empty_filter_returns_false(self):
        bf = BloomFilter(expected_items=100)
        assert bf.test('anything') is False
        assert bf.test('') is False
        assert bf.test('42') is False

    def test_add_increments_count(self):
        bf = BloomFilter(expected_items=100)
        assert bf.count == 0
        bf.add('a')
        assert bf.count == 1
        bf.add('b')
        assert bf.count == 2

    def test_add_integer_items(self):
        bf = BloomFilter(expected_items=100)
        bf.add(42)
        assert bf.test(42) is True
        # Integers are converted to string internally
        assert bf.test('42') is True

    def test_zero_false_negatives(self):
        """
        The fundamental property of a Bloom Filter: if test() returns False,
        the item was NEVER added. We verify this over a large set.
        """
        bf = BloomFilter(expected_items=10_000, false_positive_rate=0.01)
        items = {str(i) for i in range(5000)}

        for item in items:
            bf.add(item)

        for item in items:
            assert bf.test(item) is True, \
                f'False negative detected for item {item}'


# ---------------------------------------------------------------------------
# False positive rate
# ---------------------------------------------------------------------------

class TestBloomFilterFalsePositiveRate:
    def test_false_positive_rate_within_bounds(self):
        """
        The observed false positive rate should be roughly within
        3x the configured rate for a sufficiently large sample.
        """
        n = 5000
        fp_rate = 0.01
        bf = BloomFilter(expected_items=n, false_positive_rate=fp_rate)

        for i in range(n):
            bf.add(f'item_{i}')

        # Test items that were NOT added
        false_positives = 0
        test_count = 10_000
        for i in range(n, n + test_count):
            if bf.test(f'item_{i}'):
                false_positives += 1

        observed_rate = false_positives / test_count
        assert observed_rate <= fp_rate * 3, \
            f'Observed FP rate {observed_rate:.4f} exceeds 3x target {fp_rate}'


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------

class TestBloomFilterHashDeterminism:
    def test_same_input_same_positions(self):
        bf = BloomFilter(expected_items=100)
        pos1 = bf._get_hash_values('test_item')
        pos2 = bf._get_hash_values('test_item')
        assert pos1 == pos2

    def test_different_inputs_different_positions(self):
        bf = BloomFilter(expected_items=100)
        pos_a = bf._get_hash_values('alpha')
        pos_b = bf._get_hash_values('beta')
        assert pos_a != pos_b


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

class TestBloomFilterRebuild:
    def test_rebuild_clears_old_items(self):
        bf = BloomFilter(expected_items=100)
        bf.add('old_item')
        assert bf.test('old_item') is True

        bf.rebuild(['new_item'])
        assert bf.test('new_item') is True
        assert bf.count == 1

    def test_rebuild_with_empty_list(self):
        bf = BloomFilter(expected_items=100)
        bf.add('item_1')
        bf.add('item_2')

        bf.rebuild([])
        assert bf.count == 0
        assert bf.test('item_1') is False
        assert bf.test('item_2') is False

    def test_rebuild_preserves_configuration(self):
        bf = BloomFilter(expected_items=500, false_positive_rate=0.05)
        original_size = bf.size
        original_hashes = bf.num_hashes

        bf.rebuild(['a', 'b', 'c'])
        assert bf.size == original_size
        assert bf.num_hashes == original_hashes


# ---------------------------------------------------------------------------
# Serialization (to_dict / from_dict)
# ---------------------------------------------------------------------------

class TestBloomFilterSerialization:
    def test_round_trip_preserves_state(self):
        bf = BloomFilter(expected_items=500, false_positive_rate=0.02)
        for i in range(100):
            bf.add(f'doc_{i}')

        data = bf.to_dict()
        restored = BloomFilter.from_dict(data)

        for i in range(100):
            assert restored.test(f'doc_{i}') is True

        assert restored.count == bf.count
        assert restored.size == bf.size
        assert restored.num_hashes == bf.num_hashes

    def test_to_dict_structure(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add('test')
        data = bf.to_dict()

        assert 'expected_items' in data
        assert 'fp_rate' in data
        assert 'size' in data
        assert 'num_hashes' in data
        assert 'bit_array' in data
        assert 'count' in data
        assert isinstance(data['bit_array'], list)

    def test_from_dict_empty_filter(self):
        bf = BloomFilter(expected_items=100)
        data = bf.to_dict()
        restored = BloomFilter.from_dict(data)

        assert restored.count == 0
        assert restored.test('anything') is False


# ---------------------------------------------------------------------------
# Optimal parameter calculations
# ---------------------------------------------------------------------------

class TestBloomFilterOptimalParameters:
    def test_larger_n_yields_larger_size(self):
        small = BloomFilter(expected_items=100, false_positive_rate=0.01)
        large = BloomFilter(expected_items=10_000, false_positive_rate=0.01)
        assert large.size > small.size

    def test_lower_fp_rate_yields_larger_size(self):
        high_fp = BloomFilter(expected_items=1000, false_positive_rate=0.1)
        low_fp = BloomFilter(expected_items=1000, false_positive_rate=0.001)
        assert low_fp.size > high_fp.size

    def test_num_hashes_is_positive(self):
        bf = BloomFilter(expected_items=1, false_positive_rate=0.5)
        assert bf.num_hashes >= 1
