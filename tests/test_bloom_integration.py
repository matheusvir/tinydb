"""
Integration tests for the Bloom Filter within TinyDB tables.

These tests verify that TinyDB tables behave correctly with the Bloom Filter
enabled by default, including lazy initialization, insert/get/contains
short-circuits, remove/truncate rebuilds, and full workflows.
"""

from pathlib import Path

import pytest

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage, JSONStorage
from tinydb.table import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=['memory', 'json'])
def db(request, tmp_path: Path):
    """
    TinyDB instance with bloom filter active (the default).
    Parametrized to test with both MemoryStorage and JSONStorage.
    """
    if request.param == 'json':
        db_ = TinyDB(tmp_path / 'test_bloom.db', storage=JSONStorage)
    else:
        db_ = TinyDB(storage=MemoryStorage)

    db_.drop_tables()
    db_.insert_multiple({'int': 1, 'char': c} for c in 'abc')
    yield db_
    db_.close()


@pytest.fixture
def table():
    """A standalone bloom-enabled table for focused tests."""
    db_ = TinyDB(storage=MemoryStorage)
    tbl = db_.table('bloom_test')  # bloom_filter=True by default
    yield tbl
    db_.close()


# ---------------------------------------------------------------------------
# Bloom Filter default activation
# ---------------------------------------------------------------------------

class TestBloomDefault:
    def test_bloom_enabled_by_default(self):
        db_ = TinyDB(storage=MemoryStorage)
        tbl = db_.table('_default')
        assert tbl._bloom is not None
        assert tbl._use_bloom is True
        db_.close()

    def test_bloom_can_be_disabled(self):
        db_ = TinyDB(storage=MemoryStorage)
        tbl = db_.table('no_bloom', bloom_filter=False)
        assert tbl._bloom is None
        assert tbl._use_bloom is False
        db_.close()


# ---------------------------------------------------------------------------
# Lazy initialization
# ---------------------------------------------------------------------------

class TestBloomLazyInit:
    def test_not_initialized_on_construction(self):
        db_ = TinyDB(storage=MemoryStorage)
        tbl = db_.table('lazy')
        # Filter object exists but hasn't been populated yet
        assert tbl._bloom is not None
        assert tbl._bloom_initialized is False
        db_.close()

    def test_initialized_after_first_read(self):
        db_ = TinyDB(storage=MemoryStorage)
        tbl = db_.table('lazy')
        tbl.insert({'val': 1})

        # Force a _read_table call via all()
        tbl.all()
        assert tbl._bloom_initialized is True
        db_.close()

    def test_initialized_after_get_by_id(self):
        db_ = TinyDB(storage=MemoryStorage)
        tbl = db_.table('lazy')
        doc_id = tbl.insert({'val': 1})

        # get(doc_id=...) calls _read_table, which initializes the bloom
        tbl.get(doc_id=doc_id)
        assert tbl._bloom_initialized is True
        db_.close()


# ---------------------------------------------------------------------------
# Insert updates the Bloom Filter
# ---------------------------------------------------------------------------

class TestBloomInsert:
    def test_insert_updates_bloom(self, table):
        doc_id = table.insert({'name': 'alice'})

        # Trigger lazy init by reading
        table.all()

        assert table._bloom.test(str(doc_id)) is True

    def test_insert_multiple_updates_bloom(self, table):
        doc_ids = table.insert_multiple([
            {'name': 'alice'},
            {'name': 'bob'},
            {'name': 'charlie'},
        ])

        # Trigger lazy init
        table.all()

        for doc_id in doc_ids:
            assert table._bloom.test(str(doc_id)) is True

    def test_insert_with_doc_id_updates_bloom(self, table):
        table.insert(Document({'name': 'alice'}, 42))
        table.all()
        assert table._bloom.test('42') is True


# ---------------------------------------------------------------------------
# get() and contains() use Bloom Filter for fast negative lookups
# ---------------------------------------------------------------------------

class TestBloomGetAndContains:
    def test_get_nonexistent_doc_id(self, table):
        table.insert({'name': 'alice'})
        result = table.get(doc_id=9999)
        assert result is None

    def test_get_existing_doc_id(self, table):
        doc_id = table.insert({'name': 'alice'})
        result = table.get(doc_id=doc_id)
        assert result is not None
        assert result['name'] == 'alice'

    def test_contains_nonexistent_doc_id(self, table):
        table.insert({'name': 'alice'})
        assert table.contains(doc_id=9999) is False

    def test_contains_existing_doc_id(self, table):
        doc_id = table.insert({'name': 'alice'})
        assert table.contains(doc_id=doc_id) is True

    def test_get_multiple_doc_ids(self, table):
        ids = table.insert_multiple([
            {'name': 'alice'},
            {'name': 'bob'},
        ])
        docs = table.get(doc_ids=ids)
        assert len(docs) == 2

    def test_get_by_condition_still_works(self, table):
        table.insert({'name': 'alice', 'age': 30})
        table.insert({'name': 'bob', 'age': 25})

        result = table.get(where('name') == 'alice')
        assert result is not None
        assert result['name'] == 'alice'

    def test_search_still_works(self, table):
        table.insert({'name': 'alice', 'age': 30})
        table.insert({'name': 'bob', 'age': 30})
        table.insert({'name': 'charlie', 'age': 25})

        results = table.search(where('age') == 30)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Remove and truncate rebuild the Bloom Filter
# ---------------------------------------------------------------------------

class TestBloomRemove:
    def test_remove_by_doc_id_rebuilds_bloom(self, table):
        doc_id = table.insert({'name': 'alice'})
        # Ensure bloom is populated
        table.all()
        assert table._bloom.test(str(doc_id)) is True

        table.remove(doc_ids=[doc_id])
        # After rebuild, the removed ID should no longer be in the filter
        assert table._bloom.test(str(doc_id)) is False

    def test_remove_by_condition_rebuilds_bloom(self, table):
        doc_id = table.insert({'name': 'alice'})
        table.insert({'name': 'bob'})

        table.remove(where('name') == 'alice')
        assert table.get(doc_id=doc_id) is None

    def test_truncate_clears_bloom(self, table):
        table.insert_multiple([
            {'name': 'alice'},
            {'name': 'bob'},
            {'name': 'charlie'},
        ])
        table.all()  # ensure bloom is initialized
        assert table._bloom.count > 0

        table.truncate()
        assert table._bloom.count == 0


# ---------------------------------------------------------------------------
# Update operations
# ---------------------------------------------------------------------------

class TestBloomUpdate:
    def test_update_does_not_break_bloom(self, table):
        doc_id = table.insert({'name': 'alice', 'age': 30})
        table.update({'age': 31}, where('name') == 'alice')

        # The doc_id should still be found
        result = table.get(doc_id=doc_id)
        assert result is not None
        assert result['age'] == 31


# ---------------------------------------------------------------------------
# Full workflow with both storage backends
# ---------------------------------------------------------------------------

class TestBloomFullWorkflow:
    def test_full_workflow(self, db: TinyDB):
        # Verify initial data from fixture
        assert len(db) == 3

        # Insert
        doc_id = db.insert({'int': 2, 'char': 'd'})
        assert len(db) == 4

        # Get by ID
        doc = db.get(doc_id=doc_id)
        assert doc is not None
        assert doc['char'] == 'd'

        # Get nonexistent ID
        assert db.get(doc_id=99999) is None

        # Contains
        assert db.contains(doc_id=doc_id) is True
        assert db.contains(doc_id=99999) is False

        # Search
        results = db.search(where('int') == 1)
        assert len(results) == 3

        # Update
        db.update({'int': 10}, where('char') == 'a')
        assert db.get(where('int') == 10)['char'] == 'a'

        # Remove
        db.remove(where('char') == 'd')
        assert db.contains(doc_id=doc_id) is False

        # Count
        assert db.count(where('int') == 1) == 2

    def test_all_returns_all_documents(self, db: TinyDB):
        docs = db.all()
        assert len(docs) == 3

    def test_upsert_with_bloom(self, db: TinyDB):
        # Upsert existing document
        db.upsert({'int': 1, 'char': 'a', 'extra': True},
                   where('char') == 'a')
        doc = db.get(where('char') == 'a')
        assert doc['extra'] is True

        # Upsert new document
        db.upsert({'int': 5, 'char': 'z'}, where('char') == 'z')
        assert db.contains(where('char') == 'z')


# ---------------------------------------------------------------------------
# Bloom filter initialization from existing data
# ---------------------------------------------------------------------------

class TestBloomInitFromExistingData:
    def test_bloom_initialized_from_existing_data(self):
        """
        When opening a table that already has data, the lazy init should
        populate the bloom filter with existing doc_ids on first read.
        """
        db_ = TinyDB(storage=MemoryStorage)

        # Insert data with default table
        ids = db_.insert_multiple([
            {'name': 'alice'},
            {'name': 'bob'},
            {'name': 'charlie'},
        ])

        # Force new table instance by clearing the table cache
        del db_._tables['_default']
        tbl = db_.table('_default')

        # Trigger lazy init via a read
        tbl.all()

        # The bloom filter should know about existing doc_ids
        for doc_id in ids:
            assert tbl._bloom.test(str(doc_id)) is True

        # Nonexistent ID should not be found
        assert tbl._bloom.test('99999') is False
        db_.close()

    def test_bloom_on_empty_table(self):
        db_ = TinyDB(storage=MemoryStorage)
        tbl = db_.table('empty')
        assert tbl.get(doc_id=1) is None
        db_.close()


# ---------------------------------------------------------------------------
# Bloom disabled: ensure no behavioral difference
# ---------------------------------------------------------------------------

class TestBloomDisabled:
    def test_full_workflow_without_bloom(self, tmp_path: Path):
        db_ = TinyDB(storage=MemoryStorage)
        tbl = db_.table('no_bloom', bloom_filter=False)

        doc_id = tbl.insert({'name': 'alice'})
        assert tbl.get(doc_id=doc_id)['name'] == 'alice'
        assert tbl.get(doc_id=9999) is None
        assert tbl.contains(doc_id=doc_id) is True
        assert tbl.contains(doc_id=9999) is False

        tbl.remove(doc_ids=[doc_id])
        assert tbl.get(doc_id=doc_id) is None
        db_.close()
