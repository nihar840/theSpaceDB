"""Tests for BlockStore."""

import tempfile
import pytest
from spacedb._core.block_store import BlockStore
from spacedb._core.models import MemoryBlock


@pytest.fixture
def store(tmp_dir):
    return BlockStore(tmp_dir)


class TestBlockStore:

    def test_put_get_roundtrip(self, store):
        """Append a block, then read it back by id."""
        block = MemoryBlock.create("hello", sensory_type="text")
        returned_id = store.append(block)

        assert returned_id == block.id

        got = store.read(block.id)
        assert got is not None
        assert got.id == block.id
        assert got.token == "hello"
        assert got.sensory_type == "text"

    def test_get_missing_returns_none(self, store):
        """Reading a non-existent block_id returns None."""
        assert store.read("nonexistent-id") is None

    def test_persistence(self, tmp_dir):
        """Data survives a store reload from the same directory."""
        block = MemoryBlock.create("persist me", sensory_type="text")

        store1 = BlockStore(tmp_dir)
        store1.append(block)

        # Create a new store instance pointing at the same dir
        store2 = BlockStore(tmp_dir)
        got = store2.read(block.id)

        assert got is not None
        assert got.token == "persist me"

    def test_all_blocks(self, store):
        """all_ids returns every appended block id."""
        b1 = MemoryBlock.create("one", sensory_type="text")
        b2 = MemoryBlock.create("two", sensory_type="text")
        b3 = MemoryBlock.create("three", sensory_type="audio")

        store.append(b1)
        store.append(b2)
        store.append(b3)

        ids = store.all_ids()
        assert len(ids) == 3
        assert set(ids) == {b1.id, b2.id, b3.id}
