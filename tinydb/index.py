"""
Contains the :class:`BTreeIndex` for optimized field lookups.

TinyDB's default behavior is to perform a linear scan (O(n)) for every
search operation. This module provides a B-Tree implementation that
improves search complexity to O(log n) without adding any external
dependencies.

Key features:
    - Supports duplicate keys by storing a list of document IDs for each key.
    - In-memory index that stays synchronized with TinyDB's storage.
    - Supports any comparable Python type (str, int, float, etc.).

.. note:: All values in a given indexed field should be of the same
          comparable type. Mixing incompatible types (e.g. ``int`` and
          ``str``) in the same field will cause ``TypeError`` on ``<``/``>``
          comparisons. The public API methods handle this gracefully by
          returning empty results or skipping the operation.
"""

from typing import Any, List, Optional, Tuple


class BTreeNode:
    """
    A node in the B-Tree.

    Each node stores a sorted list of keys and, if it is not a leaf, pointers
    to its children. The separation between leaf and internal nodes allows
    the tree to self-balance during insertions and deletions.
    """

    def __init__(self, leaf: bool = False):
        self.keys: List[Any] = []              # Sorted keys (indexed field values)
        self.doc_ids: List[List[int]] = []     # List of document IDs for each key
        self.children: List['BTreeNode'] = []  # Child nodes (empty if leaf)
        self.leaf: bool = leaf                 # Whether this is a leaf node


class BTreeIndex:
    """
    A B-Tree index for TinyDB.

    The ``order`` parameter controls the node capacity. Higher values result
    in wider nodes and fewer tree levels.
    """

    def __init__(self, order: int = 50):
        self.root: BTreeNode = BTreeNode(leaf=True)
        self.order: int = order

    # ------------------------------------------------------------------
    # Public API - these are the methods used by table.py
    # ------------------------------------------------------------------

    def insert(self, key: Any, doc_id: int) -> None:
        """
        Insert a document ID into the index under the given key.

        If the key already exists (another document has the same field value),
        the ``doc_id`` is appended to the existing list.

        If the key has an incompatible type with existing keys, the operation
        is silently skipped.

        :param key: The field value to index
        :param doc_id: The document ID to associate with the key
        """
        try:
            root = self.root

            # Check if the key already exists to avoid duplicate keys in the tree
            existing_node, existing_idx = self._find_key_node(root, key)
            if existing_node is not None:
                existing_node.doc_ids[existing_idx].append(doc_id)
                return

            # If the root is full, the tree grows upward
            if len(root.keys) == (2 * self.order) - 1:
                new_root = BTreeNode(leaf=False)
                new_root.children.append(self.root)
                self._split_child(new_root, 0)
                self.root = new_root

            self._insert_non_full(self.root, key, doc_id)
        except TypeError:
            pass

    def search(self, key: Any) -> List[int]:
        """
        Return all document IDs that match the given key value.

        :param key: The field value to search for
        :returns: List of matching document IDs, or empty list if none found
                  (also returns empty list on type mismatch)
        """
        try:
            return self._search_node(self.root, key)
        except TypeError:
            return []

    def delete(self, key: Any, doc_id: int) -> None:
        """
        Remove a specific document ID from the index for the given key.

        The key is only removed from the tree if no more document IDs are
        associated with it. Silently skipped on type mismatch.

        :param key: The field value to remove from
        :param doc_id: The document ID to remove
        """
        try:
            node, idx = self._find_key_node(self.root, key)
        except TypeError:
            return

        if node is None:
            return  # Key doesn't exist

        if doc_id in node.doc_ids[idx]:
            node.doc_ids[idx].remove(doc_id)

        # If no more documents for this key, remove it from the tree
        if not node.doc_ids[idx]:
            try:
                self._delete_key(self.root, key)
            except TypeError:
                pass

    def update(self, old_key: Any, new_key: Any, doc_id: int) -> None:
        """
        Update the index when a document's field value changes.

        Removes the old entry and creates a new one.

        :param old_key: The previous field value
        :param new_key: The new field value
        :param doc_id: The document ID being updated
        """
        self.delete(old_key, doc_id)
        self.insert(new_key, doc_id)

    def clear(self) -> None:
        """
        Clear the index completely. Called when the table is truncated.
        """
        self.root = BTreeNode(leaf=True)

    # ------------------------------------------------------------------
    # Internal B-Tree methods
    # ------------------------------------------------------------------

    def _find_key_node(
        self, node: BTreeNode, key: Any
    ) -> Tuple[Optional[BTreeNode], int]:
        """
        Find which node in the tree stores a given key.

        Traverses the node's keys from left to right to find the correct
        position, descending to the appropriate child if necessary.
        Returns the node and key index, or (None, -1) if not found.
        """
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return node, i

        if node.leaf:
            return None, -1

        return self._find_key_node(node.children[i], key)

    def _search_node(self, node: BTreeNode, key: Any) -> List[int]:
        """
        Traverse the tree looking for a key and return its document IDs.

        Similar logic to _find_key_node, but returns the data directly.
        """
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return list(node.doc_ids[i])  # Return a copy to avoid exposing internal state

        if node.leaf:
            return []

        return self._search_node(node.children[i], key)

    def _insert_non_full(self, node: BTreeNode, key: Any, doc_id: int) -> None:
        """
        Insert a key into a node that has available space.

        In leaf nodes, finds the correct position and inserts directly.
        In internal nodes, descends to the appropriate child, splitting
        it first if necessary.
        """
        i = len(node.keys) - 1

        if node.leaf:
            # Make room at the end and shift larger keys to the right
            node.keys.append(key)  # Placeholder, will be shifted
            node.doc_ids.append([])  # Placeholder, type-safe

            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.doc_ids[i + 1] = node.doc_ids[i]
                i -= 1

            node.keys[i + 1] = key
            node.doc_ids[i + 1] = [doc_id]
        else:
            # Find the correct child by traversing right to left
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            # If the child is full, split it before descending
            if len(node.children[i].keys) == (2 * self.order) - 1:
                self._split_child(node, i)
                # After the split there are two children where there was one
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key, doc_id)

    def _split_child(self, parent: BTreeNode, i: int) -> None:
        """
        Split a full child node, promoting the middle key to the parent.

        This is how the B-Tree grows: when a node becomes full, it splits
        in two and pushes a key upward, keeping the tree balanced.
        """
        order = self.order
        child = parent.children[i]
        new_node = BTreeNode(leaf=child.leaf)
        mid = order - 1

        # The middle key moves up to the parent
        parent.keys.insert(i, child.keys[mid])
        parent.doc_ids.insert(i, child.doc_ids[mid])
        parent.children.insert(i + 1, new_node)

        # The right half goes to the new node
        new_node.keys = child.keys[mid + 1:]
        new_node.doc_ids = child.doc_ids[mid + 1:]

        # The left half stays in the original node
        child.keys = child.keys[:mid]
        child.doc_ids = child.doc_ids[:mid]

        if not child.leaf:
            new_node.children = child.children[mid + 1:]
            child.children = child.children[:mid + 1]

    def _delete_key(self, node: BTreeNode, key: Any) -> None:
        """
        Remove a key from the tree, handling three possible scenarios:
        the key is in a leaf, in an internal node, or not in the current
        node (requiring descent to the correct child).

        Before descending, ensures the child has enough keys to handle
        a removal without violating B-Tree properties.
        """
        order = self.order
        i = 0

        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and node.keys[i] == key:
            if node.leaf:
                # Simplest case: key is in a leaf, remove directly
                node.keys.pop(i)
                node.doc_ids.pop(i)
            else:
                # Key is in an internal node: replace with predecessor
                # (the largest key from the left subtree) to maintain order
                pred_node, pred_idx = self._get_predecessor(node.children[i])
                node.keys[i] = pred_node.keys[pred_idx]
                node.doc_ids[i] = pred_node.doc_ids[pred_idx]
                self._delete_key(node.children[i], pred_node.keys[pred_idx])
        else:
            if node.leaf:
                return  # Key does not exist in the tree

            # Ensure the child has enough keys before descending
            if len(node.children[i].keys) < order:
                self._fill_child(node, i)
                if i > len(node.keys):
                    i -= 1

            self._delete_key(node.children[i], key)

    def _get_predecessor(self, node: BTreeNode) -> Tuple[BTreeNode, int]:
        """
        Find the largest key in the subtree by descending rightward.
        Used to replace a key removed from an internal node.
        """
        while not node.leaf:
            node = node.children[-1]
        return node, len(node.keys) - 1

    def _fill_child(self, parent: BTreeNode, i: int) -> None:
        """
        Ensure child i has enough keys before a removal.

        First tries to borrow from an adjacent sibling; if neither has
        spare keys, merges with a sibling.
        """
        order = self.order

        if i > 0 and len(parent.children[i - 1].keys) >= order:
            self._borrow_from_left(parent, i)
        elif i < len(parent.children) - 1 and len(parent.children[i + 1].keys) >= order:
            self._borrow_from_right(parent, i)
        else:
            if i < len(parent.children) - 1:
                self._merge_children(parent, i)
            else:
                self._merge_children(parent, i - 1)

    def _borrow_from_left(self, parent: BTreeNode, i: int) -> None:
        """
        Transfer the largest key from the left sibling to child i,
        passing through the parent to maintain correct ordering.
        """
        child = parent.children[i]
        sibling = parent.children[i - 1]

        child.keys.insert(0, parent.keys[i - 1])
        child.doc_ids.insert(0, parent.doc_ids[i - 1])

        if not child.leaf:
            child.children.insert(0, sibling.children.pop())

        parent.keys[i - 1] = sibling.keys.pop()
        parent.doc_ids[i - 1] = sibling.doc_ids.pop()

    def _borrow_from_right(self, parent: BTreeNode, i: int) -> None:
        """
        Transfer the smallest key from the right sibling to child i,
        passing through the parent to maintain correct ordering.
        """
        child = parent.children[i]
        sibling = parent.children[i + 1]

        child.keys.append(parent.keys[i])
        child.doc_ids.append(parent.doc_ids[i])

        if not child.leaf:
            child.children.append(sibling.children.pop(0))

        parent.keys[i] = sibling.keys.pop(0)
        parent.doc_ids[i] = sibling.doc_ids.pop(0)

    def _merge_children(self, parent: BTreeNode, i: int) -> None:
        """
        Merge child i with child i+1 into a single node, absorbing the
        parent key that separated them. If the parent is the root and
        becomes empty after the merge, the merged child becomes the new root.
        """
        child = parent.children[i]
        sibling = parent.children[i + 1]

        # The parent key that separated the two children descends to merged node
        child.keys.append(parent.keys[i])
        child.doc_ids.append(parent.doc_ids[i])

        child.keys.extend(sibling.keys)
        child.doc_ids.extend(sibling.doc_ids)
        if not child.leaf:
            child.children.extend(sibling.children)

        parent.keys.pop(i)
        parent.doc_ids.pop(i)
        parent.children.pop(i + 1)

        # Empty root means the tree shrunk one level
        if parent is self.root and len(parent.keys) == 0:
            self.root = child

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of unique keys in the index."""
        return self._count_keys(self.root)

    def _count_keys(self, node: BTreeNode) -> int:
        """Count keys recursively by traversing all nodes."""

        count = len(node.keys)
        for child in node.children:
            count += self._count_keys(child)
        return count

    def __repr__(self) -> str:
        return f"BTreeIndex(keys={len(self)})"
