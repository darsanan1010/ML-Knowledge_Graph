"""
Tests for KG Loader to verify acceptance criteria:
- Nodes are unique by ID
- Relationships are not duplicated
- Loader can be called multiple times safely
"""

from kg_loader import KGLoader


def test_nodes_unique_by_id():
    """Test that nodes are unique by ID."""
    loader = KGLoader()
    
    # Add a node
    loader.add_node("node1", {"name": "Alice", "age": 25})
    assert loader.node_count() == 1
    assert loader.get_node("node1") == {"name": "Alice", "age": 25}
    
    # Add same node ID with different data - should merge
    loader.add_node("node1", {"age": 30, "city": "NYC"})
    assert loader.node_count() == 1  # Still only one node
    assert loader.get_node("node1") == {"name": "Alice", "age": 30, "city": "NYC"}
    
    # Add different node
    loader.add_node("node2", {"name": "Bob"})
    assert loader.node_count() == 2
    
    print("✓ Test passed: Nodes are unique by ID")


def test_relationships_not_duplicated():
    """Test that relationships are not duplicated."""
    loader = KGLoader()
    
    # Add nodes
    loader.add_node("node1", {"name": "Alice"})
    loader.add_node("node2", {"name": "Bob"})
    
    # Add relationship
    loader.add_relationship("node1", "node2", "knows")
    assert loader.relationship_count() == 1
    
    # Add same relationship again - should not duplicate
    loader.add_relationship("node1", "node2", "knows")
    assert loader.relationship_count() == 1  # Still only one relationship
    
    # Add different relationship
    loader.add_relationship("node2", "node1", "knows")
    assert loader.relationship_count() == 2  # Different direction = different relationship
    
    # Add relationship without type
    loader.add_relationship("node1", "node2")
    assert loader.relationship_count() == 3  # (node1, node2, None) is different from (node1, node2, "knows")
    
    print("✓ Test passed: Relationships are not duplicated")


def test_loader_called_multiple_times_safely():
    """Test that loader can be called multiple times safely."""
    loader = KGLoader()
    
    # First load
    loader.load(
        nodes={"node1": {"name": "Alice"}, "node2": {"name": "Bob"}},
        relationships=[("node1", "node2", "knows")]
    )
    assert loader.node_count() == 2
    assert loader.relationship_count() == 1
    
    # Second load with overlapping data - should merge safely
    loader.load(
        nodes={"node1": {"age": 30}, "node3": {"name": "Charlie"}},  # node1 updated, node3 new
        relationships=[("node1", "node2", "knows"), ("node2", "node3", "knows")]  # first is duplicate
    )
    assert loader.node_count() == 3  # node1, node2, node3
    assert loader.get_node("node1") == {"name": "Alice", "age": 30}  # Merged data
    assert loader.relationship_count() == 2  # Only 2 unique relationships (duplicate ignored)
    
    # Third load with more overlapping data
    loader.load(
        nodes={"node1": {"city": "NYC"}, "node2": {"age": 25}},
        relationships=[("node1", "node2", "knows"), ("node1", "node3", "knows")]
    )
    assert loader.node_count() == 3  # Still 3 nodes
    assert loader.get_node("node1") == {"name": "Alice", "age": 30, "city": "NYC"}
    assert loader.get_node("node2") == {"name": "Bob", "age": 25}
    assert loader.relationship_count() == 3  # 3 unique relationships
    
    print("✓ Test passed: Loader can be called multiple times safely")


def test_batch_operations():
    """Test batch operations for nodes and relationships."""
    loader = KGLoader()
    
    # Add multiple nodes at once
    loader.add_nodes({
        "node1": {"name": "Alice"},
        "node2": {"name": "Bob"},
        "node3": {"name": "Charlie"}
    })
    assert loader.node_count() == 3
    
    # Add nodes as list of tuples
    loader.add_nodes([
        ("node4", {"name": "David"}),
        ("node1", {"age": 30})  # Update existing node
    ])
    assert loader.node_count() == 4
    assert loader.get_node("node1") == {"name": "Alice", "age": 30}
    
    # Add multiple relationships
    loader.add_relationships([
        ("node1", "node2", "knows"),
        ("node2", "node3", "knows"),
        ("node3", "node4", "knows"),
        ("node1", "node2", "knows")  # Duplicate
    ])
    assert loader.relationship_count() == 3  # Duplicate ignored
    
    print("✓ Test passed: Batch operations work correctly")


def test_edge_cases():
    """Test edge cases."""
    loader = KGLoader()
    
    # Node with no data
    loader.add_node("node1")
    assert loader.get_node("node1") == {}
    
    # Relationship with no type
    loader.add_relationship("node1", "node2")
    assert loader.has_relationship("node1", "node2", None)
    
    # Relationship with empty string type (different from None)
    loader.add_relationship("node2", "node3", "")
    assert loader.has_relationship("node2", "node3", "")
    assert loader.relationship_count() == 2
    
    # Check existence methods
    assert loader.has_node("node1")
    assert not loader.has_node("nonexistent")
    assert loader.has_relationship("node1", "node2", None)
    assert not loader.has_relationship("node1", "node2", "knows")
    
    print("✓ Test passed: Edge cases handled correctly")


def run_all_tests():
    """Run all tests."""
    print("Running KG Loader tests...\n")
    
    test_nodes_unique_by_id()
    test_relationships_not_duplicated()
    test_loader_called_multiple_times_safely()
    test_batch_operations()
    test_edge_cases()
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
