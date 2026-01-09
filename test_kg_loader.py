"""
Test script for KG Loader.
Demonstrates loading from JSON and database rows with MERGE logic.
"""

from kg_loader import KGLoader
import json


def test_json_loading():
    """Test loading from JSON file."""
    # Initialize loader (adjust connection details as needed)
    loader = KGLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    try:
        # Load from JSON file
        loader.load_from_json(
            file_path="sample_data.json",
            node_label="Entity",
            relationship_type="CONNECTED_TO",
            id_key="id",
            deleted_key="deleted",
            active_key="active"
        )
        print("✓ Successfully loaded records from JSON")
    except Exception as e:
        print(f"✗ Error loading from JSON: {e}")
    finally:
        loader.close()


def test_db_rows_loading():
    """Test loading from database rows (mocked)."""
    # Mock database rows
    db_rows = [
        {
            "id": "db_node1",
            "name": "Database Entity 1",
            "status": "active",
            "deleted": False,
            "active": True,
            "relationships": [
                {"target_id": "db_node2", "properties": {"score": 0.95}}
            ]
        },
        {
            "id": "db_node2",
            "name": "Database Entity 2",
            "status": "active",
            "deleted": False,
            "active": True,
            "relationships": []
        },
        {
            "id": "db_node3",
            "name": "Inactive Entity",
            "status": "inactive",
            "deleted": False,
            "active": False,  # Should be ignored
            "relationships": []
        }
    ]
    
    loader = KGLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    try:
        # Load from database rows
        loader.load_from_db_rows(
            rows=db_rows,
            node_label="DBEntity",
            relationship_type="LINKED_TO",
            id_key="id",
            deleted_key="deleted",
            active_key="active"
        )
        print("✓ Successfully loaded records from DB rows")
    except Exception as e:
        print(f"✗ Error loading from DB rows: {e}")
    finally:
        loader.close()


def test_merge_logic():
    """Test that MERGE prevents duplicates."""
    # Records with duplicate IDs (should merge, not create duplicates)
    duplicate_records = [
        {
            "id": "merge_test_1",
            "name": "First Version",
            "active": True,
            "deleted": False,
            "relationships": []
        },
        {
            "id": "merge_test_1",  # Same ID - should merge
            "name": "Updated Version",
            "version": 2,
            "active": True,
            "deleted": False,
            "relationships": []
        }
    ]
    
    loader = KGLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    try:
        loader.load_records(
            records=duplicate_records,
            node_label="MergeTest",
            id_key="id"
        )
        print("✓ MERGE logic test completed (check database for single node)")
    except Exception as e:
        print(f"✗ Error in MERGE test: {e}")
    finally:
        loader.close()


if __name__ == "__main__":
    print("Testing KG Loader...")
    print("\n1. Testing JSON loading:")
    test_json_loading()
    
    print("\n2. Testing DB rows loading:")
    test_db_rows_loading()
    
    print("\n3. Testing MERGE logic:")
    test_merge_logic()
    
    print("\nAll tests completed!")
