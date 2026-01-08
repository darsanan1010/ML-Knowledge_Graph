"""
Knowledge Graph Loader

A KG loader that merges nodes and relationships without duplicates.
- Nodes are unique by ID
- Relationships are not duplicated
- Loader can be called multiple times safely
"""

from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union


class KGLoader:
    """
    Knowledge Graph Loader that maintains unique nodes and relationships.
    
    This loader ensures:
    - Nodes are unique by ID (duplicate IDs update existing nodes)
    - Relationships are not duplicated (same relationship added multiple times is ignored)
    - Safe to call load() multiple times with overlapping data
    """
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self._nodes: Dict[Any, Dict[str, Any]] = {}  # node_id -> node_data
        self._relationships: Set[Tuple[Any, Any, Optional[str]]] = set()  # (source_id, target_id, rel_type)
    
    def add_node(self, node_id: Any, node_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a node to the knowledge graph.
        
        If a node with the same ID already exists, its data will be merged
        (if both are dictionaries) or replaced.
        
        Args:
            node_id: Unique identifier for the node (must be hashable)
            node_data: Optional dictionary of data associated with the node
            
        Raises:
            TypeError: If node_id is not hashable
        """
        if node_data is None:
            node_data = {}
        
        if not isinstance(node_data, dict):
            raise TypeError("node_data must be a dictionary or None")
        
        # If node exists, merge the data; otherwise create new
        if node_id in self._nodes:
            self._nodes[node_id].update(node_data)
        else:
            self._nodes[node_id] = node_data.copy()
    
    def add_relationship(
        self, 
        source_id: Any, 
        target_id: Any, 
        relationship_type: Optional[str] = None
    ) -> None:
        """
        Add a relationship between two nodes.
        
        Relationships are stored as tuples (source_id, target_id, relationship_type)
        in a set, ensuring no duplicates even if called multiple times.
        
        Args:
            source_id: ID of the source node (must be hashable)
            target_id: ID of the target node (must be hashable)
            relationship_type: Optional type/name of the relationship
            
        Note:
            If source_id or target_id don't exist as nodes, the relationship
            is still added (nodes can be added later).
        """
        relationship = (source_id, target_id, relationship_type)
        self._relationships.add(relationship)
    
    def add_nodes(self, nodes: Union[Dict[Any, Dict[str, Any]], Iterable[Tuple[Any, Optional[Dict[str, Any]]]]]) -> None:
        """
        Add multiple nodes at once.
        
        Args:
            nodes: Either a dict mapping node_id -> node_data, or an iterable
                  of (node_id, node_data) tuples
                  
        Examples:
            >>> loader.add_nodes({"node1": {"name": "Alice"}, "node2": {"name": "Bob"}})
            >>> loader.add_nodes([("node1", {"name": "Alice"}), ("node2", {"name": "Bob"})])
        """
        if isinstance(nodes, dict):
            for node_id, node_data in nodes.items():
                self.add_node(node_id, node_data)
        else:
            for item in nodes:
                if isinstance(item, tuple) and len(item) >= 1:
                    node_id = item[0]
                    node_data = item[1] if len(item) > 1 else None
                    self.add_node(node_id, node_data)
                elif isinstance(item, dict) and "id" in item:
                    # Support dict format: {"id": "node1", "name": "Alice", ...}
                    node_id = item.pop("id")
                    self.add_node(node_id, item)
    
    def add_relationships(
        self, 
        relationships: Iterable[Union[Tuple[Any, Any], Tuple[Any, Any, Optional[str]]]]
    ) -> None:
        """
        Add multiple relationships at once.
        
        Args:
            relationships: Iterable of relationship tuples. Each tuple can be:
                          - (source_id, target_id) - relationship with no type
                          - (source_id, target_id, relationship_type) - relationship with type
                          
        Examples:
            >>> loader.add_relationships([("node1", "node2"), ("node2", "node3", "follows")])
        """
        for rel in relationships:
            if isinstance(rel, tuple) and len(rel) >= 2:
                source_id = rel[0]
                target_id = rel[1]
                relationship_type = rel[2] if len(rel) > 2 else None
                self.add_relationship(source_id, target_id, relationship_type)
    
    def load(
        self, 
        nodes: Optional[Union[Dict[Any, Dict[str, Any]], Iterable[Tuple[Any, Optional[Dict[str, Any]]]]]] = None,
        relationships: Optional[Iterable[Union[Tuple[Any, Any], Tuple[Any, Any, Optional[str]]]]] = None
    ) -> None:
        """
        Load nodes and relationships into the knowledge graph.
        
        This method can be called multiple times safely - it will merge
        new data without creating duplicates. Nodes with existing IDs will
        have their data merged, and duplicate relationships will be ignored.
        
        Args:
            nodes: Optional nodes to add (dict or iterable of tuples)
            relationships: Optional relationships to add (iterable of tuples)
            
        Examples:
            >>> loader.load(
            ...     nodes={"node1": {"name": "Alice"}, "node2": {"name": "Bob"}},
            ...     relationships=[("node1", "node2", "knows")]
            ... )
            >>> # Can be called again safely with overlapping data
            >>> loader.load(
            ...     nodes={"node1": {"age": 30}},  # Merges with existing node1
            ...     relationships=[("node1", "node2", "knows")]  # Duplicate ignored
            ... )
        """
        if nodes is not None:
            self.add_nodes(nodes)
        
        if relationships is not None:
            self.add_relationships(relationships)
    
    def get_node(self, node_id: Any) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node data dictionary if found, None otherwise
        """
        return self._nodes.get(node_id)
    
    def get_relationships(
        self, 
        node_id: Optional[Any] = None,
        direction: Optional[str] = None
    ) -> Set[Tuple[Any, Any, Optional[str]]]:
        """
        Get relationships, optionally filtered by node ID.
        
        Args:
            node_id: Optional node ID to filter relationships. If provided,
                    returns only relationships involving this node.
            direction: Optional filter by direction. If "outgoing", returns
                      relationships where node_id is the source. If "incoming",
                      returns relationships where node_id is the target.
                      If None, returns both.
                      
        Returns:
            Set of relationship tuples (source_id, target_id, relationship_type)
        """
        if node_id is None:
            return self._relationships.copy()
        
        if direction == "outgoing":
            return {rel for rel in self._relationships if rel[0] == node_id}
        elif direction == "incoming":
            return {rel for rel in self._relationships if rel[1] == node_id}
        else:
            return {
                rel for rel in self._relationships
                if rel[0] == node_id or rel[1] == node_id
            }
    
    def get_all_nodes(self) -> Dict[Any, Dict[str, Any]]:
        """
        Get all nodes.
        
        Returns:
            Dictionary of all nodes (node_id -> node_data)
        """
        return self._nodes.copy()
    
    def get_all_relationships(self) -> Set[Tuple[Any, Any, Optional[str]]]:
        """
        Get all relationships.
        
        Returns:
            Set of all relationships (tuples of source_id, target_id, relationship_type)
        """
        return self._relationships.copy()
    
    def node_count(self) -> int:
        """Return the number of unique nodes."""
        return len(self._nodes)
    
    def relationship_count(self) -> int:
        """Return the number of unique relationships."""
        return len(self._relationships)
    
    def has_node(self, node_id: Any) -> bool:
        """
        Check if a node exists.
        
        Args:
            node_id: ID of the node to check
            
        Returns:
            True if node exists, False otherwise
        """
        return node_id in self._nodes
    
    def has_relationship(
        self, 
        source_id: Any, 
        target_id: Any, 
        relationship_type: Optional[str] = None
    ) -> bool:
        """
        Check if a relationship exists.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Optional relationship type
            
        Returns:
            True if relationship exists, False otherwise
        """
        return (source_id, target_id, relationship_type) in self._relationships
    
    def clear(self) -> None:
        """Clear all nodes and relationships."""
        self._nodes.clear()
        self._relationships.clear()
    
    def __repr__(self) -> str:
        """String representation of the loader."""
        return f"KGLoader(nodes={self.node_count()}, relationships={self.relationship_count()})"
