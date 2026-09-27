from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.code_graph import RepositoryGraph


class GraphStore:
    """Repository-isolated in-memory graph store."""

    def __init__(self):
        self._graphs: Dict[str, RepositoryGraph] = {}

    def add_graph(self, repo_id: str, graph: RepositoryGraph) -> RepositoryGraph:
        self._graphs[repo_id] = graph
        return graph

    def get_graph(self, repo_id: str) -> Optional[RepositoryGraph]:
        return self._graphs.get(repo_id)

    def clear_repository(self, repo_id: str) -> None:
        self._graphs.pop(repo_id, None)

    def add_node(self, repo_id: str, node: Any) -> Any:
        graph = self.get_graph(repo_id)
        if graph is None:
            graph = RepositoryGraph(repo_id)
            self._graphs[repo_id] = graph
        return graph.add_node(node)

    def add_edge(self, repo_id: str, source: str, target: str, relationship: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        graph = self.get_graph(repo_id)
        if graph is None:
            graph = RepositoryGraph(repo_id)
            self._graphs[repo_id] = graph
        return graph.add_edge(source, target, relationship, metadata)

    def get_node(self, repo_id: str, node_id: str) -> Optional[Any]:
        graph = self.get_graph(repo_id)
        if graph is None:
            return None
        return graph.find_node(node_id)

    def get_callers(self, repo_id: str, symbol_name: str) -> List[Any]:
        graph = self.get_graph(repo_id)
        if graph is None:
            return []
        return graph.get_callers(symbol_name)

    def get_callees(self, repo_id: str, symbol_name: str) -> List[Any]:
        graph = self.get_graph(repo_id)
        if graph is None:
            return []
        return graph.get_callees(symbol_name)

    def get_imports(self, repo_id: str, symbol_name: str) -> List[Any]:
        graph = self.get_graph(repo_id)
        if graph is None:
            return []
        return graph.get_imports(symbol_name)

    def get_subgraph(self, repo_id: str, node_id: str, depth: int = 2) -> List[Any]:
        graph = self.get_graph(repo_id)
        if graph is None:
            return []
        return graph.get_subgraph(node_id, depth=depth)

    def get_repository_graph(self, repo_id: str) -> Optional[RepositoryGraph]:
        return self.get_graph(repo_id)
