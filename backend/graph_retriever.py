from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from backend.config import settings
from backend.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self, graph_store: Optional[GraphStore] = None):
        self.graph_store = graph_store or GraphStore()

    def find_symbol(self, repo_id: str, symbol_name: str) -> Optional[Dict[str, Any]]:
        graph = self.graph_store.get_graph(repo_id)
        if graph is None:
            return None
        node = graph.find_node(symbol_name)
        if node is None:
            return None
        return {
            "id": node.id,
            "repository": node.repository,
            "file": node.file,
            "type": node.type,
            "name": node.name,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "metadata": node.metadata,
        }

    def get_execution_flow(self, repo_id: str, symbol_name: str, depth: Optional[int] = None) -> List[Dict[str, Any]]:
        graph = self.graph_store.get_graph(repo_id)
        if graph is None:
            return []
        max_depth = depth if depth is not None else settings.graph_max_depth
        ordered = graph.get_execution_flow(symbol_name, depth=max_depth)
        found: List[Dict[str, Any]] = []
        for item in ordered:
            node = graph.find_node(item)
            if node is not None:
                found.append({
                    "id": node.id,
                    "file": node.file,
                    "name": node.name,
                    "type": node.type,
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                })
        return found

    def get_related_entities(self, repo_id: str, symbol_name: str, depth: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        graph = self.graph_store.get_graph(repo_id)
        if graph is None:
            return {"callers": [], "callees": [], "imports": []}
        max_depth = depth if depth is not None else settings.graph_max_depth
        node = graph.find_node(symbol_name)
        if node is None:
            return {"callers": [], "callees": [], "imports": []}

        callers = [self._serialize_graph_node(item) for item in graph.get_callers(symbol_name)]
        callees = [self._serialize_graph_node(item) for item in graph.get_callees(symbol_name)]
        imports = [self._serialize_graph_node(item) for item in graph.get_imports(symbol_name)]
        return {
            "callers": callers,
            "callees": callees,
            "imports": imports,
            "subgraph": [self._serialize_graph_node(item) for item in graph.get_subgraph(node.id, depth=max_depth)],
        }

    def build_context_for_query(self, repo_id: str, question: str, chunks: Sequence[Any]) -> str:
        if not chunks:
            return ""
        selected = []
        for chunk in chunks[:5]:
            name = chunk.metadata.name if getattr(chunk, 'metadata', None) is not None else None
            if name:
                selected.append(name)
        if not selected:
            return ""

        relevant = []
        for name in selected:
            flow = self.get_execution_flow(repo_id, name, depth=settings.graph_max_depth)
            if flow:
                relevant.append(f"{name}:\n" + " -> ".join([item['name'] for item in flow]))
            related = self.get_related_entities(repo_id, name)
            if related.get("callees"):
                relevant.append(f"{name} calls: " + ", ".join(item['name'] for item in related['callees']))
            if related.get("imports"):
                relevant.append(f"{name} imports: " + ", ".join(item['name'] for item in related['imports']))
        if not relevant:
            return ""
        return "\n".join(relevant)

    @staticmethod
    def _serialize_graph_node(node: Any) -> Dict[str, Any]:
        return {
            "id": node.id,
            "repository": node.repository,
            "file": node.file,
            "type": node.type,
            "name": node.name,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "metadata": node.metadata,
        }


def graph_retrieve(repo_id: str, symbol_name: str) -> Dict[str, Any]:
    return GraphRetriever().find_symbol(repo_id, symbol_name) or {}
