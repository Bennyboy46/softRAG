import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from backend.models import GraphEdge, GraphNode
from backend.parser import CodeParser


GRAPH_RELATIONSHIPS = {
    "CALLS",
    "IMPORTS",
    "CONTAINS",
    "EXPOSES",
}


class RepositoryGraph:
    """Lightweight in-memory code graph storing repository entities and relationships."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        self._name_index: Dict[str, str] = {}

    def add_node(self, node: GraphNode) -> GraphNode:
        self.nodes[node.id] = node
        self._name_index.setdefault(node.name.lower(), node.id)
        return node

    def add_edge(self, source: str, target: str, relationship: str, metadata: Optional[Dict] = None) -> GraphEdge:
        if relationship not in GRAPH_RELATIONSHIPS:
            raise ValueError(f"Unsupported relationship type: {relationship}")
        edge = GraphEdge(source=source, target=target, relationship=relationship, metadata=metadata or {})
        self.edges[(source, target, relationship)] = edge
        return edge

    def find_node(self, value: str) -> Optional[GraphNode]:
        if value in self.nodes:
            return self.nodes[value]
        lowered = value.lower()
        candidate = self._name_index.get(lowered)
        if candidate:
            return self.nodes.get(candidate)
        for node in self.nodes.values():
            if node.file.lower() == lowered or node.name.lower() == lowered:
                return node
        return None

    def get_callers(self, symbol_name: str) -> List[GraphNode]:
        node = self.find_node(symbol_name)
        if not node:
            return []
        callers: List[GraphNode] = []
        for (source, target, relationship), edge in self.edges.items():
            if relationship == "CALLS" and target == node.id:
                caller = self.nodes.get(source)
                if caller:
                    callers.append(caller)
        return callers

    def get_callees(self, symbol_name: str) -> List[GraphNode]:
        node = self.find_node(symbol_name)
        if not node:
            return []
        callees: List[GraphNode] = []
        for (source, target, relationship), edge in self.edges.items():
            if relationship == "CALLS" and source == node.id:
                callee = self.nodes.get(target)
                if callee:
                    callees.append(callee)
        return callees

    def get_imports(self, symbol_name: str) -> List[GraphNode]:
        node = self.find_node(symbol_name)
        if not node:
            return []
        imports: List[GraphNode] = []
        for (source, target, relationship), edge in self.edges.items():
            if relationship == "IMPORTS" and source == node.id:
                imported = self.nodes.get(target)
                if imported:
                    imports.append(imported)
        return imports

    def get_methods_for_class(self, class_name: str) -> List[GraphNode]:
        class_node = self.find_node(class_name)
        if not class_node:
            return []
        methods: List[GraphNode] = []
        for (source, target, relationship), edge in self.edges.items():
            if relationship == "CONTAINS" and source == class_node.id:
                target_node = self.nodes.get(target)
                if target_node and target_node.type in {"method", "function"}:
                    methods.append(target_node)
        return methods

    def get_subgraph(self, node_id: str, depth: int = 2) -> List[GraphNode]:
        if node_id not in self.nodes:
            return []
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = [(node_id, 0)]
        nodes: List[GraphNode] = []
        while queue:
            current_id, current_depth = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            node = self.nodes.get(current_id)
            if node:
                nodes.append(node)
            if current_depth >= depth:
                continue
            for (source, target, relationship), _ in self.edges.items():
                if source == current_id:
                    queue.append((target, current_depth + 1))
                if target == current_id:
                    queue.append((source, current_depth + 1))
        return nodes

    def get_execution_flow(self, symbol_name: str, depth: int = 3) -> List[str]:
        node = self.find_node(symbol_name)
        if not node:
            return []
        seen: Set[str] = set()
        queue: List[Tuple[str, int]] = [(node.id, 0)]
        order: List[str] = []
        while queue:
            current_id, current_depth = queue.pop(0)
            if current_id in seen:
                continue
            seen.add(current_id)
            order.append(current_id)
            if current_depth >= depth:
                continue
            for (source, target, relationship), _ in self.edges.items():
                if source == current_id and relationship == "CALLS":
                    queue.append((target, current_depth + 1))
        return order

    def to_dict(self) -> Dict:
        return {
            "repo_id": self.repo_id,
            "nodes": [node.model_dump() for node in self.nodes.values()],
            "edges": [edge.model_dump() for edge in self.edges.values()],
        }


def _name_from_symbol(symbol_name: str) -> str:
    if not symbol_name:
        return "anonymous"
    return symbol_name.strip()


def _file_node_id(file_path: str) -> str:
    return f"{file_path}::file"


def _route_node_id(file_path: str, route: str) -> str:
    return f"{file_path}::route::{route}"


def build_repository_graph(repo_id: str, repo_dir: Path, files: Optional[Sequence[Path]] = None) -> RepositoryGraph:
    graph = RepositoryGraph(repo_id=repo_id)
    parser = CodeParser()
    file_list = list(files or repo_dir.rglob("*"))
    file_list = [p for p in file_list if p.is_file()]

    file_nodes_by_path: Dict[str, GraphNode] = {}
    for file_path in file_list:
        relative_path = file_path.relative_to(repo_dir).as_posix()
        file_node = GraphNode(
            id=_file_node_id(relative_path),
            repository=repo_id,
            file=relative_path,
            type="file",
            name=relative_path,
            start_line=1,
            end_line=1,
            metadata={"path": relative_path},
        )
        graph.add_node(file_node)
        file_nodes_by_path[relative_path] = file_node

    for file_path in file_list:
        relative_path = file_path.relative_to(repo_dir).as_posix()
        try:
            code = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        language = Path(relative_path).suffix.lower().lstrip(".")
        parsed_symbols = parser.parse(code, language if language else "python")
        for symbol in parsed_symbols:
            symbol_name = _name_from_symbol(symbol.name)
            node_id = f"{relative_path}::{symbol_name}"
            node = GraphNode(
                id=node_id,
                repository=repo_id,
                file=relative_path,
                type=symbol.type,
                name=symbol_name,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                metadata={"parent_name": symbol.parent_name or ""},
            )
            graph.add_node(node)
            graph.add_edge(file_node_id := _file_node_id(relative_path), node.id, "CONTAINS")

        for symbol in parsed_symbols:
            if symbol.name and symbol.parent_name:
                parent_id = f"{relative_path}::{symbol.parent_name}"
                if parent_id in graph.nodes:
                    graph.add_edge(parent_id, f"{relative_path}::{symbol.name}", "CONTAINS")

        route_edges = _extract_routes(code, relative_path, repo_id, graph)
        for edge in route_edges:
            graph.add_edge(edge[0], edge[1], edge[2])

        imported_targets = _extract_imports(code, relative_path, file_nodes_by_path)
        for target_file in imported_targets:
            if target_file in file_nodes_by_path:
                graph.add_edge(_file_node_id(relative_path), _file_node_id(target_file), "IMPORTS")

        function_nodes = [n for n in graph.nodes.values() if n.file == relative_path and n.type in {"function", "method"}]
        for function_node in function_nodes:
            callees = _extract_calls_for_symbol(code, relative_path, function_node.name, graph)
            for callee in callees:
                graph.add_edge(function_node.id, callee, "CALLS")

    return graph


def _extract_imports(code: str, file_path: str, file_nodes_by_path: Dict[str, GraphNode]) -> List[str]:
    imports: List[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", stripped)
        if not match:
            continue
        module_name = (match.group(1) or match.group(2) or "").strip()
        if not module_name:
            continue
        module_path = module_name.replace(".", "/")
        candidate_paths = {
            f"{module_path}.py",
            f"{module_path}/__init__.py",
            f"{module_path}.js",
            f"{module_path}.ts",
            f"{module_path}/index.js",
            f"{module_path}/index.ts",
        }
        for candidate in candidate_paths:
            if candidate in file_nodes_by_path:
                imports.append(candidate)
    return imports


def _extract_routes(code: str, file_path: str, repo_id: str, graph: RepositoryGraph) -> List[Tuple[str, str, str]]:
    routes: List[Tuple[str, str, str]] = []
    lines = code.splitlines()
    pattern = re.compile(r"@(?:\w+\.)?(get|post|put|delete|patch|route)\s*\(\s*['\"]([^'\"]+)['\"]")
    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if not match:
            continue
        method = match.group(1).upper()
        path = match.group(2)
        if not path:
            continue
        route_name = f"{method} {path}"
        route_id = _route_node_id(file_path, route_name)
        route_node = GraphNode(
            id=route_id,
            repository=repo_id,
            file=file_path,
            type="route",
            name=route_name,
            start_line=idx + 1,
            end_line=idx + 1,
            metadata={"method": method, "path": path},
        )
        graph.add_node(route_node)
        next_idx = idx + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1
        if next_idx < len(lines):
            def_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", lines[next_idx])
            if def_match:
                call_name = def_match.group(1)
                target_id = f"{file_path}::{call_name}"
                if target_id in graph.nodes:
                    routes.append((route_node.id, target_id, "EXPOSES"))
    return routes


def _extract_calls_for_symbol(code: str, file_path: str, symbol_name: str, graph: RepositoryGraph) -> List[str]:
    lines = code.splitlines()
    names = [node.name for node in graph.nodes.values() if node.file == file_path and node.type in {"function", "method"}]
    candidate_names = set(names)
    pattern = re.compile(r"(?<!\.)\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    observed: Set[str] = set()
    for line in lines:
        for match in pattern.findall(line):
            if match in {"if", "for", "while", "print", "return", "len", "str", "int", "float", "bool", "dict", "list", "set"}:
                continue
            if match in candidate_names and match != symbol_name:
                observed.add(match)
    matched_ids: List[str] = []
    for name in observed:
        target_id = f"{file_path}::{name}"
        if target_id in graph.nodes:
            matched_ids.append(target_id)
    return matched_ids
