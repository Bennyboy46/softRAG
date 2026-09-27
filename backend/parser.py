import logging
import re
from typing import Dict, List, Optional
from tree_sitter import Language, Node, Parser

from backend.models import ParsedSymbol

logger = logging.getLogger(__name__)

# Try loading tree-sitter language grammars
LANGUAGES: Dict[str, Language] = {}
try:
    import tree_sitter_python
    LANGUAGES["python"] = Language(tree_sitter_python.language())
except Exception as e:
    logger.warning(f"Could not load tree-sitter-python: {e}")

try:
    import tree_sitter_javascript
    LANGUAGES["javascript"] = Language(tree_sitter_javascript.language())
except Exception as e:
    logger.warning(f"Could not load tree-sitter-javascript: {e}")

try:
    import tree_sitter_typescript
    LANGUAGES["typescript"] = Language(tree_sitter_typescript.language_typescript())
    LANGUAGES["tsx"] = Language(tree_sitter_typescript.language_tsx())
except Exception as e:
    logger.warning(f"Could not load tree-sitter-typescript: {e}")


class CodeParser:
    """
    Structural code parser utilizing Tree-sitter AST parsing for supported languages
    (Python, JavaScript, TypeScript, TSX), with robust domain parsers for Markdown
    and SQL, and a graceful regex-based fallback for other languages (C, C++, Java).
    """

    def __init__(self):
        self._parsers: Dict[str, Parser] = {}
        for lang_name, lang_grammar in LANGUAGES.items():
            try:
                self._parsers[lang_name] = Parser(lang_grammar)
            except Exception as e:
                logger.warning(f"Failed to create Parser for {lang_name}: {e}")

    def parse(self, code: str, language: str) -> List[ParsedSymbol]:
        """Parse source code string into a list of structured AST symbols."""
        if not code.strip():
            return []

        lang = language.lower()
        code_bytes = code.encode("utf-8")

        # 1. Tree-sitter parser if grammar available
        if lang in self._parsers:
            try:
                return self._parse_tree_sitter(code_bytes, lang)
            except Exception as e:
                logger.warning(f"Tree-sitter parse failed for {lang}, using fallback: {e}")

        # 2. Markdown dedicated parser
        if lang == "markdown":
            return self._parse_markdown(code)

        # 3. SQL dedicated parser
        if lang == "sql":
            return self._parse_sql(code)

        # 4. Fallback parser for languages without tree-sitter grammar (C, C++, Java, etc.)
        return self._parse_fallback(code, lang)

    def _parse_tree_sitter(self, code_bytes: bytes, language: str) -> List[ParsedSymbol]:
        """Extract functions, classes, and methods using Tree-sitter AST nodes."""
        parser = self._parsers[language]
        tree = parser.parse(code_bytes)
        root = tree.root_node

        symbols: List[ParsedSymbol] = []

        if language == "python":
            self._extract_python_symbols(root, code_bytes, symbols)
        elif language in ("javascript", "typescript", "tsx"):
            self._extract_js_ts_symbols(root, code_bytes, symbols)

        return symbols

    def _extract_python_symbols(
        self,
        node: Node,
        code_bytes: bytes,
        symbols: List[ParsedSymbol],
        parent_name: Optional[str] = None,
    ):
        """Recursively traverse Python AST nodes."""
        for child in node.children:
            # Handle decorated definitions (@app.get, @dataclass, etc.)
            if child.type == "decorated_definition":
                definition = child.child_by_field_name("definition")
                if definition and definition.type in ("function_definition", "class_definition"):
                    name_node = definition.child_by_field_name("name")
                    name = name_node.text.decode("utf-8", errors="replace") if name_node else "unknown"
                    sym_type = "class" if definition.type == "class_definition" else ("method" if parent_name else "function")
                    
                    symbol = ParsedSymbol(
                        name=name,
                        type=sym_type,
                        start_line=child.start_point.row + 1,  # 1-indexed, starts from decorator
                        end_line=child.end_point.row + 1,
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        parent_name=parent_name,
                    )
                    symbols.append(symbol)

                    # If it's a class, also extract its methods
                    if definition.type == "class_definition":
                        body = definition.child_by_field_name("body")
                        if body:
                            self._extract_python_symbols(body, code_bytes, symbols, parent_name=name)
                continue

            # Standard function definition
            if child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                name = name_node.text.decode("utf-8", errors="replace") if name_node else "unknown"
                sym_type = "method" if parent_name else "function"

                symbols.append(
                    ParsedSymbol(
                        name=name,
                        type=sym_type,
                        start_line=child.start_point.row + 1,
                        end_line=child.end_point.row + 1,
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        parent_name=parent_name,
                    )
                )
                continue

            # Standard class definition
            if child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                name = name_node.text.decode("utf-8", errors="replace") if name_node else "unknown"

                symbols.append(
                    ParsedSymbol(
                        name=name,
                        type="class",
                        start_line=child.start_point.row + 1,
                        end_line=child.end_point.row + 1,
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        parent_name=parent_name,
                    )
                )

                body = child.child_by_field_name("body")
                if body:
                    self._extract_python_symbols(body, code_bytes, symbols, parent_name=name)
                continue

    def _extract_js_ts_symbols(
        self,
        node: Node,
        code_bytes: bytes,
        symbols: List[ParsedSymbol],
        parent_name: Optional[str] = None,
    ):
        """Recursively traverse JavaScript / TypeScript AST nodes."""
        for child in node.children:
            curr_node = child

            # Handle export statements wrapping declarations: export function / export class
            if curr_node.type in ("export_statement", "export_default_declaration"):
                decl = curr_node.child_by_field_name("declaration")
                if decl:
                    curr_node = decl

            # 1. Function declaration
            if curr_node.type in ("function_declaration", "generator_function_declaration"):
                name_node = curr_node.child_by_field_name("name")
                name = name_node.text.decode("utf-8", errors="replace") if name_node else "anonymous"
                symbols.append(
                    ParsedSymbol(
                        name=name,
                        type="function",
                        start_line=child.start_point.row + 1,
                        end_line=child.end_point.row + 1,
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        parent_name=parent_name,
                    )
                )
                continue

            # 2. Class declaration
            if curr_node.type == "class_declaration":
                name_node = curr_node.child_by_field_name("name")
                name = name_node.text.decode("utf-8", errors="replace") if name_node else "anonymous"
                symbols.append(
                    ParsedSymbol(
                        name=name,
                        type="class",
                        start_line=child.start_point.row + 1,
                        end_line=child.end_point.row + 1,
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        parent_name=parent_name,
                    )
                )
                body = curr_node.child_by_field_name("body")
                if body:
                    self._extract_js_ts_symbols(body, code_bytes, symbols, parent_name=name)
                continue

            # 3. Class method definition
            if curr_node.type == "method_definition":
                name_node = curr_node.child_by_field_name("name")
                name = name_node.text.decode("utf-8", errors="replace") if name_node else "anonymous"
                symbols.append(
                    ParsedSymbol(
                        name=name,
                        type="method",
                        start_line=child.start_point.row + 1,
                        end_line=child.end_point.row + 1,
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        parent_name=parent_name,
                    )
                )
                continue

            # 4. Lexical variable arrow functions: const foo = () => {}
            if curr_node.type in ("lexical_declaration", "variable_declaration"):
                for var_decl in curr_node.children:
                    if var_decl.type == "variable_declarator":
                        val_node = var_decl.child_by_field_name("value")
                        if val_node and val_node.type in ("arrow_function", "function_expression"):
                            name_node = var_decl.child_by_field_name("name")
                            name = name_node.text.decode("utf-8", errors="replace") if name_node else "anonymous"
                            symbols.append(
                                ParsedSymbol(
                                    name=name,
                                    type="function",
                                    start_line=child.start_point.row + 1,
                                    end_line=child.end_point.row + 1,
                                    start_byte=child.start_byte,
                                    end_byte=child.end_byte,
                                    parent_name=parent_name,
                                )
                            )
                continue

    def _parse_markdown(self, code: str) -> List[ParsedSymbol]:
        """Split Markdown documents into structured sections based on ATX headers (#, ##, ###)."""
        lines = code.splitlines(keepends=True)
        sections: List[ParsedSymbol] = []
        header_regex = re.compile(r"^(#{1,6})\s+(.+)$")

        current_header: Optional[str] = None
        start_line = 1
        byte_offset = 0
        section_start_byte = 0

        line_offsets = []
        for line in lines:
            line_offsets.append(byte_offset)
            byte_offset += len(line.encode("utf-8"))
        line_offsets.append(byte_offset)

        for i, line in enumerate(lines):
            match = header_regex.match(line.strip())
            line_num = i + 1
            if match:
                if current_header is not None:
                    sections.append(
                        ParsedSymbol(
                            name=current_header,
                            type="section",
                            start_line=start_line,
                            end_line=line_num - 1,
                            start_byte=section_start_byte,
                            end_byte=line_offsets[i],
                        )
                    )
                current_header = match.group(2).strip()
                start_line = line_num
                section_start_byte = line_offsets[i]

        if current_header is not None and start_line <= len(lines):
            sections.append(
                ParsedSymbol(
                    name=current_header,
                    type="section",
                    start_line=start_line,
                    end_line=len(lines),
                    start_byte=section_start_byte,
                    end_byte=byte_offset,
                )
            )

        return sections

    def _parse_sql(self, code: str) -> List[ParsedSymbol]:
        """Split SQL files by DDL / major query statements."""
        lines = code.splitlines(keepends=True)
        symbols: List[ParsedSymbol] = []
        stmt_regex = re.compile(
            r"^\s*(CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|PROCEDURE|FUNCTION|INDEX)|ALTER\s+TABLE|INSERT\s+INTO|SELECT)\s+([`\"\[]?\w+[`\"\]]?)",
            re.IGNORECASE,
        )

        current_name: Optional[str] = None
        current_type: str = "query"
        start_line = 1

        for i, line in enumerate(lines):
            match = stmt_regex.match(line)
            line_num = i + 1
            if match:
                if current_name is not None and line_num > start_line:
                    symbols.append(
                        ParsedSymbol(
                            name=current_name,
                            type=current_type,
                            start_line=start_line,
                            end_line=line_num - 1,
                            start_byte=0,
                            end_byte=0,
                        )
                    )
                cmd = match.group(1).upper()
                name = match.group(2).strip('`"[]')
                current_type = "table_definition" if "TABLE" in cmd else ("procedure" if "PROCEDURE" in cmd or "FUNCTION" in cmd else "query")
                current_name = f"{cmd} {name}"
                start_line = line_num

        if current_name is not None:
            symbols.append(
                ParsedSymbol(
                    name=current_name,
                    type=current_type,
                    start_line=start_line,
                    end_line=len(lines),
                    start_byte=0,
                    end_byte=0,
                )
            )

        return symbols

    def _parse_fallback(self, code: str, language: str) -> List[ParsedSymbol]:
        """
        Regex-based fallback parser for languages where Tree-sitter grammar is not
        installed (e.g. Java, C, C++). Matches standard function and class signatures.
        """
        lines = code.splitlines()
        symbols: List[ParsedSymbol] = []

        class_pattern = re.compile(r"^\s*(?:public|private|protected|static|final|abstract)?\s*class\s+([A-Za-z0-9_]+)", re.MULTILINE)
        func_pattern = re.compile(
            r"^\s*(?:public|private|protected|static|async|virtual|inline)?\s*[\w<>\[\], ]+\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{?",
            re.MULTILINE,
        )

        for i, line in enumerate(lines):
            line_num = i + 1
            cls_match = class_pattern.match(line)
            if cls_match:
                symbols.append(
                    ParsedSymbol(
                        name=cls_match.group(1),
                        type="class",
                        start_line=line_num,
                        end_line=line_num,  # Will be adjusted by chunker
                        start_byte=0,
                        end_byte=0,
                    )
                )
                continue

            fn_match = func_pattern.match(line)
            if fn_match and not any(kw in line for kw in ("if ", "while ", "for ", "switch ", "catch ")):
                symbols.append(
                    ParsedSymbol(
                        name=fn_match.group(1),
                        type="function",
                        start_line=line_num,
                        end_line=line_num,
                        start_byte=0,
                        end_byte=0,
                    )
                )

        return symbols
