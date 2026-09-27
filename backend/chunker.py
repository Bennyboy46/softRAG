import logging
from pathlib import Path
from typing import List, Optional

from backend.models import ChunkMetadata, CodeChunk, ParsedSymbol
from backend.parser import CodeParser

logger = logging.getLogger(__name__)


class CodeAwareChunker:
    """
    Code-aware chunker that divides source code files into semantic units
    (functions, classes, methods, imports, and sections) rather than arbitrary
    character/token splits, preserving exact file provenance and line numbers.
    """

    def __init__(
        self,
        parser: Optional[CodeParser] = None,
        max_chunk_lines: int = 100,
        subchunk_overlap_lines: int = 10,
    ):
        self.parser = parser or CodeParser()
        self.max_chunk_lines = max_chunk_lines
        self.subchunk_overlap_lines = subchunk_overlap_lines

    def chunk_file(
        self,
        repo_id: str,
        relative_path: str,
        code: str,
        language: str,
    ) -> List[CodeChunk]:
        """
        Produce a list of semantic CodeChunks from a single file's source code.
        """
        if not code.strip():
            return []

        lines = code.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return []

        # Parse AST symbols
        symbols = self.parser.parse(code, language)

        # If no symbols found (e.g. flat script, config, or unparsed language)
        if not symbols:
            return self._chunk_flat_file(repo_id, relative_path, lines, language)

        chunks: List[CodeChunk] = []

        # Sort symbols by start_line
        symbols = sorted(symbols, key=lambda s: (s.start_line, s.end_line))

        # 1. Module header / imports (lines 1 to first symbol start)
        first_symbol_start = symbols[0].start_line
        if first_symbol_start > 1:
            header_lines = lines[: first_symbol_start - 1]
            header_text = "\n".join(header_lines).strip()
            if header_text and len(header_text) > 10:
                chunks.append(
                    self._create_chunk(
                        repo_id=repo_id,
                        file_path=relative_path,
                        language=language,
                        chunk_type="module",
                        name="module_header",
                        start_line=1,
                        end_line=first_symbol_start - 1,
                        code_lines=header_lines,
                    )
                )

        # 2. Iterate through parsed symbols
        last_end_line = first_symbol_start - 1

        for i, sym in enumerate(symbols):
            # Guard against invalid or out-of-bound line numbers
            start = max(1, min(sym.start_line, total_lines))
            end = max(start, min(sym.end_line, total_lines))

            # If this is a method whose parent class is also in symbols,
            # or if symbols overlap, handle appropriately
            sym_lines = lines[start - 1 : end]

            # Oversized symbol splitting
            if len(sym_lines) > self.max_chunk_lines:
                subchunks = self._split_oversized_symbol(
                    repo_id=repo_id,
                    file_path=relative_path,
                    language=language,
                    symbol=sym,
                    symbol_lines=sym_lines,
                )
                chunks.extend(subchunks)
            else:
                chunks.append(
                    self._create_chunk(
                        repo_id=repo_id,
                        file_path=relative_path,
                        language=language,
                        chunk_type=sym.type,
                        name=sym.name,
                        start_line=start,
                        end_line=end,
                        code_lines=sym_lines,
                    )
                )

            last_end_line = max(last_end_line, end)

        # 3. Trailing module code after the last symbol
        if last_end_line < total_lines:
            trailing_lines = lines[last_end_line:total_lines]
            trailing_text = "\n".join(trailing_lines).strip()
            if trailing_text and len(trailing_text) > 15:
                chunks.append(
                    self._create_chunk(
                        repo_id=repo_id,
                        file_path=relative_path,
                        language=language,
                        chunk_type="module",
                        name="module_footer",
                        start_line=last_end_line + 1,
                        end_line=total_lines,
                        code_lines=trailing_lines,
                    )
                )

        return chunks

    def _split_oversized_symbol(
        self,
        repo_id: str,
        file_path: str,
        language: str,
        symbol: ParsedSymbol,
        symbol_lines: List[str],
    ) -> List[CodeChunk]:
        """
        Split a very large function, class, or section into smaller logical chunks
        while prepending a context header so that symbol identification is not lost.
        """
        chunks: List[CodeChunk] = []
        total_sym_lines = len(symbol_lines)
        step = self.max_chunk_lines - self.subchunk_overlap_lines
        part = 1

        for idx in range(0, total_sym_lines, step):
            sub_lines = symbol_lines[idx : idx + self.max_chunk_lines]
            if not sub_lines:
                break

            sub_start = symbol.start_line + idx
            sub_end = sub_start + len(sub_lines) - 1

            # Context banner prepended to source code
            header_comment = (
                f"// Context: {file_path} | {symbol.type.capitalize()}: {symbol.name} "
                f"(Part {part}, lines {sub_start}-{sub_end})\n"
            )
            content = header_comment + "\n".join(sub_lines)

            chunk_id = f"{repo_id}:{file_path}:{symbol.type}:{symbol.name or 'anon'}_part{part}:{sub_start}_{sub_end}"
            metadata = ChunkMetadata(
                repository=repo_id,
                file=file_path,
                language=language,
                type=symbol.type,
                name=f"{symbol.name} (part {part})" if symbol.name else None,
                start_line=sub_start,
                end_line=sub_end,
            )
            chunks.append(CodeChunk(chunk_id=chunk_id, content=content, metadata=metadata))
            part += 1

            if idx + self.max_chunk_lines >= total_sym_lines:
                break

        return chunks

    def _chunk_flat_file(
        self,
        repo_id: str,
        file_path: str,
        lines: List[str],
        language: str,
    ) -> List[CodeChunk]:
        """
        Fallback chunking strategy for files without detected AST symbols.
        Uses sliding windows with line overlap and exact 1-indexed line citations.
        """
        chunks: List[CodeChunk] = []
        total_lines = len(lines)
        step = self.max_chunk_lines - self.subchunk_overlap_lines

        part = 1
        for idx in range(0, total_lines, step):
            sub_lines = lines[idx : idx + self.max_chunk_lines]
            if not sub_lines:
                break

            sub_start = idx + 1
            sub_end = sub_start + len(sub_lines) - 1

            header_comment = f"// File: {file_path} (Lines {sub_start}-{sub_end})\n"
            content = header_comment + "\n".join(sub_lines)

            chunk_id = f"{repo_id}:{file_path}:block:{part}:{sub_start}_{sub_end}"
            metadata = ChunkMetadata(
                repository=repo_id,
                file=file_path,
                language=language,
                type="block",
                name=None,
                start_line=sub_start,
                end_line=sub_end,
            )
            chunks.append(CodeChunk(chunk_id=chunk_id, content=content, metadata=metadata))
            part += 1

            if idx + self.max_chunk_lines >= total_lines:
                break

        return chunks

    def _create_chunk(
        self,
        repo_id: str,
        file_path: str,
        language: str,
        chunk_type: str,
        name: Optional[str],
        start_line: int,
        end_line: int,
        code_lines: List[str],
    ) -> CodeChunk:
        """Helper to create a normalized CodeChunk instance."""
        content = "\n".join(code_lines)
        clean_name = name or "anon"
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in clean_name)
        chunk_id = f"{repo_id}:{file_path}:{chunk_type}:{safe_name}:{start_line}_{end_line}"

        metadata = ChunkMetadata(
            repository=repo_id,
            file=file_path,
            language=language,
            type=chunk_type,
            name=name,
            start_line=start_line,
            end_line=end_line,
        )

        return CodeChunk(chunk_id=chunk_id, content=content, metadata=metadata)
