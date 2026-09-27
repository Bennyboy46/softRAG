import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
import git
from git.exc import GitCommandError

from backend.config import settings

logger = logging.getLogger(__name__)


def get_repo_version_details(repo_dir: Path, repo_url: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Collect git version info for repository tracking and stale-result prevention."""
    try:
        repo = git.Repo(str(repo_dir))
        try:
            head = repo.head.commit.hexsha
        except Exception:
            head = None
        try:
            branch = repo.active_branch.name
        except Exception:
            branch = None
        return repo_url or str(repo.remotes.origin.url) if repo.remotes else repo_url, head, branch
    except Exception:
        return repo_url, None, None


def generate_repo_id(repo_url: str) -> str:
    """
    Generate a deterministic, filesystem-safe repository identifier from its URL.
    Example: https://github.com/fastapi/fastapi -> fastapi_fastapi_a1b2c3d4
    """
    cleaned = repo_url.strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    # Extract owner/repo or path components
    match = re.search(r"[:/]([^/:]+)/([^/:]+?)$", cleaned)
    if match:
        owner, repo = match.group(1), match.group(2)
        base_slug = f"{owner}_{repo}".lower()
    else:
        base_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", cleaned).strip("_")

    # Append short 8-char hash for uniqueness against colliding names
    url_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
    safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", base_slug)
    return f"{safe_slug}_{url_hash}"


def clone_repository(repo_url: str, custom_target_dir: Optional[Path] = None) -> Tuple[str, Path]:
    """
    Clone a GitHub repository into the local storage directory.
    Uses depth=1 (shallow clone) to optimize speed and disk usage.
    
    Returns:
        (repo_id, local_repo_path)
    """
    repo_id = generate_repo_id(repo_url)
    target_dir = custom_target_dir or (settings.repositories_path / repo_id)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        if (target_dir / ".git").exists():
            logger.info(f"Repository already cloned at {target_dir}. Reusing existing clone.")
            return repo_id, target_dir
        else:
            logger.warning(f"Target directory {target_dir} exists but is not a git repo. Re-cloning.")

    logger.info(f"Cloning {repo_url} into {target_dir} (shallow clone depth=1)...")
    try:
        git.Repo.clone_from(
            url=repo_url,
            to_path=str(target_dir),
            depth=1,
            single_branch=True,
        )
        logger.info(f"Successfully cloned repository: {repo_id}")
        return repo_id, target_dir
    except GitCommandError as exc:
        logger.error(f"Git clone failed for {repo_url}: {exc}")
        if "Authentication failed" in str(exc) or "could not read Username" in str(exc):
            raise ValueError(f"Repository is private or requires authentication: {repo_url}") from exc
        elif "not found" in str(exc).lower():
            raise ValueError(f"Repository not found at URL: {repo_url}") from exc
        else:
            raise RuntimeError(f"Failed to clone repository: {exc.stderr}") from exc


def is_binary_file(file_path: Path) -> bool:
    """Detect if a file contains binary content by inspecting its initial byte chunk."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return True
        return False
    except Exception:
        return True


def filter_repository_files(repo_path: Path) -> List[Path]:
    """
    Scan the repository directory and return eligible source code files.
    Applies strict exclusions for:
    - Version control and cache folders (.git, __pycache__, node_modules)
    - Build artifacts, virtual environments, binaries
    - Excessively large files (> max_file_size_bytes)
    - Non-whitelisted file extensions
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    accepted_files: List[Path] = []
    ignored_dirs = settings.ignored_directories
    ignored_names = settings.ignored_file_names
    ignored_exts = settings.ignored_extensions
    allowed_exts = set(settings.extension_to_language.keys())

    for item in repo_path.rglob("*"):
        # Skip directories
        if item.is_dir():
            continue

        # Check if any parent part matches an ignored directory
        parts = set(item.relative_to(repo_path).parts[:-1])
        if any(ignored_part in parts for ignored_part in ignored_dirs):
            continue

        file_name = item.name.lower()
        if file_name in ignored_names:
            continue

        suffix = item.suffix.lower()
        if suffix in ignored_exts:
            continue

        if suffix not in allowed_exts:
            continue

        # Check file size limit
        try:
            size = item.stat().st_size
            if size == 0 or size > settings.max_file_size_bytes:
                continue
        except OSError:
            continue

        # Check for binary content
        if is_binary_file(item):
            continue

        accepted_files.append(item)

    accepted_files.sort()
    logger.info(f"Scanned repository: found {len(accepted_files)} eligible source files.")
    return accepted_files
