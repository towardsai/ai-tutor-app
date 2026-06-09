from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KB_DIR = Path(os.getenv("AI_TUTOR_KB_DIR", "data/kb"))
DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_MAX_OUTPUT_CHARS = 40_000
SUPPORTED_COMMANDS = frozenset({"rg", "grep", "find", "ls", "sed", "head", "cat", "wc"})
SHELL_TOKENS = frozenset({"|", "||", "&", "&&", ";", ">", ">>", "<", "<<", "`"})
RG_VALUE_OPTIONS = frozenset(
    {
        "-g",
        "--glob",
        "-t",
        "--type",
        "-T",
        "--type-not",
        "-m",
        "--max-count",
    }
)
RG_FLAG_OPTIONS = frozenset(
    {
        "-n",
        "--line-number",
        "-i",
        "--ignore-case",
        "-S",
        "--smart-case",
        "-F",
        "--fixed-strings",
        "-w",
        "--word-regexp",
        "--hidden",
        "--no-ignore",
    }
)
GREP_FLAG_OPTIONS = frozenset(
    {"-i", "--ignore-case", "-w", "--word-regexp", "-F", "-E"}
)
LS_FLAG_OPTIONS = frozenset({"-1", "-a", "-l", "-la", "-al"})
WC_FLAG_OPTIONS = frozenset({"-l", "-w", "-c", "-m"})


@dataclass(frozen=True, slots=True)
class KbCommandResult:
    command: str
    argv: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


class KbCommandError(ValueError):
    """Raised when a KB command violates the read-only command policy."""


def _resolve_root(root: Path | None) -> Path:
    resolved = (root or DEFAULT_KB_DIR).resolve()
    if not resolved.exists():
        raise KbCommandError(f"KB root does not exist: {resolved}")
    if not resolved.is_dir():
        raise KbCommandError(f"KB root is not a directory: {resolved}")
    return resolved


def _resolve_inside_root(value: str, root: Path, *, must_exist: bool = True) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root / value).resolve()
    if resolved != root and root not in resolved.parents:
        raise KbCommandError("KB command paths must stay inside data/kb")
    if must_exist and not resolved.exists():
        raise KbCommandError(f"KB command path does not exist: {value}")
    return resolved


def _relative_path_arg(value: str, root: Path) -> str:
    resolved = _resolve_inside_root(value, root)
    return "." if resolved == root else resolved.relative_to(root).as_posix()


def _reject_shell_syntax(command: str, tokens: list[str]) -> None:
    if "\n" in command or "\r" in command:
        raise KbCommandError("KB command must be a single command, not a script")
    if "$(" in command or "${" in command:
        raise KbCommandError("Shell expansion is not allowed in KB commands")
    if any(token in SHELL_TOKENS for token in tokens):
        raise KbCommandError("Pipes, redirects, and command chaining are not allowed")


def _numeric_value(token: str, option: str) -> str:
    if not token.isdigit():
        raise KbCommandError(f"{option} expects a positive integer")
    return token


def _safe_pattern_value(token: str, option: str) -> str:
    if "\x00" in token:
        raise KbCommandError(f"{option} contains an invalid value")
    if Path(token).is_absolute() or ".." in Path(token).parts:
        raise KbCommandError(f"{option} values must stay inside data/kb")
    return token


def _is_broad_raw_path(path: str) -> bool:
    normalized = path.rstrip("/")
    return normalized in {".", "raw", "raw/courses", "raw/docs"}


def _reject_unbounded_raw_search(
    command: str, paths: list[str], has_max_count: bool
) -> None:
    if has_max_count:
        return
    if any(_is_broad_raw_path(path) for path in paths):
        raise KbCommandError(
            f"`{command}` over broad raw paths requires -m/--max-count "
            "or a narrower source/file path"
        )


def _build_rg(tokens: list[str], root: Path) -> list[str]:
    if shutil.which("rg") is None:
        raise KbCommandError("`rg` is not installed in this runtime")
    options: list[str] = []
    positionals: list[str] = []
    has_max_count = False
    idx = 1
    parsing_options = True
    while idx < len(tokens):
        token = tokens[idx]
        if parsing_options and token == "--":
            parsing_options = False
            idx += 1
            continue
        if parsing_options and token in RG_FLAG_OPTIONS:
            options.append(token)
            idx += 1
            continue
        if parsing_options and token in RG_VALUE_OPTIONS:
            if idx + 1 >= len(tokens):
                raise KbCommandError(f"{token} requires a value")
            value = tokens[idx + 1]
            if token in {"-m", "--max-count"}:
                value = _numeric_value(value, token)
                has_max_count = True
            else:
                value = _safe_pattern_value(value, token)
            options.extend([token, value])
            idx += 2
            continue
        positionals.append(token)
        parsing_options = False
        idx += 1

    if not positionals:
        raise KbCommandError("`rg` requires a search pattern")
    pattern = positionals[0]
    paths = [_relative_path_arg(value, root) for value in positionals[1:]] or ["."]
    _reject_unbounded_raw_search("rg", paths, has_max_count)
    return [
        "rg",
        "--color=never",
        "--line-number",
        "--no-heading",
        *options,
        pattern,
        *paths,
    ]


def _build_grep(tokens: list[str], root: Path) -> list[str]:
    options: list[str] = []
    positionals: list[str] = []
    idx = 1
    parsing_options = True
    while idx < len(tokens):
        token = tokens[idx]
        if parsing_options and token == "--":
            parsing_options = False
            idx += 1
            continue
        if parsing_options and token in GREP_FLAG_OPTIONS:
            options.append(token)
            idx += 1
            continue
        positionals.append(token)
        parsing_options = False
        idx += 1
    if not positionals:
        raise KbCommandError("`grep` requires a search pattern")
    pattern = positionals[0]
    paths = [_relative_path_arg(value, root) for value in positionals[1:]] or ["."]
    return ["grep", "-R", "-n", "--color=never", *options, pattern, *paths]


def _build_find(tokens: list[str], root: Path) -> list[str]:
    path = "."
    idx = 1
    if idx < len(tokens) and not tokens[idx].startswith("-"):
        path = _relative_path_arg(tokens[idx], root)
        idx += 1

    argv = ["find", path]
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"-maxdepth", "-mindepth"}:
            if idx + 1 >= len(tokens):
                raise KbCommandError(f"{token} requires a value")
            argv.extend([token, _numeric_value(tokens[idx + 1], token)])
            idx += 2
            continue
        if token == "-type":
            if idx + 1 >= len(tokens) or tokens[idx + 1] not in {"f", "d"}:
                raise KbCommandError("-type supports only `f` or `d`")
            argv.extend([token, tokens[idx + 1]])
            idx += 2
            continue
        if token in {"-name", "-iname"}:
            if idx + 1 >= len(tokens):
                raise KbCommandError(f"{token} requires a value")
            argv.extend([token, _safe_pattern_value(tokens[idx + 1], token)])
            idx += 2
            continue
        raise KbCommandError(
            "`find` supports only path, -maxdepth, -mindepth, -type, -name, and -iname"
        )
    return argv


def _build_ls(tokens: list[str], root: Path) -> list[str]:
    argv = ["ls"]
    paths: list[str] = []
    for token in tokens[1:]:
        if token in LS_FLAG_OPTIONS:
            argv.append(token)
            continue
        paths.append(_relative_path_arg(token, root))
    return [*argv, *(paths or ["."])]


def _build_sed(tokens: list[str], root: Path) -> list[str]:
    if len(tokens) != 4 or tokens[1] != "-n":
        raise KbCommandError("`sed` supports only: sed -n START,ENDp FILE")
    expression = tokens[2]
    if not re.fullmatch(r"\d+,\d+p", expression):
        raise KbCommandError("`sed` supports only START,ENDp read expressions")
    path = _relative_path_arg(tokens[3], root)
    return ["sed", "-n", expression, path]


def _build_head(tokens: list[str], root: Path) -> list[str]:
    argv = ["head"]
    paths: list[str] = []
    idx = 1
    if idx < len(tokens) and tokens[idx] == "-n":
        if idx + 1 >= len(tokens):
            raise KbCommandError("-n requires a value")
        argv.extend(["-n", _numeric_value(tokens[idx + 1], "-n")])
        idx += 2
    while idx < len(tokens):
        paths.append(_relative_path_arg(tokens[idx], root))
        idx += 1
    if not paths:
        raise KbCommandError("`head` requires at least one file path")
    return [*argv, *paths]


def _build_cat(tokens: list[str], root: Path) -> list[str]:
    if len(tokens) < 2:
        raise KbCommandError("`cat` requires at least one file path")
    return ["cat", *[_relative_path_arg(token, root) for token in tokens[1:]]]


def _build_wc(tokens: list[str], root: Path) -> list[str]:
    argv = ["wc"]
    paths: list[str] = []
    for token in tokens[1:]:
        if token in WC_FLAG_OPTIONS:
            argv.append(token)
            continue
        paths.append(_relative_path_arg(token, root))
    if not paths:
        raise KbCommandError("`wc` requires at least one file path")
    return [*argv, *paths]


def build_kb_command_argv(
    command: str, *, root: Path | None = None
) -> tuple[list[str], Path]:
    resolved_root = _resolve_root(root)
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise KbCommandError(f"Invalid KB command syntax: {exc}") from exc
    if not tokens:
        raise KbCommandError("KB command cannot be empty")
    _reject_shell_syntax(command, tokens)

    executable = tokens[0]
    if executable not in SUPPORTED_COMMANDS:
        supported = ", ".join(sorted(SUPPORTED_COMMANDS))
        raise KbCommandError(
            f"Unsupported KB command `{executable}`. Supported: {supported}"
        )

    builders = {
        "rg": _build_rg,
        "grep": _build_grep,
        "find": _build_find,
        "ls": _build_ls,
        "sed": _build_sed,
        "head": _build_head,
        "cat": _build_cat,
        "wc": _build_wc,
    }
    return builders[executable](tokens, resolved_root), resolved_root


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    marker = f"\n... output truncated to {max_chars} characters ..."
    return text[: max(0, max_chars - len(marker))] + marker, True


def run_kb_command(
    command: str,
    *,
    root: Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> KbCommandResult:
    argv, resolved_root = build_kb_command_argv(command, root=root)
    timeout = max(1, min(int(timeout_seconds), 30))
    output_limit = max(1_000, min(int(max_output_chars), 80_000))
    env = {
        "HOME": os.getenv("HOME", ""),
        "LANG": os.getenv("LANG", "C.UTF-8"),
        "LC_ALL": os.getenv("LC_ALL", "C.UTF-8"),
        "PATH": os.getenv("PATH", ""),
    }
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=resolved_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        exit_code = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = (
            stderr + "\n" if stderr else ""
        ) + f"Command timed out after {timeout}s."

    stdout, stdout_truncated = _truncate(stdout, output_limit)
    stderr, stderr_truncated = _truncate(stderr, min(output_limit, 8_000))
    return KbCommandResult(
        command=command,
        argv=argv,
        cwd=resolved_root.as_posix(),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=stdout_truncated or stderr_truncated,
    )


def format_command_payload(result: KbCommandResult) -> str:
    lines = [
        f"$ {result.command}",
        f"cwd: {result.cwd}",
        f"exit_code: {result.exit_code}",
    ]
    if result.timed_out:
        lines.append("timed_out: true")
    if result.truncated:
        lines.append("truncated: true")
    if result.stdout:
        lines.extend(["stdout:", result.stdout.rstrip()])
    if result.stderr:
        lines.extend(["stderr:", result.stderr.rstrip()])
    return "\n".join(lines)
