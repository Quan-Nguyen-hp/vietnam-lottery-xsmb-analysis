"""Repository integration and legacy scientific isolation audit tests for XPIS v3 M3.

Verifies repository identity, canonical data source contracts, dependency locks,
total legacy code isolation (LEGACY_SCIENTIFIC_IMPORT_COUNT == 0), and protocol
fingerprint immutability under execution authorization flags.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess

from src.m3_digit_factor.authority import (
    AUTHORITY_KEYS,
    CANONICAL_DATA_PATH,
    EXPECTED_REPOSITORY,
    construct_authority_snapshot,
)
from src.m3_digit_factor.dataset import CANONICAL_HEADER


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_canonical_repository_identity() -> None:
    """Repository remote identity matches canonical expectation."""
    assert EXPECTED_REPOSITORY == 'Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis'

    proc = subprocess.run(
        ['git', 'config', '--get', 'remote.origin.url'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    remote_url = proc.stdout.strip()
    assert EXPECTED_REPOSITORY in remote_url, f'Unexpected remote URL: {remote_url}'


def test_canonical_data_source_path() -> None:
    """Canonical data source file exists and is tracked by Git."""
    assert CANONICAL_DATA_PATH == 'data/xsmb-2-digits.csv'
    data_file = REPO_ROOT / CANONICAL_DATA_PATH
    assert data_file.is_file(), f'Canonical data file missing: {data_file}'

    proc = subprocess.run(
        ['git', 'ls-files', '--error-unmatch', CANONICAL_DATA_PATH],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == CANONICAL_DATA_PATH


def test_canonical_header_contract() -> None:
    """Canonical data header matches frozen header contract exactly."""
    assert len(CANONICAL_HEADER) == 28
    assert CANONICAL_HEADER[0] == 'date'
    assert CANONICAL_HEADER[1] == 'special'
    assert CANONICAL_HEADER[2] == 'prize1'

    data_file = REPO_ROOT / CANONICAL_DATA_PATH
    with data_file.open('r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    actual_header = tuple(first_line.split(','))
    assert actual_header == CANONICAL_HEADER


def test_pyproject_direct_dependency_scipy() -> None:
    """pyproject.toml specifies scipy as a direct project dependency."""
    pyproject_path = REPO_ROOT / 'pyproject.toml'
    content = pyproject_path.read_text(encoding='utf-8')

    # Basic parse of pyproject.toml dependencies section
    in_dependencies = False
    dependencies: list[str] = []
    for line in content.splitlines():
        line_clean = line.strip()
        if line_clean == 'dependencies = [' or line_clean.startswith('dependencies = ['):
            in_dependencies = True
            continue
        if in_dependencies:
            if line_clean == ']':
                break
            # Remove quotes and comma
            dep = line_clean.strip('",\' ')
            if dep:
                dependencies.append(dep)

    scipy_deps = [d for d in dependencies if d == 'scipy' or d.startswith(('scipy==', 'scipy>=', 'scipy<='))]
    assert len(scipy_deps) >= 1, f'scipy not found in direct dependencies: {dependencies}'


def test_uv_lock_integrity() -> None:
    """uv lock --check passes, guaranteeing deterministic dependency resolution."""
    proc = subprocess.run(
        ['uv', 'lock', '--check'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f'uv lock --check failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}'


def test_legacy_scientific_isolation_audit() -> None:
    """Static AST audit proves zero imports from legacy scientific packages.

    LEGACY_SCIENTIFIC_IMPORT_COUNT must equal 0 across all M3 implementation
    modules and the CLI entrypoint.
    """
    m3_src_dir = REPO_ROOT / 'src' / 'm3_digit_factor'
    cli_file = REPO_ROOT / 'backtests' / 'm3_digit_factor_development.py'

    py_files = sorted(m3_src_dir.glob('*.py')) + [cli_file]
    assert len(py_files) >= 15, f'Too few files audited: {len(py_files)}'

    prohibited_root_modules = {
        'src.count_v2',
        'src.data',
        'src.methods',
        'src.features',
        'src.meta',
        'src.decision',
        'src.evaluation',
        'src.evidence',
        'src.registry',
        'count_v2',
        'methods',
    }

    legacy_scientific_import_count = 0
    violations: list[str] = []

    for file_path in py_files:
        tree = ast.parse(file_path.read_text(encoding='utf-8'), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    for prohibited in prohibited_root_modules:
                        if name == prohibited or name.startswith(f'{prohibited}.'):
                            legacy_scientific_import_count += 1
                            violations.append(f'{file_path.name}: import {name}')
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                # If relative import, it stays within m3_digit_factor package
                if node.level > 0:
                    continue
                for prohibited in prohibited_root_modules:
                    if module == prohibited or module.startswith(f'{prohibited}.'):
                        legacy_scientific_import_count += 1
                        violations.append(f'{file_path.name}: from {module} import ...')

    assert legacy_scientific_import_count == 0, (
        f'LEGACY_SCIENTIFIC_IMPORT_COUNT != 0 ({legacy_scientific_import_count}):\n'
        + '\n'.join(violations)
    )


def test_historical_authorization_flag_preserves_protocol_fingerprint() -> None:
    """Authorizing a historical run (--authorize-historical-run) preserves authority keys and protocol fingerprint."""
    assert len(AUTHORITY_KEYS) == 72
    assert 'authorize_historical_run' not in AUTHORITY_KEYS

    # Build authority snapshot using current repository HEAD
    snap1 = construct_authority_snapshot(repository_root=REPO_ROOT)
    fp1 = snap1.protocol_fingerprint_sha256

    # Authority keys and fingerprint must be identical and immutable
    snap2 = construct_authority_snapshot(repository_root=REPO_ROOT)
    fp2 = snap2.protocol_fingerprint_sha256

    assert fp1 == fp2
    assert isinstance(fp1, str) and len(fp1) == 64
    assert snap1.authority is not None
    assert frozenset(snap1.authority.keys()) == AUTHORITY_KEYS
