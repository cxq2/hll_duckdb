#!/usr/bin/env python3
"""Helpers for Role C benchmark scripts.

This module uses two DuckDB runtimes for different jobs:
1. The Python `duckdb` package is only used to generate/reuse TPC-H data.
2. The locally built DuckDB CLI is used for exact/HLL benchmarking so the
   runtime matches the locally built `quack` extension.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import duckdb


DEFAULT_REQUIRED_TPCH_TABLES = (
    "lineitem",
    "orders",
    "partsupp",
    "customer",
    "supplier",
    "part",
    "nation",
    "region",
)

_TIMER_RE = re.compile(r"^Run Time \(s\): real ([0-9.]+) ")


def detect_default_duckdb_binary(repo_root: Path) -> Path:
    executable_names = ["duckdb.exe", "duckdb"] if os.name == "nt" else ["duckdb", "duckdb.exe"]
    search_roots = [
        repo_root / "build" / "release",
        repo_root / "my_hll_extension" / "build" / "release",
    ]

    candidates: list[Path] = []
    for search_root in search_roots:
        for executable_name in executable_names:
            candidate = search_root / executable_name
            candidates.append(candidate)
            if candidate.exists():
                return candidate

    return candidates[0]


def split_target(target: str) -> tuple[str, str]:
    if "." not in target:
        raise ValueError(f"Invalid target '{target}', expected table.column")
    table_name, column_name = target.split(".", 1)
    if not table_name or not column_name:
        raise ValueError(f"Invalid target '{target}', expected table.column")
    return table_name, column_name


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_name = ?
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0] > 0)


def ensure_tpch_data(
    database: Path,
    sf: int,
    force_dbgen: bool,
    required_tables: Sequence[str] = DEFAULT_REQUIRED_TPCH_TABLES,
) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database))
    try:
        has_all = all(table_exists(con, table_name) for table_name in required_tables)
        if has_all and not force_dbgen:
            print(f"[info] Reusing existing TPC-H tables in {database.name}.")
            return

        print(f"[info] Generating TPC-H tables with DuckDB Python client: CALL dbgen(sf={sf}) ...")
        con.execute("INSTALL tpch")
        con.execute("LOAD tpch")

        if force_dbgen:
            for table_name in required_tables:
                con.execute(f"DROP TABLE IF EXISTS {table_name}")

        start = time.perf_counter()
        con.execute(f"CALL dbgen(sf={sf})")
        elapsed = time.perf_counter() - start
        print(f"[info] dbgen completed in {elapsed:.3f}s for sf={sf}")
    finally:
        con.close()


def _normalize_statement(statement: str) -> str:
    stripped = statement.strip()
    if not stripped:
        return ""
    if stripped.startswith("."):
        return stripped
    if stripped.endswith(";"):
        return stripped
    return stripped + ";"


@dataclass
class ScalarQueryRun:
    values: list[str]
    times_s: list[float]
    stdout: str
    stderr: str


class DuckDBCliRunner:
    def __init__(self, duckdb_binary: Path, database: Path, *, unsigned: bool = True):
        self.duckdb_binary = duckdb_binary.resolve()
        self.database = database.resolve()
        self.unsigned = unsigned

        if not self.duckdb_binary.exists():
            raise FileNotFoundError(f"DuckDB CLI binary not found: {self.duckdb_binary}")
        if not self.database.exists():
            raise FileNotFoundError(f"DuckDB database not found: {self.database}")

    def run_script(self, statements: Sequence[str]) -> subprocess.CompletedProcess[str]:
        normalized = [_normalize_statement(statement) for statement in statements if statement.strip()]
        if not normalized:
            raise ValueError("No SQL statements provided")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sql",
            delete=False,
            encoding="ascii",
            newline="\n",
        ) as handle:
            handle.write("\n".join(normalized))
            handle.write("\n")
            script_path = Path(handle.name)

        cmd = [
            str(self.duckdb_binary),
            str(self.database),
            "-batch",
            "-csv",
            "-noheader",
        ]
        if self.unsigned:
            cmd.append("-unsigned")
        cmd.extend(["-f", str(script_path)])

        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        except subprocess.CalledProcessError as ex:
            raise RuntimeError(
                "DuckDB CLI execution failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{ex.stdout}\n"
                f"STDERR:\n{ex.stderr}"
            ) from ex
        finally:
            script_path.unlink(missing_ok=True)

    def run_scalar_query_repeated(
        self,
        sql: str,
        *,
        repeats: int,
        pre_statements: Sequence[str] | None = None,
    ) -> ScalarQueryRun:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")

        statements = list(pre_statements or [])
        statements.append(".timer on")
        statements.extend(sql for _ in range(repeats))

        completed = self.run_script(statements)
        values: list[str] = []
        times_s: list[float] = []

        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = _TIMER_RE.match(line)
            if match:
                times_s.append(float(match.group(1)))
                continue
            parsed = next(csv.reader([raw_line]))
            if not parsed:
                continue
            values.append(parsed[0])

        if len(values) != repeats:
            raise RuntimeError(
                f"Expected {repeats} scalar values but parsed {len(values)}.\nSTDOUT:\n{completed.stdout}"
            )
        if len(times_s) != repeats:
            raise RuntimeError(
                f"Expected {repeats} timing rows but parsed {len(times_s)}.\nSTDOUT:\n{completed.stdout}"
            )

        return ScalarQueryRun(values=values, times_s=times_s, stdout=completed.stdout, stderr=completed.stderr)

    def run_timed_statements(
        self,
        statements: Sequence[str],
        *,
        pre_statements: Sequence[str] | None = None,
    ) -> list[float]:
        sql_statements = [statement for statement in statements if statement.strip()]
        if not sql_statements:
            raise ValueError("No statements provided for timing")

        script = list(pre_statements or [])
        script.append(".timer on")
        script.extend(sql_statements)

        completed = self.run_script(script)
        times_s: list[float] = []
        for raw_line in completed.stdout.splitlines():
            match = _TIMER_RE.match(raw_line.strip())
            if match:
                times_s.append(float(match.group(1)))

        if len(times_s) != len(sql_statements):
            raise RuntimeError(
                f"Expected {len(sql_statements)} timing rows but parsed {len(times_s)}.\nSTDOUT:\n{completed.stdout}"
            )
        return times_s

    def query_scalar(self, sql: str, *, pre_statements: Sequence[str] | None = None) -> str:
        run = self.run_scalar_query_repeated(sql, repeats=1, pre_statements=pre_statements)
        return run.values[0]
