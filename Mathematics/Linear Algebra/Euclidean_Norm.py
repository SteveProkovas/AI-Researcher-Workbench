"""euclidean_norm.py

Compute the Euclidean (L2) norm of a vector.

Provides:
 - euclidean_norm(vector) -> float
 - CLI to compute the norm from command-line arguments or an interactive prompt

Examples
--------
>>> euclidean_norm([3, -4])
5.0
>>> euclidean_norm([])
0.0

Usage (CLI):
  python euclidean_norm.py 3 -4
  python euclidean_norm.py -s "3, -4, 0"
  python euclidean_norm.py     # will prompt for input

"""

from __future__ import annotations

import math
import argparse
import sys
import re
from typing import Iterable, List


def euclidean_norm(vector: Iterable[float]) -> float:
    """Return the Euclidean (L2) norm of the given vector.

    Parameters
    ----------
    vector : Iterable[float]
        Iterable of numeric values (int/float).

    Returns
    -------
    float
        The Euclidean norm: sqrt(sum(x_i^2)).

    Notes
    -----
    - Non-numeric values will raise ValueError.
    - An empty vector returns 0.0.

    Examples
    --------
    >>> euclidean_norm([3, -4])
    5.0
    """
    total = 0.0
    count = 0
    for x in vector:
        try:
            val = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"Non-numeric value encountered: {x!r}")
        total += val * val
        count += 1

    if count == 0:
        return 0.0
    return math.sqrt(total)


def parse_numbers(text: str) -> List[float]:
    """Parse a string of numbers separated by spaces, commas or both.

    Examples
    --------
    >>> parse_numbers("3, -4 5")
    [3.0, -4.0, 5.0]
    """
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    # split on commas or whitespace (any number)
    tokens = re.split(r"[\s,]+", text)
    nums: List[float] = []
    for t in tokens:
        if t == "":
            continue
        try:
            nums.append(float(t))
        except ValueError:
            raise ValueError(f"Unable to parse token as float: {t!r}")
    return nums


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns exit code (0 success)."""
    parser = argparse.ArgumentParser(
        prog="euclidean_norm",
        description="Compute the Euclidean (L2) norm of a vector of numbers",
    )
    parser.add_argument(
        "numbers",
        nargs="*",
        help="numbers of the vector (e.g. 3 -4 5)",
        type=float,
    )
    parser.add_argument(
        "-s",
        "--string",
        help='provide the vector as a single quoted string, e.g. "3, -4, 5"',
        type=str,
    )
    parser.add_argument(
        "-e",
        "--example",
        action="store_true",
        help="run a few built-in examples and exit",
    )

    args = parser.parse_args(argv)

    if args.example:
        examples = [([3, -4], "classic 3,-4 -> 5"), ([1, 2, 2], "should be 3"), ([], "empty -> 0")]
        for vec, desc in examples:
            print(f"{desc}: vector={vec} => norm={euclidean_norm(vec)}")
        return 0

    nums: List[float] = []
    if args.string:
        nums = parse_numbers(args.string)
    elif args.numbers:
        nums = list(args.numbers)
    else:
        # interactive prompt
        try:
            raw = input("Enter numbers separated by spaces or commas (e.g. 3 -4 5): ")
        except EOFError:
            print("No input provided. Exiting.")
            return 1
        nums = parse_numbers(raw)

    try:
        norm = euclidean_norm(nums)
    except ValueError as e:
        print(f"Error: {e}")
        return 2

    # print result with reasonable precision
    print(f"Euclidean norm: {norm:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

