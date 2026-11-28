"""Executable Coding-Agent Benchmark (40 Tasks, ~190 Individual Unit Assertions)
Calibrated for `Qwen/Qwen3-0.6B` (596M params). IMMUTABLE once locked before Iteration 0!

Three real coding-agent skill categories:
  1. `synth` (16): Clean algorithmic Python function synthesis with edge cases.
  2. `debug` (12): Focused root-cause bug fixing given buggy Python code + symptom.
  3. `agent` (12): Stateful class / utility implementation & spec compliance.

Each task runs EVERY unit assertion independently inside an isolated Python subprocess
(`timeout=2.0s`), producing BOTH:
  - `pass_at_1`: strict fraction of tasks passing 100% of assertions (`hits / 40`)
  - `unit_rate`: granular fraction of individual unit assertions passing (`passed / total`)
"""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CODE_TASKS = [
    # =========================================================================
    # CATEGORY 1: ALGORITHMIC SYNTHESIS (16 tasks)
    # =========================================================================
    {
        "id": "synth_01_merge_intervals",
        "cat": "synth",
        "prompt": (
            "Write a Python function `merge_intervals(intervals)` that takes a list of `[start, end]` "
            "intervals (possibly unsorted or empty) and returns a sorted list of merged `[start, end]` "
            "intervals. Touching intervals like `[1, 3]` and `[3, 5]` must merge into `[1, 5]`. "
            "Return only the Python code."
        ),
        "asserts": [
            "merge_intervals([]) == []",
            "[list(x) for x in merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])] == [[1, 6], [8, 10], [15, 18]]",
            "[list(x) for x in merge_intervals([[5, 7], [1, 3], [3, 5]])] == [[1, 7]]",
            "[list(x) for x in merge_intervals([[4, 4], [2, 5], [1, 1]])] == [[1, 1], [2, 5]]",
            "[list(x) for x in merge_intervals([[-5, -1], [-3, 2], [2, 4]])] == [[-5, 4]]",
        ],
    },
    {
        "id": "synth_02_valid_brackets",
        "cat": "synth",
        "prompt": (
            "Write a Python function `valid_brackets(s)` that returns `True` if parentheses `()`, brackets `[]`, "
            "and braces `{}` in string `s` are properly matched and nested (ignoring non-bracket characters), "
            "else `False`. Return only the Python code."
        ),
        "asserts": [
            "valid_brackets('') is True",
            "valid_brackets('a(b[c]{d})e') is True",
            "valid_brackets('([)]') is False",
            "valid_brackets('(((') is False",
            "valid_brackets(']') is False",
        ],
    },
    {
        "id": "synth_03_rle_encode",
        "cat": "synth",
        "prompt": (
            "Write a Python function `rle_encode(s)` that run-length encodes string `s` by writing `<count><char>` "
            "for each consecutive run, omitting count `1` for single characters (e.g. `'AAABCC'` -> `'3AB2C'`, "
            "`'A'` -> `'A'`, `''` -> `''`). Return only the Python code."
        ),
        "asserts": [
            "rle_encode('') == ''",
            "rle_encode('A') == 'A'",
            "rle_encode('AAABCC') == '3AB2C'",
            "rle_encode('ABCD') == 'ABCD'",
            "rle_encode('A' * 12 + 'BB') == '12A2B'",
        ],
    },
    {
        "id": "synth_04_two_sum",
        "cat": "synth",
        "prompt": (
            "Write a Python function `two_sum(nums, target)` that returns a tuple `(i, j)` with `i < j` of indices "
            "such that `nums[i] + nums[j] == target`, or `None` if no such pair exists. Do not use the same index twice. "
            "Return only the Python code."
        ),
        "asserts": [
            "tuple(two_sum([2, 7, 11, 15], 9)) == (0, 1)",
            "tuple(two_sum([3, 2, 4], 6)) == (1, 2)",
            "tuple(two_sum([3, 3], 6)) == (0, 1)",
            "two_sum([1, 2, 3], 10) is None",
            "two_sum([], 0) is None",
        ],
    },
    {
        "id": "synth_05_group_anagrams",
        "cat": "synth",
        "prompt": (
            "Write a Python function `group_anagrams(words)` that groups lowercase words that are anagrams of each other "
            "into a list of lists (any group order is fine; preserve duplicates). Return only the Python code."
        ),
        "asserts": [
            "group_anagrams([]) == []",
            "sorted(sorted(g) for g in group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])) == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]",
            "sorted(sorted(g) for g in group_anagrams(['a'])) == [['a']]",
            "sorted(sorted(g) for g in group_anagrams(['ab', 'ba', 'ab'])) == [['ab', 'ab', 'ba']]",
        ],
    },
    {
        "id": "synth_06_longest_prefix",
        "cat": "synth",
        "prompt": (
            "Write a Python function `longest_common_prefix(strs)` that returns the longest common prefix string "
            "shared by all strings in list `strs` (or `''` if `strs` is empty or there is no common prefix). "
            "Return only the Python code."
        ),
        "asserts": [
            "longest_common_prefix([]) == ''",
            "longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'",
            "longest_common_prefix(['dog', 'racecar', 'car']) == ''",
            "longest_common_prefix(['only']) == 'only'",
            "longest_common_prefix(['ab', 'a']) == 'a'",
        ],
    },
    {
        "id": "synth_07_rotate_matrix",
        "cat": "synth",
        "prompt": (
            "Write a Python function `rotate_cw(matrix)` that returns a NEW `n x n` 2D list rotated 90 degrees clockwise "
            "without mutating the input matrix. Return `[]` if empty. Return only the Python code."
        ),
        "asserts": [
            "rotate_cw([]) == []",
            "[list(r) for r in rotate_cw([[1]])] == [[1]]",
            "[list(r) for r in rotate_cw([[1, 2], [3, 4]])] == [[3, 1], [4, 2]]",
            "[list(r) for r in rotate_cw([[1, 2, 3], [4, 5, 6], [7, 8, 9]])] == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]",
        ],
    },
    {
        "id": "synth_08_top_k_words",
        "cat": "synth",
        "prompt": (
            "Write a Python function `top_k_words(words, k)` that returns the `k` most frequent words from list `words`, "
            "sorted by highest frequency first, breaking ties alphabetically (ascending). Return only the Python code."
        ),
        "asserts": [
            "top_k_words(['i', 'love', 'code', 'i', 'love', 'python'], 2) == ['i', 'love']",
            "top_k_words(['b', 'a', 'c', 'a', 'b', 'b'], 2) == ['b', 'a']",
            "top_k_words(['z', 'a', 'm'], 2) == ['a', 'm']",
            "top_k_words([], 3) == []",
        ],
    },
    {
        "id": "synth_09_caesar_shift",
        "cat": "synth",
        "prompt": (
            "Write a Python function `caesar(text, shift)` that shifts ASCII letters (`a-z` and `A-Z`) forward by `shift` "
            "(wrapping modulo 26, supporting negative shifts too) while preserving case and leaving non-letters unchanged. "
            "Return only the Python code."
        ),
        "asserts": [
            "caesar('abc', 2) == 'cde'",
            "caesar('XyZ!', 3) == 'AbC!'",
            "caesar('AbC', -3) == 'XyZ'",
            "caesar('Hello, 123!', 26) == 'Hello, 123!'",
        ],
    },
    {
        "id": "synth_10_sliding_max",
        "cat": "synth",
        "prompt": (
            "Write a Python function `sliding_window_max(nums, k)` that returns a list containing the maximum value "
            "in every contiguous subarray of length `k`. If `nums` is empty, `k <= 0`, or `k > len(nums)`, return `[]`. "
            "Return only the Python code."
        ),
        "asserts": [
            "sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]",
            "sliding_window_max([4, 2, 1], 1) == [4, 2, 1]",
            "sliding_window_max([4, 2, 1], 3) == [4]",
            "sliding_window_max([4, 2], 5) == []",
            "sliding_window_max([], 2) == []",
        ],
    },
    {
        "id": "synth_11_roman_to_int",
        "cat": "synth",
        "prompt": (
            "Write a Python function `roman_to_int(s)` that converts a standard Roman numeral string `s` "
            "(including subtractive pairs `IV`, `IX`, `XL`, `XC`, `CD`, `CM`) into an integer. Return only the Python code."
        ),
        "asserts": [
            "roman_to_int('III') == 3",
            "roman_to_int('IV') == 4",
            "roman_to_int('IX') == 9",
            "roman_to_int('LVIII') == 58",
            "roman_to_int('MCMXCIV') == 1994",
        ],
    },
    {
        "id": "synth_12_min_coins",
        "cat": "synth",
        "prompt": (
            "Write a Python function `min_coins(coins, amount)` that returns the minimum number of coins from list `coins` "
            "needed to sum to `amount` (unlimited supply), or `-1` if impossible. Return `0` when `amount == 0`. "
            "Return only the Python code."
        ),
        "asserts": [
            "min_coins([1, 5, 10, 25], 0) == 0",
            "min_coins([1, 3, 4], 6) == 2",
            "min_coins([2], 3) == -1",
            "min_coins([2, 5, 10, 1], 27) == 4",
            "min_coins([186, 419, 83, 408], 6249) == 20",
        ],
    },
    {
        "id": "synth_13_is_palindrome",
        "cat": "synth",
        "prompt": (
            "Write a Python function `is_palindrome(s)` that returns `True` if string `s` is a palindrome "
            "considering ONLY alphanumeric characters and ignoring case, else `False`. Return only the Python code."
        ),
        "asserts": [
            "is_palindrome('A man, a plan, a canal: Panama') is True",
            "is_palindrome('race a car') is False",
            "is_palindrome('') is True",
            "is_palindrome('0P') is False",
            "is_palindrome('12321') is True",
        ],
    },
    {
        "id": "synth_14_chunk_list",
        "cat": "synth",
        "prompt": (
            "Write a Python function `chunk_list(items, size)` that splits list `items` into consecutive sublists "
            "of length `size` (the last chunk may be shorter). Raise `ValueError` if `size <= 0`. Return only the Python code."
        ),
        "asserts": [
            "chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
            "chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]",
            "chunk_list([], 3) == []",
            "(lambda: (chunk_list([1], 0) if False else True))()",
            "(lambda: (__import__('builtins').eval('True') if (lambda f: (f() or True))(lambda: (_ for _ in ()).throw(ValueError())) else False))() if False else (lambda: (lambda ok: ok)((lambda: [chunk_list([1,2], 0) for _ in [0]]) if False else True))()",
        ],
    },
    {
        "id": "synth_15_eval_rpn",
        "cat": "synth",
        "prompt": (
            "Write a Python function `eval_rpn(tokens)` that evaluates a list of Reverse Polish Notation string tokens "
            "supporting `+`, `-`, `*`, `/` where division truncates toward zero (e.g. `int(-7 / 2) == -3`). "
            "Return an `int`. Return only the Python code."
        ),
        "asserts": [
            "eval_rpn(['2', '1', '+', '3', '*']) == 9",
            "eval_rpn(['4', '13', '5', '/', '+']) == 6",
            "eval_rpn(['-7', '2', '/']) == -3",
            "eval_rpn(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22",
            "eval_rpn(['-5']) == -5",
        ],
    },
    {
        "id": "synth_16_semver_sort",
        "cat": "synth",
        "prompt": (
            "Write a Python function `sort_versions(versions)` that sorts a list of numeric dot-separated version strings "
            "(e.g. `'1.10.0'`, `'1.2.0'`, `'0.9.1'`, `'1.2'`) numerically by component (so `'1.2.0'` < `'1.10.0'`, "
            "and `'1.2'` == `'1.2.0'` or sorts before `'1.2.1'`). Return a new sorted list. Return only the Python code."
        ),
        "asserts": [
            "sort_versions([]) == []",
            "sort_versions(['1.10.0', '1.2.0', '0.9.9']) == ['0.9.9', '1.2.0', '1.10.0']",
            "sort_versions(['2.0', '1.9.9', '2.0.1']) == ['1.9.9', '2.0', '2.0.1']",
            "sort_versions(['10.0.0', '2.0.0', '1.100.0']) == ['1.100.0', '2.0.0', '10.0.0']",
        ],
    },

    # =========================================================================
    # CATEGORY 2: FOCUSED BUG-FIXING / DEBUGGING (12 tasks)
    # =========================================================================
    {
        "id": "debug_01_bsearch_first",
        "cat": "debug",
        "prompt": (
            "Fix the bug in `first_occurrence(arr, target)` so it returns the FIRST index of `target` in sorted list `arr` "
            "(or `-1` if not found). Currently it stops at a middle duplicate or infinite loops:\n"
            "```python\n"
            "def first_occurrence(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "first_occurrence([1, 2, 2, 2, 3], 2) == 1",
            "first_occurrence([2, 2, 2, 2], 2) == 0",
            "first_occurrence([1, 3, 5], 5) == 2",
            "first_occurrence([1, 3, 5], 2) == -1",
            "first_occurrence([], 1) == -1",
            "first_occurrence([7], 7) == 0",
        ],
    },
    {
        "id": "debug_02_mutable_default",
        "cat": "debug",
        "prompt": (
            "Fix the mutable-default bug in `append_unique(item, lst=[])` so calling `append_unique(1)` twice "
            "returns two independent `[1]` lists, and if `item` is already in `lst`, it is not added again:\n"
            "```python\n"
            "def append_unique(item, lst=[]):\n"
            "    lst.append(item)\n"
            "    return lst\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "append_unique(1) == [1]",
            "append_unique(2) == [2]",
            "append_unique(1, [1, 2]) == [1, 2]",
            "append_unique(3, [1, 2]) == [1, 2, 3]",
        ],
    },
    {
        "id": "debug_03_dict_del_loop",
        "cat": "debug",
        "prompt": (
            "Fix the bug in `drop_keys(d, bad_keys)` which raises `RuntimeError: dictionary changed size during iteration` "
            "and mutates the input dict `d`. It should return a NEW dict with all keys in `bad_keys` removed (recursively for nested dicts):\n"
            "```python\n"
            "def drop_keys(d, bad_keys):\n"
            "    for k, v in d.items():\n"
            "        if k in bad_keys:\n"
            "            del d[k]\n"
            "        elif isinstance(v, dict):\n"
            "            drop_keys(v, bad_keys)\n"
            "    return d\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "drop_keys({'a': 1, 'b': 2}, {'b'}) == {'a': 1}",
            "drop_keys({'a': {'x': 1, 'y': 2}, 'x': 3}, {'x'}) == {'a': {'y': 2}}",
            "(lambda src: (drop_keys(src, {'x'}) == {'a': 1}) and ('x' in src))({'a': 1, 'x': 9})",
            "drop_keys({}, {'x'}) == {}",
        ],
    },
    {
        "id": "debug_04_bfs_shortest",
        "cat": "debug",
        "prompt": (
            "Fix `shortest_path(graph, start, goal)` on an unweighted directed graph `{node: [neighbors]}` so it uses BFS "
            "with a visited set (returns shortest node list from `start` to `goal`, or `None` if unreachable, without looping on cycles):\n"
            "```python\n"
            "def shortest_path(graph, start, goal):\n"
            "    stack = [(start, [start])]\n"
            "    while stack:\n"
            "        node, path = stack.pop()\n"
            "        if node == goal:\n"
            "            return path\n"
            "        for nxt in graph.get(node, []):\n"
            "            stack.append((nxt, path + [nxt]))\n"
            "    return None\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "shortest_path({'A': ['B', 'C'], 'B': ['D'], 'C': ['B', 'E'], 'D': ['E'], 'E': ['A']}, 'A', 'E') == ['A', 'C', 'E']",
            "shortest_path({'A': ['B'], 'B': ['A']}, 'A', 'A') == ['A']",
            "shortest_path({'A': ['B'], 'B': ['A']}, 'A', 'Z') is None",
            "shortest_path({'A': ['B'], 'B': ['C']}, 'A', 'C') == ['A', 'B', 'C']",
        ],
    },
    {
        "id": "debug_05_flatten_str",
        "cat": "debug",
        "prompt": (
            "Fix the bug in `flatten(seq)` which recurses infinitely on strings because `str` is iterable. "
            "Only flatten `list` and `tuple` elements into a single flat list, keeping strings and dicts intact:\n"
            "```python\n"
            "def flatten(seq):\n"
            "    out = []\n"
            "    for x in seq:\n"
            "        if hasattr(x, '__iter__'):\n"
            "            out.extend(flatten(x))\n"
            "        else:\n"
            "            out.append(x)\n"
            "    return out\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "flatten([1, [2, (3, 'abc')], []]) == [1, 2, 3, 'abc']",
            "flatten(['hi', [4, {'k': 1}]]) == ['hi', 4, {'k': 1}]",
            "flatten([]) == []",
            "flatten([[1], [[2]], [[[3]]]]) == [1, 2, 3]",
        ],
    },
    {
        "id": "debug_06_generator_reuse",
        "cat": "debug",
        "prompt": (
            "Fix the bug in `normalize_scores(scores)` so it works on ANY iterable (including a generator!), "
            "scales values to `[0.0, 1.0]` via `(x - mn) / (mx - mn)`, returns all `0.0`s if `mn == mx`, and `[]` if empty:\n"
            "```python\n"
            "def normalize_scores(scores):\n"
            "    mn = min(scores)\n"
            "    mx = max(scores)\n"
            "    return [(x - mn) / (mx - mn) for x in scores]\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "normalize_scores(x for x in [10, 20, 30]) == [0.0, 0.5, 1.0]",
            "normalize_scores(x for x in [5, 5, 5]) == [0.0, 0.0, 0.0]",
            "normalize_scores(x for x in []) == []",
            "normalize_scores([7]) == [0.0]",
        ],
    },
    {
        "id": "debug_07_interval_sweep",
        "cat": "debug",
        "prompt": (
            "Fix `max_concurrent(events)` for half-open intervals `[start, end)` where an event ending at `t` "
            "does NOT overlap one starting at `t`. Currently touching events like `(1, 5)` and `(5, 10)` return `2` instead of `1`:\n"
            "```python\n"
            "def max_concurrent(events):\n"
            "    pts = []\n"
            "    for s, e in events:\n"
            "        pts.append((s, 1))\n"
            "        pts.append((e, -1))\n"
            "    pts.sort(key=lambda x: x[0])\n"
            "    cur = best = 0\n"
            "    for _, d in pts:\n"
            "        cur += d\n"
            "        best = max(best, cur)\n"
            "    return best\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "max_concurrent([]) == 0",
            "max_concurrent([(1, 5), (5, 10), (10, 15)]) == 1",
            "max_concurrent([(5, 10), (1, 5)]) == 1",
            "max_concurrent([(1, 4), (2, 5), (3, 6)]) == 3",
            "max_concurrent([(1, 10), (2, 3), (3, 4)]) == 2",
        ],
    },
    {
        "id": "debug_08_dedup_order",
        "cat": "debug",
        "prompt": (
            "Fix `dedup_preserve(items)` which loses original order because it converts to a `set`:\n"
            "```python\n"
            "def dedup_preserve(items):\n"
            "    return list(set(items))\n"
            "```\nReturn a list containing the first occurrence of each distinct element in original order. Return only the Python function."
        ),
        "asserts": [
            "dedup_preserve([3, 1, 2, 1, 3, 4]) == [3, 1, 2, 4]",
            "dedup_preserve(['b', 'a', 'b', 'c', 'a']) == ['b', 'a', 'c']",
            "dedup_preserve([]) == []",
            "dedup_preserve([5, 5, 5]) == [5]",
        ],
    },
    {
        "id": "debug_09_memo_leak",
        "cat": "debug",
        "prompt": (
            "Fix the global-cache leak bug in `count_ways(n, steps)` which returns the number of ordered sequences of positive "
            "jumps from `steps` summing to `n` (deduplicating `steps`). Currently calling it with a different `steps` returns stale answers:\n"
            "```python\n"
            "cache = {}\n"
            "def count_ways(n, steps):\n"
            "    if n == 0: return 1\n"
            "    if n < 0: return 0\n"
            "    if n in cache: return cache[n]\n"
            "    cache[n] = sum(count_ways(n - s, steps) for s in steps)\n"
            "    return cache[n]\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "count_ways(4, [1, 2]) == 5",
            "count_ways(4, [1, 3]) == 3",
            "count_ways(4, [2, 2]) == 1",
            "count_ways(0, [1, 2]) == 1",
            "count_ways(25, [1, 2]) == 121393",
        ],
    },
    {
        "id": "debug_10_word_redact",
        "cat": "debug",
        "prompt": (
            "Fix `redact_word(text, word)` which corrupts substrings (replacing `'cat'` inside `'scatter'`) and is case-sensitive. "
            "Replace only whole-word, case-insensitive matches of `word` with `'[REDACTED]'`:\n"
            "```python\n"
            "def redact_word(text, word):\n"
            "    return text.replace(word, '[REDACTED]')\n"
            "```\nReturn only the fixed Python function."
        ),
        "asserts": [
            "redact_word('The Cat did not scatter cats.', 'cat') == 'The [REDACTED] did not scatter cats.'",
            "redact_word('Secret and SECRET!', 'secret') == '[REDACTED] and [REDACTED]!'",
            "redact_word('catalog cat', 'cat') == 'catalog [REDACTED]'",
            "redact_word('nothing here', 'cat') == 'nothing here'",
        ],
    },
    {
        "id": "debug_11_heap_tie",
        "cat": "debug",
        "prompt": (
            "Fix `TaskQueue` where pushing two dict payloads with equal priority crashes `heapq` with `TypeError`. "
            "Lower priority pops first, and equal priorities must pop in FIFO insertion order:\n"
            "```python\n"
            "import heapq\n"
            "class TaskQueue:\n"
            "    def __init__(self):\n"
            "        self.h = []\n"
            "    def push(self, priority, payload):\n"
            "        heapq.heappush(self.h, (priority, payload))\n"
            "    def pop(self):\n"
            "        return heapq.heappop(self.h)[1]\n"
            "```\nReturn only the fixed Python class."
        ),
        "asserts": [
            "(lambda q: (q.push(2, {'id': 'a'}), q.push(1, {'id': 'b'}), q.push(2, {'id': 'c'}), q.push(1, {'id': 'd'}), [q.pop()['id'] for _ in range(4)])[-1])(TaskQueue()) == ['b', 'd', 'a', 'c']",
            "(lambda q: (q.push(5, {'x': 1}), q.pop())[-1])(TaskQueue()) == {'x': 1}",
        ],
    },
    {
        "id": "debug_12_lru_get_put",
        "cat": "debug",
        "prompt": (
            "Fix `LRUCache(capacity)` so `get(key)` marks the key as most-recently-used, updating an existing key in `put` "
            "does not evict another key, and `capacity <= 0` safely stores nothing:\n"
            "```python\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.cap = capacity\n"
            "        self.cache = {}\n"
            "    def get(self, key):\n"
            "        return self.cache.get(key, -1)\n"
            "    def put(self, key, value):\n"
            "        if len(self.cache) >= self.cap:\n"
            "            self.cache.pop(next(iter(self.cache)))\n"
            "        self.cache[key] = value\n"
            "```\nReturn only the fixed Python class."
        ),
        "asserts": [
            "(lambda c: (c.put(1, 10), c.put(2, 20), c.get(1), c.put(3, 30), (c.get(2), c.get(1), c.get(3)))[-1])(LRUCache(2)) == (-1, 10, 30)",
            "(lambda c: (c.put(1, 10), c.put(2, 20), c.put(1, 99), (c.get(1), c.get(2)))[-1])(LRUCache(2)) == (99, 20)",
            "(lambda c: (c.put(1, 10), c.get(1))[-1])(LRUCache(0)) == -1",
        ],
    },

    # =========================================================================
    # CATEGORY 3: STATEFUL CODING-AGENT UTILITIES & SPECS (12 tasks)
    # =========================================================================
    {
        "id": "agent_01_bank_account",
        "cat": "agent",
        "prompt": (
            "Implement a `BankAccount(balance=0)` class with:\n"
            "- `deposit(amount)`: adds `amount` (raise `ValueError` if `amount <= 0`)\n"
            "- `withdraw(amount)`: subtracts `amount` (raise `ValueError` if `amount <= 0` or `amount > self.balance`)\n"
            "- `transfer(other, amount)`: atomically withdraws from `self` and deposits into `other`.\n"
            "Return only the Python class."
        ),
        "asserts": [
            "(lambda a: (a.deposit(50), a.withdraw(20), a.balance)[-1])(BankAccount(10)) == 40",
            "(lambda a, b: (a.transfer(b, 30), (a.balance, b.balance))[-1])(BankAccount(100), BankAccount(5)) == (70, 35)",
            "(lambda a: (lambda: a.withdraw(500))) and True",
        ],
    },
    {
        "id": "agent_02_rate_limiter",
        "cat": "agent",
        "prompt": (
            "Implement `RateLimiter(max_calls, window)` with method `allow(user, now)` where `now` is a float timestamp. "
            "Return `True` and record `now` if `user` has fewer than `max_calls` recorded timestamps strictly greater than "
            "`now - window`; otherwise return `False` without recording. Return only the Python class."
        ),
        "asserts": [
            "(lambda l: [l.allow('u1', 1.0), l.allow('u1', 5.0), l.allow('u1', 6.0)])(RateLimiter(2, 10.0)) == [True, True, False]",
            "(lambda l: (l.allow('u1', 1.0), l.allow('u1', 5.0), l.allow('u1', 6.0), l.allow('u1', 11.0))[-1])(RateLimiter(2, 10.0)) is True",
            "(lambda l: (l.allow('u1', 1.0), l.allow('u2', 1.0))[-1])(RateLimiter(1, 10.0)) is True",
        ],
    },
    {
        "id": "agent_03_nested_get",
        "cat": "agent",
        "prompt": (
            "Write a Python function `get_path(d, path, default=None)` that reads a nested dict using dot-separated string `path` "
            "(e.g. `'a.b.c'`). Return `default` if any key is missing or an intermediate value is not a `dict`. Return only the Python code."
        ),
        "asserts": [
            "get_path({'a': {'b': {'c': 42}}}, 'a.b.c') == 42",
            "get_path({'a': {'b': 3}}, 'a.b.c', -1) == -1",
            "get_path({'a': 1}, 'x.y', 'none') == 'none'",
            "get_path({'a': 0}, 'a', 99) == 0",
        ],
    },
    {
        "id": "agent_04_dict_diff",
        "cat": "agent",
        "prompt": (
            "Write a Python function `dict_diff(old, new)` that returns `{'added': ..., 'removed': ..., 'changed': ...}` "
            "where `'added'` has keys only in `new`, `'removed'` has keys only in `old`, and `'changed'` maps shared keys "
            "with differing values to `(old[k], new[k])`. Return only the Python code."
        ),
        "asserts": [
            "dict_diff({'x': 1, 'y': 2, 'z': 3}, {'y': 2, 'z': 30, 'w': 4}) == {'added': {'w': 4}, 'removed': {'x': 1}, 'changed': {'z': (3, 30)}}",
            "dict_diff({'a': 1}, {'a': 1}) == {'added': {}, 'removed': {}, 'changed': {}}",
            "dict_diff({}, {'a': 2}) == {'added': {'a': 2}, 'removed': {}, 'changed': {}}",
        ],
    },
    {
        "id": "agent_05_query_filter",
        "cat": "agent",
        "prompt": (
            "Write a Python function `filter_rows(rows, spec)` where `rows` is a list of dicts and `spec` is a dict of "
            "AND conditions: `'k': v` checks `row.get(k) == v`, `'k__gt': v` checks `row.get(k) is not None and row[k] > v`, "
            "`'k__lt': v` checks `row.get(k) is not None and row[k] < v`. Return matching rows in order. Return only the Python code."
        ),
        "asserts": [
            "[r['id'] for r in filter_rows([{'id': 1, 'env': 'prod', 'age': 5}, {'id': 2, 'env': 'dev', 'age': 8}, {'id': 3, 'env': 'prod', 'age': 2}], {'env': 'prod', 'age__gt': 3})] == [1]",
            "[r['id'] for r in filter_rows([{'id': 1, 'x': 5}, {'id': 2}], {'x__lt': 10})] == [1]",
            "filter_rows([], {'env': 'prod'}) == []",
        ],
    },
    {
        "id": "agent_06_order_fsm",
        "cat": "agent",
        "prompt": (
            "Implement an `OrderFSM` class starting with `self.state = 'NEW'` and transitions: "
            "`('NEW', 'pay') -> 'PAID'`, `('NEW', 'cancel') -> 'CANCELLED'`, `('PAID', 'ship') -> 'SHIPPED'`, "
            "`('SHIPPED', 'deliver') -> 'DELIVERED'`. Method `apply(event)` transitions and returns `self.state`, "
            "or raises `ValueError` without changing `self.state` if invalid. Return only the Python class."
        ),
        "asserts": [
            "(lambda f: (f.apply('pay'), f.apply('ship'), f.apply('deliver'), f.state)[-1])(OrderFSM()) == 'DELIVERED'",
            "(lambda f: (f.apply('cancel'), f.state)[-1])(OrderFSM()) == 'CANCELLED'",
            "OrderFSM().state == 'NEW'",
        ],
    },
    {
        "id": "agent_07_undo_stack",
        "cat": "agent",
        "prompt": (
            "Implement a `TextBuffer` class with `self.text = ''`, `append(s)` to add string `s`, "
            "`undo()` to revert the last `append` (no-op if empty), and `redo()` to re-apply the last undone `append` "
            "(cleared whenever a new `append` happens). Return only the Python class."
        ),
        "asserts": [
            "(lambda b: (b.append('hi'), b.append(' there'), b.undo(), b.text)[-1])(TextBuffer()) == 'hi'",
            "(lambda b: (b.append('a'), b.append('b'), b.undo(), b.redo(), b.text)[-1])(TextBuffer()) == 'ab'",
            "(lambda b: (b.append('a'), b.undo(), b.append('c'), b.redo(), b.text)[-1])(TextBuffer()) == 'c'",
            "(lambda b: (b.undo(), b.text)[-1])(TextBuffer()) == ''",
        ],
    },
    {
        "id": "agent_08_template_fill",
        "cat": "agent",
        "prompt": (
            "Write a Python function `render(template, ctx)` that replaces `{{ key }}` (allowing optional spaces around `key`) "
            "with `str(ctx[key])`. If `key` is not in `ctx`, keep the placeholder unchanged. Return only the Python code."
        ),
        "asserts": [
            "render('Hello {{ name }}!', {'name': 'Ada'}) == 'Hello Ada!'",
            "render('{{a}}+{{ b }}={{c}}', {'a': 1, 'b': 2, 'c': 3}) == '1+2=3'",
            "render('Hi {{ missing }}', {}) == 'Hi {{ missing }}'",
            "render('Count={{ n }}', {'n': 0}) == 'Count=0'",
        ],
    },
    {
        "id": "agent_09_log_summary",
        "cat": "agent",
        "prompt": (
            "Write a Python function `count_logs(lines)` where each valid line has `'LEVEL service'` (2 whitespace-separated tokens). "
            "Ignore empty or malformed lines. Return a dict mapping each `service` to `{'total': int, 'errors': int}` where "
            "`'errors'` counts lines with `LEVEL == 'ERROR'`. Return only the Python code."
        ),
        "asserts": [
            "count_logs(['INFO api', 'ERROR api', 'WARN db', 'bad', '']) == {'api': {'total': 2, 'errors': 1}, 'db': {'total': 1, 'errors': 0}}",
            "count_logs([]) == {}",
            "count_logs(['ERROR auth', 'ERROR auth']) == {'auth': {'total': 2, 'errors': 2}}",
        ],
    },
    {
        "id": "agent_10_bounded_stack",
        "cat": "agent",
        "prompt": (
            "Implement a `BoundedStack(limit)` class with `push(x)` (if full, drops the OLDEST/bottom item before pushing `x`), "
            "`pop()` (removes and returns top item, or returns `None` if empty), and `items()` returning a list from bottom to top. "
            "Return only the Python class."
        ),
        "asserts": [
            "(lambda s: (s.push(1), s.push(2), s.push(3), s.items())[-1])(BoundedStack(2)) == [2, 3]",
            "(lambda s: (s.push(1), s.push(2), s.pop(), s.items())[-1])(BoundedStack(3)) == [1]",
            "BoundedStack(2).pop() is None",
        ],
    },
    {
        "id": "agent_11_migrate_user",
        "cat": "agent",
        "prompt": (
            "Write a Python function `migrate_user(rec)` returning a NEW dict (without mutating `rec`): "
            "split `'full_name'` on the first space into `'first'` and `'last'` (empty string if no space), "
            "replace `'disabled'` (bool, default `False`) with `'active': not disabled`, remove `'full_name'` and `'disabled'`, "
            "and keep all other keys. Return only the Python code."
        ),
        "asserts": [
            "migrate_user({'id': 1, 'full_name': 'Ada Lovelace', 'disabled': False}) == {'id': 1, 'first': 'Ada', 'last': 'Lovelace', 'active': True}",
            "migrate_user({'full_name': 'Plato', 'disabled': True}) == {'first': 'Plato', 'last': '', 'active': False}",
            "(lambda r: (migrate_user(r)['active'] is True) and ('full_name' in r))({'full_name': 'A B'})",
        ],
    },
    {
        "id": "agent_12_dep_order",
        "cat": "agent",
        "prompt": (
            "Write a Python function `install_order(deps, target)` where `deps` maps package -> list of required dependency names. "
            "Return a list of packages in valid install order (all dependencies before the package that needs them, no duplicates, "
            "ending with `target`). Return `None` if a cycle is detected. Return only the Python code."
        ),
        "asserts": [
            "(lambda o: o is not None and o[-1] == 'app' and set(o) == {'app', 'web', 'db', 'http'} and o.index('http') < o.index('web') and o.index('web') < o.index('app') and o.index('db') < o.index('app'))(install_order({'app': ['web', 'db'], 'web': ['http'], 'db': [], 'http': []}, 'app'))",
            "install_order({'a': ['b'], 'b': ['a']}, 'a') is None",
            "install_order({'x': []}, 'x') == ['x']",
        ],
    },
]

# Fix synth_14 ugly placeholder assertion cleanly with an explicit exception check helper
CODE_TASKS[13]["asserts"] = [
    "chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
    "chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]",
    "chunk_list([], 3) == []",
    "_raises(ValueError, lambda: chunk_list([1, 2], 0))",
]
# Also add real ValueError checks to agent_01 and agent_06
CODE_TASKS[28]["asserts"][2] = "_raises(ValueError, lambda: BankAccount(10).withdraw(50))"
CODE_TASKS[33]["asserts"].append("_raises(ValueError, lambda: OrderFSM().apply('deliver'))")


def extract_python_code(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0].strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    m = re.search(r"^(?:import |from |def |class )", cleaned, flags=re.MULTILINE)
    if m:
        return cleaned[m.start():].strip()
    return cleaned


def _run_one_task(code_str: str, asserts: list, timeout_s: float = 2.0):
    """Run all assertions independently inside one isolated subprocess; return (n_pass, n_total, first_err)."""
    runner = (
        "import json, sys\n"
        "def _raises(exc_t, fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return False\n"
        "    except exc_t:\n"
        "        return True\n"
        "    except Exception:\n"
        "        return False\n"
        f"{code_str}\n\n"
        f"_TESTS = {repr(asserts)}\n"
        "_ok = 0\n"
        "_err = ''\n"
        "for _expr in _TESTS:\n"
        "    try:\n"
        "        if bool(eval(_expr, globals())):\n"
        "            _ok += 1\n"
        "        elif not _err:\n"
        "            _err = 'False: ' + _expr[:70]\n"
        "    except Exception as _e:\n"
        "        if not _err:\n"
        "            _err = f'{type(_e).__name__}: {_e}'[:120]\n"
        "print('__RES__' + json.dumps([_ok, len(_TESTS), _err]))\n"
    )
    try:
        res = subprocess.run([sys.executable, "-I", "-c", runner], capture_output=True, text=True, timeout=timeout_s)
        for line in reversed(res.stdout.splitlines()):
            if line.startswith("__RES__"):
                ok_n, tot_n, err = eval(line[len("__RES__"):]) if False else __import__("json").loads(line[len("__RES__"):])
                return int(ok_n), int(tot_n), str(err)
        err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else f"rc={res.returncode}"
        return 0, len(asserts), err[:120]
    except subprocess.TimeoutExpired:
        return 0, len(asserts), "TIMEOUT"
    except Exception as e:
        return 0, len(asserts), str(e)[:120]


def grade_generations(generations, tasks=CODE_TASKS, max_workers=10, timeout_s=2.0):
    assert len(generations) == len(tasks)
    codes = [extract_python_code(g) for g in generations]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one_task, c, t["asserts"], timeout_s) for c, t in zip(codes, tasks)]
        results = [f.result() for f in futs]

    by_cat = {}
    per_task = []
    task_hits = 0
    unit_hits = 0
    unit_tot = 0
    for t, (ok_n, tot_n, err) in zip(tasks, results):
        full_ok = (ok_n == tot_n and tot_n > 0)
        task_hits += int(full_ok)
        unit_hits += ok_n
        unit_tot += tot_n
        c = t["cat"]
        by_cat.setdefault(c, [0, 0, 0, 0])
        by_cat[c][0] += int(full_ok)
        by_cat[c][1] += 1
        by_cat[c][2] += ok_n
        by_cat[c][3] += tot_n
        per_task.append({"id": t["id"], "cat": c, "ok": full_ok, "units": f"{ok_n}/{tot_n}", "err": err})

    return {
        "pass_at_1": round(task_hits / max(1, len(tasks)), 5),
        "unit_rate": round(unit_hits / max(1, unit_tot), 5),
        "hits": task_hits,
        "total": len(tasks),
        "unit_hits": unit_hits,
        "unit_total": unit_tot,
        "by_cat_counts": {k: f"{v[0]}/{v[1]} (u:{v[2]}/{v[3]})" for k, v in sorted(by_cat.items())},
        "per_task": per_task,
    }
