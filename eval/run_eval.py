"""
eval/run_eval.py — MTG Judge Evaluation Script
================================================
Tests the /ask endpoint against a fixed set of questions and checks whether
the API's answer explicitly cites the expected rule number.

Metric: citation accuracy
    A test PASSES if the expected_rule_citation string appears anywhere in
    the generated answer. This is intentionally strict and simple — if the
    judge doesn't cite the rule at all, the answer is considered wrong even
    if it's factually correct.

Usage (server must be running first):
    uvicorn main:app --reload    # in one terminal
    python eval/run_eval.py      # in another terminal

No extra dependencies — uses only the Python standard library.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000/ask"

# Path is relative to this file's location, not where you run the script from.
TEST_CASES_FILE = Path(__file__).parent / "test_cases.json"

# Seconds to wait for a response before timing out.
REQUEST_TIMEOUT = 60

# Seconds to pause between requests — avoids hitting OpenAI rate limits.
PAUSE_BETWEEN_REQUESTS = 2


# ── API helper ────────────────────────────────────────────────────────────────

def ask_judge(question: str) -> dict:
    """
    POST a question to the /ask endpoint and return the parsed JSON response.

    We use urllib from the standard library so this script has zero extra
    dependencies beyond what main.py already requires.

    Raises:
        urllib.error.HTTPError  : server returned a 4xx/5xx status
        urllib.error.URLError   : could not connect (server not running, etc.)
        json.JSONDecodeError    : response was not valid JSON
    """
    payload = json.dumps({"question": question}).encode("utf-8")

    request = urllib.request.Request(
        url=API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


# ── Result helpers ────────────────────────────────────────────────────────────

def check_citation(answer: str, expected_rule: str) -> bool:
    """
    Return True if expected_rule appears in the answer text.

    The check is intentionally broad: "702.2" will match "702.2", "702.2b",
    "702.2a", etc. This is fine because we're checking that the judge
    *referenced the right section*, not that it cited one specific sub-rule.
    """
    return expected_rule in answer


def print_result(index: int, case: dict, passed: bool, answer: str = "", error: str = "") -> None:
    """Print a single test result line with optional detail on failure."""
    icon = "✓" if passed else "✗"
    label = "PASS" if passed else ("FAIL" if not error else "ERROR")
    rule = case["expected_rule_citation"]
    desc = case["description"]

    print(f"  {icon} [{index:02d}] {label}  rule={rule}  — {desc}")

    # On failure, show a truncated answer so the developer can investigate.
    if not passed:
        if error:
            print(f"         Error  : {error}")
        else:
            preview = answer[:120].replace("\n", " ")
            print(f"         Answer : {preview}{'...' if len(answer) > 120 else ''}")


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_eval() -> None:
    # ── Load test cases ───────────────────────────────────────────────────────
    if not TEST_CASES_FILE.exists():
        print(f"ERROR: Test cases file not found at {TEST_CASES_FILE}")
        sys.exit(1)

    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        test_cases: list[dict] = json.load(f)

    total = len(test_cases)
    print(f"MTG Judge Evaluation — {total} test cases\n")
    print(f"  Endpoint : {API_URL}")
    print(f"  Metric   : citation accuracy (expected rule cited in answer)")
    print("-" * 60)

    # ── Run each test ─────────────────────────────────────────────────────────
    passes = 0
    fails = 0
    errors = 0
    results: list[dict] = []  # collected for the detailed summary at the end

    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        expected_rule = case["expected_rule_citation"]

        try:
            response = ask_judge(question)
            answer = response.get("answer", "")
            keywords = response.get("keywords", [])
            passed = check_citation(answer, expected_rule)

            if passed:
                passes += 1
            else:
                fails += 1

            print_result(i, case, passed, answer=answer)
            results.append({
                "case": case,
                "passed": passed,
                "keywords": keywords,
                "answer": answer,
            })

        except urllib.error.HTTPError as e:
            # Server responded but with an error status (e.g. 500)
            errors += 1
            fails += 1
            err_body = e.read().decode("utf-8") if e.fp else str(e)
            print_result(i, case, passed=False, error=f"HTTP {e.code}: {err_body[:80]}")
            results.append({"case": case, "passed": False, "error": str(e)})

        except urllib.error.URLError as e:
            # Could not reach the server at all
            errors += 1
            fails += 1
            print_result(i, case, passed=False, error=f"Connection error: {e.reason}")
            results.append({"case": case, "passed": False, "error": str(e)})

            # If the very first request can't connect, no point continuing.
            if i == 1:
                print("\n  ⚠ Could not reach the API on the first request.")
                print("    Make sure the server is running: uvicorn main:app --reload")
                sys.exit(1)

        # Pause to avoid OpenAI rate-limit errors, except after the last case.
        if i < total:
            time.sleep(PAUSE_BETWEEN_REQUESTS)

    # ── Print summary ─────────────────────────────────────────────────────────
    accuracy = (passes / total) * 100 if total > 0 else 0.0

    print("-" * 60)
    print(f"\n  Results  : {passes} passed / {fails} failed / {errors} errors")
    print(f"  Accuracy : {accuracy:.1f}%  ({passes}/{total} rules correctly cited)")

    if fails > 0:
        print("\n  Failed cases:")
        for r in results:
            if not r.get("passed"):
                c = r["case"]
                print(f"    • [{c['id']:02d}] Expected rule {c['expected_rule_citation']} — {c['description']}")

    print()


if __name__ == "__main__":
    run_eval()
