import os
import sys
import json
import re

REPORT_DIR = "reports"
MAX_RAM_MB = 48000.0  # 48 GB ceiling (big model baseline is ~36 GB)
MAX_TIME_S = 600.0  # 10 mins max per sweep


def parse_report(filepath):
    print(f"🔍 Analyzing {filepath}...")
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found.")
        return False

    max_ram = 0.0
    execution_time = 0.0

    with open(filepath, "r") as f:
        content = f.read()

    # Parse JSON structured logs
    for line in content.split("\n"):
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if "ram_usage_mb" in data:
                    max_ram = max(max_ram, data["ram_usage_mb"])
            except json.JSONDecodeError:
                pass

    # Parse hotpath execution time (e.g. "[hotpath] 32.76s | timing, alloc, threads")
    time_match = re.search(r"\[hotpath\]\s+([\d\.]+)s", content)
    if time_match:
        execution_time = float(time_match.group(1))

    print(f"   Max RAM Usage: {max_ram:.2f} MB")
    print(f"   Total Time:    {execution_time:.2f} s")

    passed = True
    if max_ram > MAX_RAM_MB:
        print(f"   ❌ REGRESSION: RAM exceeded limit! ({max_ram:.2f} > {MAX_RAM_MB})")
        passed = False
    elif max_ram == 0.0:
        print(f"   ❌ ERROR: No RAM metrics found in {filepath}")
        passed = False

    if execution_time > MAX_TIME_S:
        print(
            f"   ❌ REGRESSION: Time exceeded limit! ({execution_time:.2f} > {MAX_TIME_S})"
        )
        passed = False
    elif execution_time == 0.0:
        print(f"   ❌ ERROR: No hotpath timing found in {filepath}")
        passed = False

    if passed:
        print("   ✅ Passed Limits!")
    return passed


def main():
    reports = [
        "hotpath_small_shallow.txt",
        "hotpath_small_deep.txt",
        "hotpath_big_shallow.txt",
        "hotpath_big_deep.txt",
    ]

    all_passed = True
    for report in reports:
        path = os.path.join(REPORT_DIR, report)
        if not parse_report(path):
            all_passed = False

    if all_passed:
        print("\n🎉 All 4 Quadrants passed performance and memory regression checks!")
        sys.exit(0)
    else:
        print("\n🚨 REGRESSIONS DETECTED! Profile check failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
