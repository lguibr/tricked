import os
import json
import re

REPORT_DIR = "reports"
OUTPUT_FILE = "PERFORMANCE_REPORT.md"
JSON_OUTPUT = os.path.join(REPORT_DIR, "performance_metrics.json")


def extract_json_metrics(content):
    match = re.search(r'(\{.*?"ram_usage_mb".*?\})', content)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None


def extract_table_section(content, title):
    lines = content.split("\n")
    in_section = False
    table_lines = []
    for line in lines:
        if line.startswith(title):
            in_section = True
        elif in_section and line.startswith("+"):
            table_lines.append(line)
        elif in_section and line.startswith("|"):
            table_lines.append(line)
        elif in_section and line.strip() == "" and len(table_lines) > 0:
            break

    if not table_lines:
        return ""

    md_table = []
    for idx, line in enumerate(table_lines):
        if line.startswith("+"):
            if idx == 1:
                cols = table_lines[0].count("|") - 1
                md_table.append("|" + "|".join(["---"] * cols) + "|")
            continue
        md_table.append(line)
    return "\n".join(md_table)


def parse_ascii_table(table_str):
    if not table_str:
        return []
    lines = table_str.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("|")]
    if len(data_lines) < 2:
        return []
    headers = [h.strip() for h in data_lines[0].strip("|").split("|")]

    rows = []
    for line in data_lines[1:]:
        cols = [c.strip() for c in line.strip("|").split("|")]
        rows.append(dict(zip(headers, cols)))
    return rows


def make_report():
    out = []
    out.append("# 🏎️ Tricked AI Performance Report")
    out.append(
        "This report outlines the hardware footprint and operational latency of the Tricked Engine across 4 benchmark quadrants.\n"
    )

    files = [
        ("Small_Shallow", "hotpath_small_shallow.txt"),
        ("Small_Deep", "hotpath_small_deep.txt"),
        ("Big_Shallow", "hotpath_big_shallow.txt"),
        ("Big_Deep", "hotpath_big_deep.txt"),
    ]

    all_json_data = {}

    for title, fname in files:
        fpath = os.path.join(REPORT_DIR, fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath, "r") as f:
            content = f.read()

        metrics = extract_json_metrics(content)
        time_match = re.search(r"\[hotpath\] ([\d\.]+)s", content)
        total_time_s = float(time_match.group(1)) if time_match else 0.0

        nps = 0
        if metrics and total_time_s > 0:
            nps = (
                metrics.get("mcts_depth_mean", 0) * metrics.get("game_count", 0)
            ) / total_time_s

        timing_md = extract_table_section(content, "timing - ")
        alloc_md = extract_table_section(content, "alloc-bytes - ")

        timing_data = parse_ascii_table(timing_md)
        alloc_data = parse_ascii_table(alloc_md)

        all_json_data[title] = {
            "total_time_s": total_time_s,
            "nps": nps,
            "metrics": metrics or {},
            "top_bottlenecks": timing_data,
            "top_allocations": alloc_data,
        }

        out.append(
            f"## {title.replace('_', ' Model, ').replace('Shallow', 'Shallow Search').replace('Deep', 'Deep Search')}"
        )
        out.append(f"**Total Execution Time:** `{total_time_s}s`")
        if metrics:
            out.append(f"- **RAM Usage:** `{metrics.get('ram_usage_mb', 0):.2f} MB`")
            out.append(f"- **GPU VRAM:** `{metrics.get('vram_usage_mb', 0):.2f} MB`")
            out.append(f"- **Nodes / Sec (NPS):** `{nps:.2f}`")

        out.append("\n### Top Execution Bottlenecks")
        if timing_md:
            out.append(timing_md)
        out.append("\n### Largest Memory Allocations")
        if alloc_md:
            out.append(alloc_md)
        out.append("\n---")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(out))

    with open(JSON_OUTPUT, "w") as f:
        json.dump(all_json_data, f, indent=4)

    print(f"Generated {OUTPUT_FILE} and {JSON_OUTPUT} successfully!")


if __name__ == "__main__":
    make_report()
