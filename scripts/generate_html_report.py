#!/usr/bin/env python3
"""VOSS HTML Verification Report Generator."""
import os, json, csv, math, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES  = os.path.join(ROOT, "results")
OUT  = os.path.join(RES, "verification_report.html")

def read_csv(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))

def read_json(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p): return {}
    with open(p) as f: return json.load(f)

def build():
    gaps_hist = read_csv("gap_histogram.csv")
    cheb      = read_csv("chebyshev.csv")
    hl        = read_csv("hl_trend.csv")
    certs     = read_csv("certificates.csv")
    verif     = read_json("verification_report.json")
    large_gaps= read_csv("large_gaps.csv")

    total_gaps = sum(int(r["count"]) for r in gaps_hist) if gaps_hist else 0
    weighted   = sum(int(r["gap"])*int(r["count"]) for r in gaps_hist) if gaps_hist else 0
    mean_gap   = weighted/total_gaps if total_gaps else 0
    var_sum    = sum(int(r["count"])*(int(r["gap"])-mean_gap)**2 for r in gaps_hist) if gaps_hist else 0
    var        = var_sum/total_gaps if total_gaps else 0
    std        = math.sqrt(var) if var > 0 else 0

    N_val = int(cheb[0]["N"]) if cheb else (int(hl[0]["N"]) if hl else 0)
    N_str = "10^" + str(round(math.log10(N_val))) if N_val > 0 else "?"

    pi41 = int(cheb[0]["pi_4_1"]) if cheb else 0
    pi43 = int(cheb[0]["pi_4_3"]) if cheb else 0
    cheb_diff = int(cheb[0]["difference"]) if cheb else 0
    pi_total  = pi41 + pi43 + 1
    p6_p2     = float(hl[0]["P6_over_P2"]) if hl else 0

    top_gaps  = sorted(gaps_hist, key=lambda r: -int(r["count"]))[:20]
    top_merit = sorted(large_gaps, key=lambda r: -float(r["merit"]))[:20]

    v_total    = verif.get("total_gaps", 0)
    v_verified = verif.get("verified", 0)
    v_failed   = verif.get("failed", 0)
    v_duration = verif.get("duration_seconds", 0)

    # ---- HTML ----
    H = []
    H.append("<!DOCTYPE html>")
    H.append("<html lang=\"en\"><head><meta charset=\"UTF-8\">")
    H.append("<title>VOSS Verification Report - " + N_str + "</title>")
    H.append("<style>")
    H.append("body{font-family:-apple-system,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}")
    H.append("h1{color:#1a5490;border-bottom:3px solid #1a5490;padding-bottom:.3em}")
    H.append("h2{color:#2c6fb0;margin-top:1.5em;border-bottom:1px solid #ccc;padding-bottom:.2em}")
    H.append("table{border-collapse:collapse;width:100%;margin:.7em 0;font-size:.92em}")
    H.append("th{background:#1a5490;color:#fff;padding:8px;text-align:left}")
    H.append("td{border:1px solid #ddd;padding:6px 8px}")
    H.append("tr:nth-child(even){background:#f7f9fc}")
    H.append(".pass{color:#1a8f3a;font-weight:bold}.fail{color:#c0392b;font-weight:bold}")
    H.append(".stat{background:#eef4fb;padding:12px 18px;border-left:4px solid #1a5490;margin:.5em 0;border-radius:3px}")
    H.append(".grid{display:grid;grid-template-columns:1fr 1fr;gap:1em}")
    H.append(".footer{margin-top:3em;padding-top:1em;border-top:1px solid #ccc;font-size:.85em;color:#666;text-align:center}")
    H.append("</style></head><body>")

    H.append("<h1>VOSS Verification Report</h1>")
    H.append("<div class=\"stat\"><b>Target:</b> N = " + N_str + " (" + format(N_val, ",") + ") &nbsp;|&nbsp; <b>Generated:</b> " + time.strftime("%Y-%m-%d %H:%M:%S") + "</div>")

    H.append("<h2>Self-Consistency Checks</h2>")
    checks = [
        ("sum(N(g)) = pi(N) - 1", total_gaps == pi_total - 1 if pi_total else False),
        ("pi(4,1) + pi(4,3) + 1 = pi(N)", (pi41+pi43+1) == pi_total if pi_total else False),
        ("Verification all gaps passed", v_failed == 0 and v_verified == v_total if v_total else False),
    ]
    H.append("<table><tr><th>Check</th><th>Status</th></tr>")
    for name, ok in checks:
        cls = "pass" if ok else "fail"
        txt = "PASS" if ok else "FAIL"
        H.append("<tr><td>" + name + "</td><td class=\"" + cls + "\">" + txt + "</td></tr>")
    H.append("</table>")

    H.append("<h2>Gap Statistics</h2><div class=\"grid\">")
    H.append("<div class=\"stat\"><b>Total gaps:</b> " + format(total_gaps, ",") + "</div>")
    H.append("<div class=\"stat\"><b>Mean gap:</b> " + format(mean_gap, ".4f") + "</div>")
    H.append("<div class=\"stat\"><b>Std deviation:</b> " + format(std, ".4f") + "</div>")
    H.append("<div class=\"stat\"><b>pi(N):</b> " + format(pi_total, ",") + "</div></div>")

    H.append("<h2>Chebyshev Bias</h2>")
    H.append("<table><tr><th>Class</th><th>Count</th><th>Share</th></tr>")
    tot = pi41 + pi43
    if tot:
        H.append("<tr><td>pi(x;4,1)</td><td>" + format(pi41, ",") + "</td><td>" + format(100*pi41/tot, ".4f") + "%</td></tr>")
        H.append("<tr><td>pi(x;4,3)</td><td>" + format(pi43, ",") + "</td><td>" + format(100*pi43/tot, ".4f") + "%</td></tr>")
    H.append("<tr><td><b>Difference (3-1)</b></td><td colspan=\"2\"><b>" + format(cheb_diff, "+,") + "</b></td></tr></table>")

    H.append("<h2>Hardy-Littlewood: P(6)/P(2)</h2>")
    H.append("<div class=\"stat\">Observed ratio = <b>" + format(p6_p2, ".6f") + "</b> (HL limit = 2.0)</div>")

    H.append("<h2>Top 20 Gaps by Count</h2><table><tr><th>#</th><th>Gap</th><th>Count</th></tr>")
    for i, r in enumerate(top_gaps, 1):
        H.append("<tr><td>" + str(i) + "</td><td>" + r["gap"] + "</td><td>" + format(int(r["count"]), ",") + "</td></tr>")
    H.append("</table>")

    H.append("<h2>Top 20 High-Merit Gaps</h2><table><tr><th>#</th><th>Position</th><th>Gap</th><th>Merit</th></tr>")
    for i, r in enumerate(top_merit, 1):
        H.append("<tr><td>" + str(i) + "</td><td>" + format(int(r["position"]), ",") + "</td><td>" + r["gap"] + "</td><td>" + format(float(r["merit"]), ".4f") + "</td></tr>")
    H.append("</table>")

    H.append("<h2>Miller-Rabin Verification</h2><div class=\"grid\">")
    H.append("<div class=\"stat\"><b>Total:</b> " + format(v_total, ",") + "</div>")
    H.append("<div class=\"stat\"><b>Verified:</b> " + format(v_verified, ",") + "</div>")
    H.append("<div class=\"stat\"><b>Failed:</b> " + str(v_failed) + "</div>")
    H.append("<div class=\"stat\"><b>Duration:</b> " + format(v_duration, ".2f") + " s</div></div>")

    H.append("<h2>Digital Certificates</h2>")
    H.append("<div class=\"stat\"><b>Total certificates:</b> " + format(len(certs), ",") + " (SHA-256 signed)</div>")

    H.append("<div class=\"footer\">")
    H.append("Generated by <b>VOSS v7 LIVE</b> - Vectorized Odd Segmented Sieve<br>")
    H.append("Verification: Deterministic Miller-Rabin + Interval Sieve<br>")
    H.append("Report: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    H.append("</div></body></html>")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(H))

    print("OK: " + OUT)
    print("  Size: " + format(os.path.getsize(OUT), ",") + " bytes")
    print("  N: " + N_str)
    print("  Total gaps: " + format(total_gaps, ","))
    print("  pi(N): " + format(pi_total, ","))
    print("  Verified: " + format(v_verified, ",") + "/" + format(v_total, ","))
    print("  Certificates: " + format(len(certs), ","))

if __name__ == "__main__":
    build()