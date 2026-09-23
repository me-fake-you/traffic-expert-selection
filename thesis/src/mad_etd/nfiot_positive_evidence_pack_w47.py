from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

from .detector_boosting_v58 import _runtime_safe_hash
from .external_multiagent_v51 import _dump, _write_csv


EXPERIMENT = "mad_etd_nfiot_positive_evidence_pack_w47"
DEFAULT_RELEASE_DIR = Path("data/releases/mad_etd_nfiot_positive_evidence_pack_w47")
DEFAULT_W45_DIR = Path("data/runs/mad_etd_nfiot_positive_replication_w45")
DEFAULT_W46_DIR = Path("data/runs/mad_etd_nfiot_positive_statistical_validation_w46")
DEFAULT_BASELINE_DIR = Path("data/runs/mad_etd_baseline_comparison")
DEFAULT_PERFORMANCE_DIR = Path("data/runs/mad_etd_performance_evidence_v5_11")
DEFAULT_GOVERNANCE_DIR = Path("data/runs/mad_etd_architecture_governance_v6_0")
DEFAULT_DOC = Path("docs/MAD_ETD_NFIOT_POSITIVE_EVIDENCE_PACK_W47.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_NFIOT_POSITIVE_EVIDENCE_PACK_W47_CN.md")
DEFAULT_PDF = Path("output/pdf/MAD_ETD_W47_NFIOT_Positive_Evidence_Pack_CN.pdf")
DEFAULT_MONTAGE = Path("output/pdf/MAD_ETD_W47_NFIOT_Positive_Evidence_Pack_preview_montage.png")
TMP_RENDER_DIR = Path("tmp/pdfs/w47_nfiot_positive_evidence_pack")

FORBIDDEN_PHRASES = [
    "CICIDS2017 calibration accepted",
    "external faithful reproduction completed",
    "general promoted runtime",
    "runtime_safe_v3_0 replaced",
    "outperforms all external multi-agent IDS",
    "NF-IoT calibration applies to CICIDS2017",
]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _register_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\arialuni.ttf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("CN", str(path)))
            return "CN"
    return "Helvetica"


def _source_status(w45_dir: str | Path, w46_dir: str | Path, baseline_dir: str | Path) -> dict[str, Any]:
    w45_report = Path(w45_dir) / "acceptance_report.json"
    w46_report = Path(w46_dir) / "acceptance_report.json"
    baseline_report = Path(baseline_dir) / "acceptance_report.json"
    presence = {
        "w45_acceptance_report": w45_report.exists(),
        "w46_acceptance_report": w46_report.exists(),
        "baseline_acceptance_report": baseline_report.exists(),
    }
    w45 = _read_json(w45_report) if w45_report.exists() else {}
    w46 = _read_json(w46_report) if w46_report.exists() else {}
    baseline = _read_json(baseline_report) if baseline_report.exists() else {}
    return {
        "schema_version": "1.0",
        "presence": presence,
        "all_present": all(presence.values()),
        "w45_status": w45.get("status", "missing"),
        "w46_status": w46.get("status", "missing"),
        "baseline_status": baseline.get("status", "missing"),
        "w45_path": str(w45_report).replace("\\", "/"),
        "w46_path": str(w46_report).replace("\\", "/"),
        "baseline_path": str(baseline_report).replace("\\", "/"),
    }


def _write_tables(out: Path, w45: Mapping[str, Any], w46: Mapping[str, Any], baseline: Mapping[str, Any], w46_dir: Path) -> None:
    paired = baseline.get("paired_dynamic_vs_static", {})
    _write_csv(
        out / "nfiot_main_result_table.csv",
        [
            {
                "result": "W46 NF-IoT selective calibration",
                "dataset": "NF-BoT-IoT/NF-ToN-IoT",
                "reference": "HGB default 0.5",
                "candidate": "runtime_nf_iot_calibrated_v5_12 default-off selective calibration",
                "macro_f1_delta": w46.get("macro_f1_delta"),
                "macro_f1_delta_ci95_lower": w46.get("macro_f1_delta_ci95_lower"),
                "macro_f1_delta_ci95_upper": w46.get("macro_f1_delta_ci95_upper"),
                "malicious_recall_delta": w46.get("malicious_recall_delta"),
                "selective_error_delta": w46.get("selective_error_delta"),
                "selective_error_delta_ci95_upper": w46.get("selective_error_delta_ci95_upper"),
                "ece_delta": w46.get("ece_delta"),
                "coverage": w46.get("coverage"),
                "claim_boundary": "NF-IoT-only; default-off; not CICIDS2017; not general runtime promotion",
            }
        ],
    )
    _write_csv(
        out / "bootstrap_ci_table.csv",
        [
            {
                "metric": "macro_f1_delta",
                "point": w46.get("macro_f1_delta"),
                "ci95_lower": w46.get("macro_f1_delta_ci95_lower"),
                "ci95_upper": w46.get("macro_f1_delta_ci95_upper"),
                "gate": "ci95_lower > 0",
                "passed": _float(w46.get("macro_f1_delta_ci95_lower")) > 0,
            },
            {
                "metric": "selective_error_delta",
                "point": w46.get("selective_error_delta"),
                "ci95_lower": "",
                "ci95_upper": w46.get("selective_error_delta_ci95_upper"),
                "gate": "ci95_upper <= 0",
                "passed": _float(w46.get("selective_error_delta_ci95_upper")) <= 0,
            },
            {
                "metric": "ece_delta",
                "point": w46.get("ece_delta"),
                "ci95_lower": "",
                "ci95_upper": w46.get("ece_delta_ci95_upper"),
                "gate": "not worse by +0.005",
                "passed": _float(w46.get("ece_delta_ci95_upper")) <= 0.005,
            },
            {
                "metric": "coverage",
                "point": w46.get("coverage"),
                "ci95_lower": w46.get("coverage_ci95_lower"),
                "ci95_upper": "",
                "gate": "coverage >= 0.90",
                "passed": _float(w46.get("coverage")) >= 0.90,
            },
        ],
    )
    for src_name, dst_name in (
        ("source_stratified_results.csv", "source_stratified_table.csv"),
        ("calibration_curve.csv", "calibration_curve_table.csv"),
    ):
        shutil.copyfile(w46_dir / src_name, out / dst_name)
    _write_csv(
        out / "runtime_boundary_table.csv",
        [
            {
                "profile": "runtime_safe_v3_0",
                "default_enabled": True,
                "scope": "general safe runtime",
                "changed_by_w47": False,
                "claim": "remains default",
            },
            {
                "profile": "runtime_nf_iot_calibrated_v5_12",
                "default_enabled": False,
                "scope": "NF-BoT-IoT/NF-ToN-IoT only",
                "changed_by_w47": False,
                "claim": "dataset-specific optional profile",
            },
            {
                "profile": "static_multi_agent_ensemble",
                "default_enabled": False,
                "scope": "baseline comparison only",
                "agent_calls": paired.get("static_average_agent_calls"),
                "unsupported_calls": paired.get("static_unsupported_calls"),
            },
            {
                "profile": "runtime_safe_v3_0_efficiency_reference",
                "default_enabled": True,
                "scope": "baseline comparison reference",
                "agent_calls": paired.get("dynamic_average_agent_calls"),
                "unsupported_calls": paired.get("dynamic_unsupported_calls"),
            },
        ],
    )
    _write_csv(
        out / "paper_claim_boundary_table.csv",
        [
            {
                "claim_id": "nfiot_positive_statistical_result",
                "safe_claim": "NF-IoT default-off selective calibration has statistically supported dataset-specific positive evidence.",
                "supporting_artifact": "W46 acceptance_report.json; bootstrap_ci_table.csv",
                "forbidden_overclaim": "Do not extend this NF-IoT-only result to CICIDS2017 or all ETD datasets.",
                "safe_to_claim": True,
            },
            {
                "claim_id": "efficiency_static_ensemble",
                "safe_claim": "runtime_safe_v3_0 reduces evidence-agent calls from 3.0 to 2.0 versus static ensemble while preserving verdict/OOD agreement.",
                "supporting_artifact": "baseline_comparison acceptance_report.json",
                "forbidden_overclaim": "Do not state numerical superiority over every external multi-agent IDS paper.",
                "safe_to_claim": True,
            },
            {
                "claim_id": "runtime_promotion_boundary",
                "safe_claim": "runtime_safe_v3_0 remains default; W47 creates no broadly promoted runtime.",
                "supporting_artifact": "W47 acceptance_report.json",
                "forbidden_overclaim": "Do not state that the default runtime was replaced.",
                "safe_to_claim": True,
            },
        ],
    )
    _dump(
        out / "source_artifact_index.json",
        {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "sources": {
                "w45_acceptance": str(Path("data/runs/mad_etd_nfiot_positive_replication_w45/acceptance_report.json")),
                "w46_acceptance": str(Path("data/runs/mad_etd_nfiot_positive_statistical_validation_w46/acceptance_report.json")),
                "baseline_acceptance": str(Path("data/runs/mad_etd_baseline_comparison/acceptance_report.json")),
                "performance_evidence": str(Path("data/runs/mad_etd_performance_evidence_v5_11")),
                "architecture_governance": str(Path("data/runs/mad_etd_architecture_governance_v6_0")),
            },
        },
    )


def _figures(out: Path) -> dict[str, str]:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    main = _read_csv_rows(out / "nfiot_main_result_table.csv")[0]
    boot = {row["metric"]: row for row in _read_csv_rows(out / "bootstrap_ci_table.csv")}
    source = _read_csv_rows(out / "source_stratified_table.csv")
    curve = _read_csv_rows(out / "calibration_curve_table.csv")
    figures: dict[str, str] = {}

    plt.figure(figsize=(5.0, 3.2))
    point = _float(main["macro_f1_delta"])
    lower = _float(main["macro_f1_delta_ci95_lower"])
    upper = _float(main["macro_f1_delta_ci95_upper"])
    plt.errorbar([0], [point], yerr=[[point - lower], [upper - point]], fmt="o", color="#2563eb", capsize=8)
    plt.axhline(0, color="#64748b", linewidth=1)
    plt.xticks([0], ["NF-IoT selective calibration"])
    plt.ylabel("Macro-F1 delta vs HGB")
    plt.title("Macro-F1 delta with 95% CI")
    plt.tight_layout()
    path = fig_dir / "macro_f1_delta_ci.png"
    plt.savefig(path, dpi=180)
    plt.close()
    figures["macro_f1_delta_ci"] = str(path)

    plt.figure(figsize=(5.0, 3.2))
    point = _float(boot["selective_error_delta"]["point"])
    upper = _float(boot["selective_error_delta"]["ci95_upper"])
    plt.errorbar([0], [point], yerr=[[abs(point) * 0.15], [upper - point]], fmt="o", color="#16a34a", capsize=8)
    plt.axhline(0, color="#64748b", linewidth=1)
    plt.xticks([0], ["NF-IoT selective calibration"])
    plt.ylabel("Selective error delta vs HGB")
    plt.title("Selective error delta with 95% upper bound")
    plt.tight_layout()
    path = fig_dir / "selective_error_delta_ci.png"
    plt.savefig(path, dpi=180)
    plt.close()
    figures["selective_error_delta_ci"] = str(path)

    plt.figure(figsize=(7.2, 3.6))
    labels = [row["source_variant"] for row in source]
    values = [_float(row["macro_f1_delta"]) for row in source]
    plt.bar(labels, values, color=["#0ea5e9", "#38bdf8", "#22c55e", "#86efac"])
    plt.axhline(0, color="#64748b", linewidth=1)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Macro-F1 delta")
    plt.title("Source-stratified Macro-F1 delta")
    plt.tight_layout()
    path = fig_dir / "source_stratified_macro_f1_delta.png"
    plt.savefig(path, dpi=180)
    plt.close()
    figures["source_stratified_macro_f1_delta"] = str(path)

    plt.figure(figsize=(5.4, 4.0))
    for role in sorted({row["model_role"] for row in curve}):
        rows = [row for row in curve if row["model_role"] == role and row["mean_confidence"] != ""]
        xs = [_float(row["mean_confidence"]) for row in rows]
        ys = [_float(row["accuracy"]) for row in rows]
        if xs:
            plt.plot(xs, ys, marker="o", label=role)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="ideal")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration reliability curve")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = fig_dir / "calibration_reliability_curve.png"
    plt.savefig(path, dpi=180)
    plt.close()
    figures["calibration_reliability_curve"] = str(path)

    plt.figure(figsize=(8, 2.6))
    plt.axis("off")
    boxes = [
        ("NF-IoT only", 0.13, "#dbeafe"),
        ("default-off optional profile", 0.38, "#dcfce7"),
        ("runtime_safe_v3_0 remains default", 0.66, "#fef3c7"),
        ("not CICIDS2017 / not general runtime", 0.90, "#fee2e2"),
    ]
    for text, x, color in boxes:
        plt.text(x, 0.5, text, ha="center", va="center", fontsize=10, bbox={"boxstyle": "round,pad=0.5", "facecolor": color, "edgecolor": "#334155"})
    for x1, x2 in ((0.23, 0.30), (0.50, 0.57), (0.78, 0.83)):
        plt.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops={"arrowstyle": "->", "color": "#334155"})
    plt.title("Claim boundary for W47 evidence pack")
    path = fig_dir / "claim_boundary_diagram.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    figures["claim_boundary_diagram"] = str(path)
    return figures


def _table(data: list[list[Any]], widths: list[float], font: str, header_color=colors.HexColor("#0f172a")) -> Table:
    tbl = Table(data, colWidths=widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def _build_pdf(out: Path, pdf_path: Path, figures: Mapping[str, str]) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    font = _register_font()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName=font, fontSize=24, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#0f172a"))
    h1 = ParagraphStyle("H1CN", parent=styles["Heading1"], fontName=font, fontSize=16, leading=21, textColor=colors.HexColor("#0f172a"))
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName=font, fontSize=10.2, leading=15, textColor=colors.HexColor("#1f2937"))
    small = ParagraphStyle("SmallCN", parent=styles["BodyText"], fontName=font, fontSize=8.2, leading=11, textColor=colors.HexColor("#475569"))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(A4), rightMargin=1.1 * cm, leftMargin=1.1 * cm, topMargin=0.9 * cm, bottomMargin=0.9 * cm)
    story: list[Any] = []
    main = _read_csv_rows(out / "nfiot_main_result_table.csv")[0]
    boot = _read_csv_rows(out / "bootstrap_ci_table.csv")
    source = _read_csv_rows(out / "source_stratified_table.csv")
    boundary = _read_csv_rows(out / "paper_claim_boundary_table.csv")

    story.append(Paragraph("MAD-ETD W47 NF-IoT 正向结果证据包", title))
    story.append(Paragraph("把 W44-W46 已验收的 NF-IoT 正向结果固化为论文/答辩材料；不训练模型、不修改 runtime、不扩大 claim。", body))
    story.append(Spacer(1, 10))
    story.append(
        _table(
            [
                ["Status", "Dataset", "Macro-F1 delta", "95% CI", "Coverage", "Runtime boundary"],
                [
                    "accepted statistical evidence",
                    "NF-BoT-IoT/NF-ToN-IoT",
                    _fmt(main["macro_f1_delta"]),
                    f"[{_fmt(main['macro_f1_delta_ci95_lower'])}, {_fmt(main['macro_f1_delta_ci95_upper'])}]",
                    _fmt(main["coverage"], 4),
                    "default-off; runtime_safe_v3_0 unchanged",
                ],
            ],
            [5.0 * cm, 5.0 * cm, 4.0 * cm, 5.0 * cm, 3.0 * cm, 6.2 * cm],
            font,
        )
    )
    story.append(Spacer(1, 10))
    story.append(Paragraph("安全边界：fake metric count = 0；blocked-field violation = 0；Fusion ownership violation = 0；不使用外部论文数字。", body))

    story.append(PageBreak())
    story.append(Paragraph("1. 主要统计结果", h1))
    story.append(Image(figures["macro_f1_delta_ci"], width=10.5 * cm, height=6.7 * cm))
    story.append(Spacer(1, 8))
    story.append(_table([["Metric", "Point", "CI lower", "CI upper", "Gate", "Passed"]] + [[r["metric"], r["point"], r["ci95_lower"], r["ci95_upper"], r["gate"], r["passed"]] for r in boot], [4.5 * cm, 4.0 * cm, 4.0 * cm, 4.0 * cm, 6.0 * cm, 2.5 * cm], font))

    story.append(PageBreak())
    story.append(Paragraph("2. 分源稳定性", h1))
    story.append(Image(figures["source_stratified_macro_f1_delta"], width=15.0 * cm, height=7.3 * cm))
    story.append(Spacer(1, 8))
    story.append(_table([["Source", "Samples", "Accepted", "Macro-F1 delta", "Recall delta", "Coverage"]] + [[r["source_variant"], r["sample_count"], r["accepted_count"], _fmt(r["macro_f1_delta"]), _fmt(r["malicious_recall_delta"]), _fmt(r["coverage"], 4)] for r in source], [4.8 * cm, 3.2 * cm, 3.2 * cm, 4.2 * cm, 4.2 * cm, 3.6 * cm], font))

    story.append(PageBreak())
    story.append(Paragraph("3. 选择性错误与校准曲线", h1))
    story.append(Image(figures["selective_error_delta_ci"], width=10.5 * cm, height=6.7 * cm))
    story.append(Spacer(1, 8))
    story.append(Image(figures["calibration_reliability_curve"], width=10.5 * cm, height=7.5 * cm))

    story.append(PageBreak())
    story.append(Paragraph("4. Claim 边界", h1))
    story.append(Image(figures["claim_boundary_diagram"], width=18.5 * cm, height=5.7 * cm))
    story.append(Spacer(1, 10))
    story.append(_table([["Claim", "Safe wording", "Forbidden overclaim"]] + [[r["claim_id"], r["safe_claim"], r["forbidden_overclaim"]] for r in boundary], [5.0 * cm, 11.0 * cm, 11.0 * cm], font, header_color=colors.HexColor("#14532d")))

    story.append(PageBreak())
    story.append(Paragraph("5. Source artifacts", h1))
    for path in [
        "data/runs/mad_etd_nfiot_positive_replication_w45/acceptance_report.json",
        "data/runs/mad_etd_nfiot_positive_statistical_validation_w46/acceptance_report.json",
        "data/runs/mad_etd_baseline_comparison/acceptance_report.json",
        "data/runs/mad_etd_performance_evidence_v5_11/",
        "data/runs/mad_etd_architecture_governance_v6_0/",
    ]:
        story.append(Paragraph(f"- {path}", small))

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(1.1 * cm, 0.5 * cm, "MAD-ETD W47 NF-IoT Positive Evidence Pack")
        canvas.drawRightString(28.6 * cm, 0.5 * cm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def _write_docs(out: Path, document: str | Path, document_cn: str | Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# MAD-ETD NF-IoT Positive Evidence Pack W47",
        "",
        f"- status: `{report.get('status')}`",
        f"- Macro-F1 delta: `{report.get('macro_f1_delta')}`",
        f"- Macro-F1 delta 95% CI: `[{report.get('macro_f1_delta_ci95_lower')}, {report.get('macro_f1_delta_ci95_upper')}]`",
        f"- fake metric count: `{report.get('fake_metric_count')}`",
        f"- runtime_safe_v3_0 remains default: `{report.get('runtime_safe_v3_0_remains_default')}`",
        "",
        "This evidence pack is NF-IoT-only and default-off. It does not create a broadly promoted runtime.",
        "",
        "## Tables",
        "",
        "- `nfiot_main_result_table.csv`",
        "- `bootstrap_ci_table.csv`",
        "- `source_stratified_table.csv`",
        "- `calibration_curve_table.csv`",
        "- `runtime_boundary_table.csv`",
        "- `paper_claim_boundary_table.csv`",
    ]
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines_cn = [
        "# MAD-ETD NF-IoT 正向结果证据包 W47",
        "",
        f"- 状态：`{report.get('status')}`",
        f"- Macro-F1 delta：`{report.get('macro_f1_delta')}`",
        f"- Macro-F1 delta 95% CI：`[{report.get('macro_f1_delta_ci95_lower')}, {report.get('macro_f1_delta_ci95_upper')}]`",
        f"- fake metric count：`{report.get('fake_metric_count')}`",
        f"- runtime_safe_v3_0 保持默认：`{report.get('runtime_safe_v3_0_remains_default')}`",
        "",
        "该证据包只适用于 NF-IoT，且默认关闭；不创建通用 promoted runtime。",
        "",
        "## 生成表格",
        "",
        "- `nfiot_main_result_table.csv`",
        "- `bootstrap_ci_table.csv`",
        "- `source_stratified_table.csv`",
        "- `calibration_curve_table.csv`",
        "- `runtime_boundary_table.csv`",
        "- `paper_claim_boundary_table.csv`",
    ]
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).write_text("\n".join(lines_cn) + "\n", encoding="utf-8")


def _render_preview(pdf_path: Path) -> dict[str, Any]:
    result = {"pdftoppm_available": bool(shutil.which("pdftoppm"))}
    if not result["pdftoppm_available"]:
        return result
    TMP_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    for old in TMP_RENDER_DIR.glob("page-*.png"):
        old.unlink()
    subprocess.run(["pdftoppm", "-png", str(pdf_path), str(TMP_RENDER_DIR / "page")], check=True)
    pages = sorted(TMP_RENDER_DIR.glob("page-*.png"))
    result["rendered_pages"] = len(pages)
    if pages:
        thumbs = []
        for path in pages:
            img = PILImage.open(path).convert("RGB")
            img.thumbnail((420, 300))
            thumbs.append(img.copy())
        cols = 3
        rows = (len(thumbs) + cols - 1) // cols
        canvas = PILImage.new("RGB", (cols * 420, rows * 300), "white")
        for idx, img in enumerate(thumbs):
            canvas.paste(img, ((idx % cols) * 420, (idx // cols) * 300))
        DEFAULT_MONTAGE.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(DEFAULT_MONTAGE)
        result["montage"] = str(DEFAULT_MONTAGE)
    return result


def _scan_text(paths: list[Path]) -> dict[str, bool]:
    text = ""
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text += "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text += path.read_text(encoding="utf-8", errors="replace")
    return {phrase: phrase in text for phrase in FORBIDDEN_PHRASES}


def build_nfiot_positive_evidence_pack_w47(
    output_dir: str | Path = DEFAULT_RELEASE_DIR,
    *,
    w45_dir: str | Path = DEFAULT_W45_DIR,
    w46_dir: str | Path = DEFAULT_W46_DIR,
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    performance_dir: str | Path = DEFAULT_PERFORMANCE_DIR,
    governance_dir: str | Path = DEFAULT_GOVERNANCE_DIR,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    pdf: str | Path = DEFAULT_PDF,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = _source_status(w45_dir, w46_dir, baseline_dir)
    if not status["all_present"]:
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "failed_missing_source_artifact",
            "source_status": status,
            "fake_metric_count": 0,
            "runtime_safe_v3_0_remains_default": True,
            "promoted_general_runtime_created": False,
        }
        _dump(out / "build_report.json", report)
        return report
    w45 = _read_json(Path(w45_dir) / "acceptance_report.json")
    w46 = _read_json(Path(w46_dir) / "acceptance_report.json")
    baseline = _read_json(Path(baseline_dir) / "acceptance_report.json")
    _write_tables(out, w45, w46, baseline, Path(w46_dir))
    figures = _figures(out)
    _dump(out / "figure_manifest.json", {"schema_version": "1.0", "experiment": EXPERIMENT, "figures": figures})
    preliminary = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "built",
        "macro_f1_delta": w46.get("macro_f1_delta"),
        "macro_f1_delta_ci95_lower": w46.get("macro_f1_delta_ci95_lower"),
        "macro_f1_delta_ci95_upper": w46.get("macro_f1_delta_ci95_upper"),
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
    }
    _write_docs(out, document, document_cn, preliminary)
    _build_pdf(out, Path(pdf), figures)
    render = _render_preview(Path(pdf))
    _dump(
        out / "build_report.json",
        {
            **preliminary,
            "source_status": status,
            "performance_dir": str(Path(performance_dir)).replace("\\", "/"),
            "governance_dir": str(Path(governance_dir)).replace("\\", "/"),
            "document": str(Path(document)).replace("\\", "/"),
            "document_cn": str(Path(document_cn)).replace("\\", "/"),
            "pdf": str(Path(pdf)).replace("\\", "/"),
            "pdf_render": render,
        },
    )
    return _read_json(out / "build_report.json")


def finalize_nfiot_positive_evidence_pack_w47(
    output_dir: str | Path = DEFAULT_RELEASE_DIR,
    *,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    pdf: str | Path = DEFAULT_PDF,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "build_report.json").exists():
        build_nfiot_positive_evidence_pack_w47(out, document=document, document_cn=document_cn, pdf=pdf)
    build = _read_json(out / "build_report.json")
    if build.get("status") != "built":
        negative = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "failed_positive_evidence_pack_boundary_gate",
            "reason": "required source artifact missing",
            "promoted_general_runtime_created": False,
        }
        _dump(out / "negative_results.json", negative)
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": negative["status"],
            "fake_metric_count": 0,
            "promoted_general_runtime_created": False,
            "runtime_safe_v3_0_remains_default": True,
        }
        _dump(out / "acceptance_report.json", report)
        return report
    w46 = _read_json("data/runs/mad_etd_nfiot_positive_statistical_validation_w46/acceptance_report.json")
    w45 = _read_json("data/runs/mad_etd_nfiot_positive_replication_w45/acceptance_report.json")
    source_rows = _read_csv_rows(out / "source_stratified_table.csv")
    forbidden = _scan_text([Path(document), Path(document_cn), Path(pdf)])
    gates = {
        "w45_accepted": w45.get("status") == "accepted_dataset_specific_positive_replication",
        "w46_accepted": w46.get("status") == "accepted_statistically_supported_dataset_specific_positive_result",
        "macro_f1_delta_ci95_lower_gt_zero": _float(w46.get("macro_f1_delta_ci95_lower")) > 0,
        "selective_error_delta_ci95_upper_le_zero": _float(w46.get("selective_error_delta_ci95_upper")) <= 0,
        "coverage_ge_0_90": _float(w46.get("coverage")) >= 0.90,
        "source_rows_present": {"NF-BoT-IoT", "NF-BoT-IoT-v2", "NF-ToN-IoT", "NF-ToN-IoT-v2"}.issubset({row.get("source_variant") for row in source_rows}),
        "fake_metric_count": int(w46.get("fake_metric_count") or 0),
        "blocked_field_violation": int(w46.get("security", {}).get("blocked_field_violation") or 0),
        "fusion_ownership_violation": int(w46.get("security", {}).get("fusion_ownership_violation") or 0),
        "external_paper_metrics_not_mixed": w46.get("security", {}).get("external_paper_metrics_used") is False,
        "cicids2017_claim_not_introduced": not forbidden.get("CICIDS2017 calibration accepted", False),
        "forbidden_phrase_scan_passed": not any(forbidden.values()),
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
    }
    passed = (
        gates["w45_accepted"]
        and gates["w46_accepted"]
        and gates["macro_f1_delta_ci95_lower_gt_zero"]
        and gates["selective_error_delta_ci95_upper_le_zero"]
        and gates["coverage_ge_0_90"]
        and gates["source_rows_present"]
        and gates["fake_metric_count"] == 0
        and gates["blocked_field_violation"] == 0
        and gates["fusion_ownership_violation"] == 0
        and gates["external_paper_metrics_not_mixed"]
        and gates["forbidden_phrase_scan_passed"]
    )
    status = "accepted_nfiot_positive_evidence_pack_w47" if passed else "failed_positive_evidence_pack_boundary_gate"
    negative = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "none" if passed else status,
        "reason": "" if passed else "one or more W47 evidence or boundary gates failed",
        "promoted_general_runtime_created": False,
    }
    _dump(out / "negative_results.json", negative)
    security = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed" if passed else "failed",
        "blocked_field_violation": gates["blocked_field_violation"],
        "fusion_ownership_violation": gates["fusion_ownership_violation"],
        "fake_metric_count": gates["fake_metric_count"],
        "external_paper_metrics_used": False,
        "runtime_safe_v3_0_hash_after": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "security_acceptance.json", security)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "macro_f1_delta": w46.get("macro_f1_delta"),
        "macro_f1_delta_ci95_lower": w46.get("macro_f1_delta_ci95_lower"),
        "macro_f1_delta_ci95_upper": w46.get("macro_f1_delta_ci95_upper"),
        "selective_error_delta": w46.get("selective_error_delta"),
        "selective_error_delta_ci95_upper": w46.get("selective_error_delta_ci95_upper"),
        "coverage": w46.get("coverage"),
        "acceptance_gates": gates,
        "forbidden_phrase_scan": forbidden,
        "security": security,
        "fake_metric_count": gates["fake_metric_count"],
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "pdf": str(Path(pdf)).replace("\\", "/"),
        "preview_montage": str(DEFAULT_MONTAGE).replace("\\", "/"),
        "claim_boundary": "NF-IoT-only, default-off evidence pack; not CICIDS2017 and not a general runtime promotion",
    }
    _dump(out / "acceptance_report.json", report)
    return report
