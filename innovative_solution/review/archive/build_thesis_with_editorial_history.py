"""Build the reviewed manuscript from saved experimental evidence, not hand-entered scores."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
RESULTS = PACKAGE / "outputs" / "review"
TITLE = "基于机器学习的TFT-LCD多工序质量特性预测研究"
DEST = HERE / f"{TITLE}_审稿修订版.docx"
KEYS = dict(kusiak=3, competition=2, qu=1, friedman=7, xgboost=10,
            rf=8, et=9, stacking=12, ridge=11, tabular=13, tabpfn=17,
            missing=6, drift=14, tashman=15, conformal=16)
VENUES = {
    3: "International Journal of Production Research", 7: "The Annals of Statistics",
    8: "Machine Learning", 9: "Machine Learning", 11: "Technometrics",
    12: "Neural Networks", 14: "ACM Computing Surveys",
    15: "International Journal of Forecasting", 16: "Foundations and Trends in Machine Learning",
    17: "Nature",
}


def load_evidence():
    manifest = json.loads((RESULTS / "review_manifest.json").read_text(encoding="utf-8"))
    with (RESULTS / "nested_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    score = {r["model"]: r for r in rows}
    tail, a = manifest["tail_metrics"], manifest["retrospective_A_metrics"]
    if a is None:
        raise ValueError("This manuscript describes the recorded A retrospective evaluation; no A score is available.")
    values = {"nested_mse": score["nested_selected"]["mse"],
              "raw_mse": score["fixed_raw_d2"]["mse"],
              "full_mse": score["fixed_full_d2"]["mse"],
              "selected": manifest["selected_candidate"],
              "coverage": f"{manifest['observed_tail_coverage']:.0%}",
              "n_dev": manifest["n_development"], "n_cal": manifest["n_calibration"],
              "n_tail": manifest["n_tail"], "radius": manifest["empirical_radius"],
              "covered_n": round(manifest["observed_tail_coverage"] * manifest["n_tail"])}
    for prefix, metrics in [("tail", tail), ("A", a)]:
        values.update({f"{prefix}_{k}": v for k, v in metrics.items()})
    for key in list(values):
        if key.endswith(("mse", "mae", "r2")) or key == "radius":
            values[key] = f"{float(values[key]):.6f}"
    labels = {"mean_baseline": "训练均值", "fixed_raw_d2": "固定原始视图 深度2",
              "fixed_full_d2": "固定完整视图 深度2", "nested_selected": "内层选择流程"}
    lines = ["表5-1 同一300条外层记录的合并指标", "",
             "| 方法 | MSE | RMSE | MAE | R² |", "|---|---:|---:|---:|---:|"]
    for name in ["mean_baseline", "fixed_raw_d2", "fixed_full_d2", "nested_selected"]:
        row = score[name]
        lines.append("| " + labels[name] + " | " + " | ".join(f"{float(row[k]):.6f}" for k in ["mse", "rmse", "mae", "r2"]) + " |")
    values["nested_table"] = "\n".join(lines)
    delta = 100 * (float(score["fixed_raw_d2"]["mse"]) - float(score["fixed_full_d2"]["mse"])) / float(score["fixed_raw_d2"]["mse"])
    values["nested_interpretation"] = (
        f"固定完整视图的合并MSE比固定原始视图低约{delta:.2f}%，这是本次相同样本比较中的描述性差异。"
        "内层选择流程反而高于两种固定对照，说明增加选择步骤并未在这些外层窗口中带来稳定收益。"
        "仅有三个相关历史窗口，本文不据此宣称工序分支显著有效；也不在查看外层分数后将固定完整视图重新命名为确认最优。"
        "正式全量重训仍遵守前段内层选择规则，保留该规则产生的full_d3配置。")
    text = (HERE / "revised_manuscript.md").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    assert "{{" not in text, "Unfilled result placeholder"
    order = list(dict.fromkeys(re.findall(r"\[@([a-z]+)\]", text)))
    refs = json.loads((PACKAGE / "review" / "references_verified.json").read_text(encoding="utf-8"))["references"]
    refs = {r["id"]: r for r in refs}
    entries = []
    for i, key in enumerate(order, 1):
        source = refs[KEYS[key]]
        entries.append({"number": i, "key": key, "original_id": source["id"],
                        "text": reference_text(source), "verification": source["verification"]})
        text = text.replace(f"[@{key}]", f"[{i}]")
    return text, entries


def reference_text(r):
    rid = r["id"]
    if rid == 1:
        return "屈方磊. 基于集成学习的智能制造质量预测研究[D]. 中北大学, 2024."
    if rid == 2:
        return ("阿里云天池. 智能制造质量预测赛题[EB/OL]. " + r["url"] +
                "（赛题内容依据保存的资料；在线正文访问未核实）.")
    names = r["authors"]
    authors = ", ".join(names[:3]) + (", et al" if len(names) > 3 else "")
    title = r["title"]
    year = r.get("year", r.get("year_print"))
    if rid == 6:
        return f"{authors}. {title}[M]. Wiley, {year}. DOI: {r['doi']}."
    if rid == 10:
        body = "[C]//Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016: 785-794"
    elif rid == 13:
        body = "[C]//Advances in Neural Information Processing Systems 35. 2022: 507-520"
    else:
        body = f"[J]. {VENUES[rid]}, {year}, {r['volume']}({r['issue']}): {r['pages']}"
    return f"{authors}. {title}{body}. DOI: {r['doi']}."


def font(run, size=12, bold=False, chinese="宋体"):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    pr = run._r.get_or_add_rPr()
    fonts = pr.rFonts
    fonts.set(qn("w:eastAsia"), chinese)


def paragraph(doc, text="", size=12, indent=True, center=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Pt(24) if indent else Pt(0)
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.widow_control = True
    # Numeric citations are ordinary editable text and stay linked to the auto-numbered bibliography.
    font(p.add_run(text), size=size)
    return p


def field(par, instruction, placeholder="1"):
    for typ, text in [("begin", None), (None, instruction), ("separate", None), ("text", placeholder), ("end", None)]:
        r = par.add_run()
        font(r, size=10)
        if typ == "text":
            r.text = text
        elif typ is None:
            elem = OxmlElement("w:instrText"); elem.set(qn("xml:space"), "preserve"); elem.text = text
            r._r.append(elem)
        else:
            elem = OxmlElement("w:fldChar"); elem.set(qn("w:fldCharType"), typ); r._r.append(elem)


def heading(doc, text, level):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(12 if level > 1 else 0)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.page_break_before = True
    font(p.add_run(text), size=16 if level == 1 else 13, bold=True, chinese="黑体")
    return p


def table(doc, lines):
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r"[-:]+", c) for c in r)]
    n = len(rows[0])
    tab = doc.add_table(rows=len(rows), cols=n)
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    tab.autofit = False
    if n == 5 and rows[0][0] == "数据":
        widths = [2.4, 2.2, 2.2, 3.2, 5.4]
    elif n == 5:
        widths = [5.0, 2.6, 2.6, 2.6, 2.6]
    elif n == 4:
        widths = [2.8, 4.0, 4.2, 4.4]
    else:
        widths = [15.4/n] * n
    for col, width in zip(tab.columns, widths):
        col.width = Cm(width)
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "bottom", "insideH"]:
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6" if edge != "insideH" else "3"); e.set(qn("w:color"), "777777"); borders.append(e)
    tab._tbl.tblPr.append(borders)
    for i, row in enumerate(rows):
        trpr = tab.rows[i]._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for j, value in enumerate(row):
            cell = tab.cell(i, j); cell.width = Cm(widths[j]); cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_before = Pt(5); p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = i < len(rows)-1
            p.paragraph_format.keep_together = True
            font(p.add_run(value), size=10, bold=i == 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


# Equations use structural OMML fractions, scripts and summation limits; all remain editable.
def mr(value):
    r = OxmlElement("m:r"); t = OxmlElement("m:t"); t.text = str(value); r.append(t); return r


def expr(value):
    return [mr(value)] if isinstance(value, (str, int)) else (value if isinstance(value, list) else [value])


def wrap(tag, value):
    e = OxmlElement("m:" + tag); e.extend(expr(value)); return e


def frac(a, b):
    e = OxmlElement("m:f"); e.append(wrap("num", a)); e.append(wrap("den", b)); return e


def sub(a, b):
    e = OxmlElement("m:sSub"); e.append(wrap("e", a)); e.append(wrap("sub", b)); return e


def sup(a, b):
    e = OxmlElement("m:sSup"); e.append(wrap("e", a)); e.append(wrap("sup", b)); return e


def hat(a):
    e = OxmlElement("m:acc"); pr = OxmlElement("m:accPr"); ch = OxmlElement("m:chr"); ch.set(qn("m:val"), "̂"); pr.append(ch); e.append(pr); e.append(wrap("e", a)); return e


def summation(body, lower="i=1", upper="n"):
    e = OxmlElement("m:nary"); pr = OxmlElement("m:naryPr"); ch = OxmlElement("m:chr"); ch.set(qn("m:val"), "∑"); pr.append(ch)
    e.append(pr); e.append(wrap("sub", lower)); e.append(wrap("sup", upper)); e.append(wrap("e", body)); return e


def equation(doc, key):
    yi = sub("y", "i"); yhat = sub(hat("y"), "i")
    err = [mr("("), yi, mr("−"), yhat, mr(")")]
    spec = {
        "robust": ([sub("z", "ij"), mr("=clip("), frac([sub("x", "ij"), mr("−"), sub("m", "j")], sub("s", "j")), mr(",−8,8)")], "3-1"),
        "score": ([sub("S", "j"), mr("="), frac([mr("0.70"), sub("r", "I,j"), mr("+0.30"), sub("r", "C,j")], sup([mr("(1+"), sub("d", "j"), mr(")")], "0.25"))], "3-2"),
        "metrics": ([mr("MSE="), frac("1", "n"), summation(sup(err, "2"))], "4-1"),
        "ensemble": ([hat("y"), mr("(x)="), frac("1", "3"), summation([sub("f", "s"), mr("(x)")], "s=1", "3")], "4-2"),
        "interval": ([mr("k=⌈(m+1)(1−α)⌉,  q="), sub("r", "(k)"), mr(",  I(x)=["), hat("y"), mr("(x)−q,"), hat("y"), mr("(x)+q]")], "4-3"),
    }
    body, number = spec[key]
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_together = True
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(8)
    math = OxmlElement("m:oMath"); math.extend(body); p._p.append(math)
    font(p.add_run(f"    （{number}）"), size=11)


def build():
    text, references = load_evidence()
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"; normal.font.size = Pt(12)
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    for name in ["Title", "Heading 1", "Heading 2", "Heading 3"]:
        doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)
    # The packaged Word default may carry a blue paragraph border on Title.
    for style in doc.styles:
        for border in style.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    for s in doc.sections:
        s.page_width = Cm(21); s.page_height = Cm(29.7)
        s.top_margin = Cm(2.4); s.bottom_margin = Cm(2.3)
        s.left_margin = Cm(2.8); s.right_margin = Cm(2.8)
        s.header_distance = Cm(1.1); s.footer_distance = Cm(1.0)
    # Quiet cover; no fabricated institution, author, date or approval status.
    p = paragraph(doc, "研究论文", size=17, center=True, indent=False)
    p.paragraph_format.space_before = Pt(100)
    p = paragraph(doc, "基于机器学习的TFT-LCD\n多工序质量特性预测研究", size=23, center=True, indent=False)
    p.style = doc.styles["Title"]
    title_border = OxmlElement("w:pBdr")
    for side in ["top", "bottom", "left", "right"]:
        edge = OxmlElement("w:" + side); edge.set(qn("w:val"), "nil"); title_border.append(edge)
    p._p.get_or_add_pPr().append(title_border)
    p.paragraph_format.space_before = Pt(45); p.paragraph_format.line_spacing = 1.6
    font(p.runs[0], size=23, bold=True, chinese="黑体")
    p = paragraph(doc, "审稿修订稿", size=13, center=True, indent=False)
    p.paragraph_format.space_before = Pt(90)
    paragraph(doc, "离线特性回归  实体隔离  可复现验证", size=11, center=True, indent=False)
    front = doc.add_section(WD_SECTION.NEW_PAGE)
    front.footer.is_linked_to_previous = False
    field(front.footer.paragraphs[0], " PAGE ")
    front.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    pg = OxmlElement("w:pgNumType"); pg.set(qn("w:fmt"), "upperRoman"); pg.set(qn("w:start"), "1"); front._sectPr.append(pg)
    first_heading = True
    body_started = False
    lines = text.splitlines(); i = 0
    comments_added = set()
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line:
            continue
        if line == "# 1 绪论":
            toc_heading = paragraph(doc, "目录", size=16, indent=False)
            toc_heading.paragraph_format.page_break_before = True
            toc_heading.paragraph_format.keep_with_next = True
            font(toc_heading.runs[0], size=16, bold=True, chinese="黑体")
            toc = paragraph(doc, indent=False)
            field(toc, ' TOC \\o "1-2" \\h \\z \\u ', "目录将在Word中更新")
            section = doc.add_section(WD_SECTION.NEW_PAGE)
            section.footer.is_linked_to_previous = False
            section.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            field(section.footer.paragraphs[0], " PAGE ")
            existing = section._sectPr.find(qn("w:pgNumType"))
            if existing is not None:
                section._sectPr.remove(existing)
            pg = OxmlElement("w:pgNumType"); pg.set(qn("w:fmt"), "decimal"); pg.set(qn("w:start"), "1"); section._sectPr.append(pg)
            body_started = True
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            p = heading(doc, line[level:].strip(), min(level, 3))
            if first_heading or line == "# 1 绪论":
                p.paragraph_format.page_break_before = False
                first_heading = False
            # TOC excludes front matter and itself.
            if not body_started:
                out = OxmlElement("w:outlineLvl"); out.set(qn("w:val"), "9"); p._p.get_or_add_pPr().append(out)
            continue
        if line.startswith("|"):
            block = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip()); i += 1
            table(doc, block); continue
        figure = re.fullmatch(r"!\[(.*)\]\((.*)\)", line)
        if figure:
            imagepath = RESULTS / "plots" / figure[2]
            if not imagepath.exists():
                raise FileNotFoundError(imagepath)
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(imagepath), width=Cm(15.2))
            cap = paragraph(doc, figure[1], size=10.5, indent=False, center=True)
            cap.paragraph_format.space_after = Pt(10)
            continue
        if line.startswith("$$"):
            equation(doc, line.strip("$")); continue
        is_caption = bool(re.match(r"^表\d+-\d+\s", line))
        p = paragraph(doc, line, indent=not (line.startswith(("关键词", "Keywords")) or is_caption))
        if is_caption:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            font(p.runs[0], size=10.5)
        remaining = next((item.strip() for item in lines[i:] if item.strip()), "")
        if remaining.startswith("$$"):
            p.paragraph_format.keep_with_next = True
        if re.search(r"[A-Za-z]{4}", line) and not re.search(r"[\u4e00-\u9fff]", line):
            p.paragraph_format.first_line_indent = Pt(0)
        notes = {
            "原始X共5952列": "R14：已区分原始字段与清理视图的统计分母，数字依据missing_denominator_audit.json。",
            "本轮A配置与拟合": "R05：A标签曾被历史研究查看，只保留回顾性结果身份，未将脚本读取顺序当作独立测试依据。",
            "即使校准与评价": "R06：冻结拟合、校准、尾段用途分离仍不能证明交换性，只报告87%的经验覆盖。",
            "固定完整视图的合并MSE": "R12：保留选择流程未胜过固定对照的负向结果，所有方法在同300条记录上比较。",
        }
        for trigger, note in notes.items():
            if trigger in line and trigger not in comments_added:
                doc.add_comment(p.runs, note, author="审稿修订", initials="R")
                comments_added.add(trigger)
    heading(doc, "参考文献", 1)
    for ref in references:
        p = paragraph(doc, f"[{ref['number']}] {ref['text']}", size=10.5, indent=False)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Pt(20); p.paragraph_format.first_line_indent = Pt(-20)
        p.paragraph_format.line_spacing = 1.15; p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_together = True
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "TFT-LCD匿名数据的离线机器学习特性回归与历史验证"
    doc.core_properties.author = ""
    doc.core_properties.comments = "实验结果来自outputs/review；引用核验等级单独保留。"
    doc.save(DEST)
    reftext = "\n\n# 参考文献\n\n" + "\n\n".join(f"[{r['number']}] {r['text']}" for r in references)
    (HERE / "论文审稿修订版.md").write_text(text + reftext, encoding="utf-8")
    (HERE / "citation_map.json").write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"docx": str(DEST), "references": len(references), "comments": len(comments_added)}, ensure_ascii=False))


if __name__ == "__main__":
    build()
