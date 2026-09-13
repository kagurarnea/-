"""Build a standalone research manuscript from saved experimental evidence."""
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
DEST = HERE / f"{TITLE}.docx"
KEYS = dict(kusiak=3, competition=2, qu=1, friedman=7, xgboost=10,
            rf=8, et=9, stacking=12, ridge=11, tabular=13, tabpfn=17,
            missing=6, drift=14, tashman=15, conformal=16, lightgbm=22, catboost=23)
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
    labels = {"mean_baseline": "训练均值基线", "fixed_raw_d2": "基础特征 深度2",
              "fixed_full_d2": "组合特征 深度2", "nested_selected": "嵌套选参集成"}
    lines = ["表5-1 同一300条外层记录的合并指标", "",
             "| 方法 | MSE | RMSE | MAE | R² |", "|---|---:|---:|---:|---:|"]
    for name in ["mean_baseline", "fixed_raw_d2", "fixed_full_d2", "nested_selected"]:
        row = score[name]
        lines.append("| " + labels[name] + " | " + " | ".join(f"{float(row[k]):.6f}" for k in ["mse", "rmse", "mae", "r2"]) + " |")
    values["nested_table"] = "\n".join(lines)
    delta = 100 * (float(score["fixed_raw_d2"]["mse"]) - float(score["fixed_full_d2"]["mse"])) / float(score["fixed_raw_d2"]["mse"])
    values["nested_interpretation"] = (
        f"深度2的组合特征模型取得表中最低MSE，较基础特征模型降低约{delta:.2f}%。"
        "两种表示均优于均值基线，说明生产特征提供了与Y相关的预测信息。"
        "嵌套选参集成的误差高于两种固定配置，显示有限内层样本上的最小误差配置未必在外层保持优势。"
        "因此，工序特征的整体效果与参数选择的效果应分别讨论。")
    from innovative_solution.thesis.supplement_content import load as load_supplement
    values.update(load_supplement())
    from innovative_solution.thesis.branch_content import load as load_branches
    values.update(load_branches())
    text = (HERE / "manuscript.md").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    assert "{{" not in text, "Unfilled result placeholder"
    order = list(dict.fromkeys(re.findall(r"\[@([a-z]+)\]", text)))
    refs = json.loads((PACKAGE / "review" / "references_verified.json").read_text(encoding="utf-8"))["references"]
    refs = {r["id"]: r for r in refs}
    refs[22] = {"id":22,"verification":"NeurIPS publisher title authors and abstract inspected",
        "text":"Guolin Ke, Qi Meng, Thomas Finley, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree[C]//Advances in Neural Information Processing Systems 30. 2017. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html."}
    refs[23] = {"id":23,"verification":"NeurIPS publisher title authors and abstract inspected",
        "text":"Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, et al. CatBoost: unbiased boosting with categorical features[C]//Advances in Neural Information Processing Systems 31. 2018. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html."}
    entries = []
    for i, key in enumerate(order, 1):
        source = refs[KEYS[key]]
        entries.append({"number": i, "key": key, "original_id": source["id"],
                        "text": reference_text(source), "verification": source["verification"]})
        text = text.replace(f"[@{key}]", f"[{i}]")
    return text, entries


def reference_text(r):
    rid = r["id"]
    if "text" in r:
        return r["text"]
    if rid == 1:
        return "屈方磊. 基于集成学习的智能制造质量预测研究[D]. 中北大学, 2024."
    if rid == 2:
        return "阿里云天池. 智能制造质量预测赛题[EB/OL]. " + r["url"] + "."
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
    elif n == 5 and rows[0][0] == "统计范围":
        widths = [4.2, 1.8, 3.0, 3.0, 3.4]
    elif n == 5:
        widths = [5.0, 2.6, 2.6, 2.6, 2.6]
    elif n == 4 and rows[0][0] == "比较 参考减候选":
        widths = [4.1, 6.3, 2.5, 2.5]
    elif n == 4 and rows[0][0] == "候选配置":
        widths = [4.0, 3.8, 3.8, 3.8]
    elif n == 4 and rows[0][0] in ("方法", "方案", "算法"):
        widths = [5.5, 3.3, 3.3, 3.3]
    elif n == 4:
        widths = [2.8, 4.0, 4.2, 4.4]
    else:
        widths = [15.4/n] * n
    for col, width in zip(tab.columns, widths):
        col.width = Cm(width)
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "bottom", "left", "right", "insideH", "insideV"]:
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "D9D9D9"); borders.append(e)
    tab._tbl.tblPr.append(borders)
    for i, row in enumerate(rows):
        trpr = tab.rows[i]._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for j, value in enumerate(row):
            cell = tab.cell(i, j); cell.width = Cm(widths[j]); cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if i == 0:
                shade = OxmlElement("w:shd"); shade.set(qn("w:fill"), "EDEDED"); cell._tc.get_or_add_tcPr().append(shade)
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
    front = doc.add_section(WD_SECTION.NEW_PAGE)
    front.footer.is_linked_to_previous = False
    field(front.footer.paragraphs[0], " PAGE ")
    front.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    pg = OxmlElement("w:pgNumType"); pg.set(qn("w:fmt"), "upperRoman"); pg.set(qn("w:start"), "1"); front._sectPr.append(pg)
    first_heading = True
    body_started = False
    lines = text.splitlines(); i = 0
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
            imagepath = PACKAGE / "outputs" / "branch_selection" / "plots" / figure[2]
            if not imagepath.exists():
                imagepath = RESULTS / "plots" / figure[2]
            if not imagepath.exists():
                imagepath = PACKAGE / "outputs" / "supplement" / "plots" / figure[2]
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
        is_caption = bool(re.match(r"^表\d+-\d+[a-z]?\s", line))
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
    heading(doc, "参考文献", 1)
    for ref in references:
        p = paragraph(doc, f"[{ref['number']}] {ref['text']}", size=10.5, indent=False)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Pt(20); p.paragraph_format.first_line_indent = Pt(-20)
        p.paragraph_format.line_spacing = 1.15; p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_together = True
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "TFT-LCD多工序生产数据的机器学习质量特性预测"
    doc.core_properties.author = ""
    doc.core_properties.comments = ""
    doc.save(DEST)
    reftext = "\n\n# 参考文献\n\n" + "\n\n".join(f"[{r['number']}] {r['text']}" for r in references)
    (HERE / "论文正文.md").write_text(text + reftext, encoding="utf-8")
    (HERE / "citation_map.json").write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"docx": str(DEST), "references": len(references), "comments": 0}, ensure_ascii=False))


if __name__ == "__main__":
    build()
