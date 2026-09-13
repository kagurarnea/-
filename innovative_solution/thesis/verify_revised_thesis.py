"""Structural checks complement, but do not replace, full-page visual review."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DOCX = HERE / "基于机器学习的TFT-LCD多工序质量特性预测研究.docx"
PDF = HERE / "paper_render" / "thesis.pdf"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}


def verify():
    with ZipFile(DOCX) as z:
        root = etree.fromstring(z.read("word/document.xml"))
        text = "".join(root.xpath("//w:t/text()", namespaces=NS))
        assert "{{" not in text and "[@" not in text
        assert "目录将在Word中更新" not in text
        equations = root.xpath("//m:oMath", namespaces=NS)
        assert len(equations) == 5
        assert len(root.xpath("//m:f", namespaces=NS)) == 4
        ids = set()
        if "word/comments.xml" in z.namelist():
            comments = etree.fromstring(z.read("word/comments.xml"))
            ids = set(comments.xpath("//w:comment/@w:id", namespaces=NS))
        for tag in ["commentRangeStart", "commentRangeEnd", "commentReference"]:
            assert set(root.xpath(f"//w:{tag}/@w:id", namespaces=NS)) == ids
        assert len(ids) == 0
        for phrase in ["旧实验", "旧方案", "原实验", "原稿", "修订", "审稿", "本轮", "第一版", "原训练"]:
            assert phrase not in text, phrase
        manifest=json.loads((HERE.parent/'outputs'/'branch_selection'/'review_manifest.json').read_text(encoding='utf-8'))
        assert manifest['selected_candidate'] in text
        assert f"{manifest['tail_metrics']['mse']:.6f}" in text
        assert f"{manifest['retrospective_A_metrics']['mse']:.6f}" in text
        assert "0.023242" in text and "0.005999" in text and "0.022253" in text
        assert "87/100" in text and "0.2683" in text
        assert len(root.xpath("//w:tbl", namespaces=NS)) == 13
    manuscript = (HERE / "论文正文.md").read_text(encoding="utf-8")
    before_refs, references = manuscript.split("# 参考文献")
    appearance = list(dict.fromkeys(re.findall(r"\[(\d+)\]", before_refs)))
    assert appearance == [str(i) for i in range(1, 18)]
    assert re.findall(r"(?m)^\[(\d+)\]", references) == appearance
    pages = PdfReader(PDF).pages
    extracted = [p.extract_text() for p in pages]
    assert all(len(t.strip()) > 3 for t in extracted)
    assert not any("Error!" in t or "错误!" in t or "�" in t for t in extracted)
    (HERE / "paper_render" / "rendered_text.txt").write_text("\n\f\n".join(extracted), encoding="utf-8")
    report = {
        "docx_sha256": hashlib.sha256(DOCX.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "pdf_pages": len(pages), "native_equations": 5, "tables": 13,
        "references_in_first_appearance_order": 17, "anchored_review_comments": 0,
        "structural_checks": "passed", "render_engine": "Microsoft Word COM",
        "canonical_renderer_failure": "Bundled LibreOffice unavailable; log retained",
        "page_summaries": [{"page": i+1, "chars": len(t), "opening": t[:100], "ending": t[-80:]} for i, t in enumerate(extracted)],
        "visual_review": "See separate final visual review record; structural checks alone do not verify layout.",
    }
    (HERE / "paper_render" / "structural_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(pages), "equations": 5, "references": 17, "comments": 0, "status": "passed"}))


if __name__ == "__main__":
    verify()
