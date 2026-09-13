$ErrorActionPreference = 'Stop'
$thesisFolder = $PSScriptRoot
$source = Join-Path $thesisFolder '基于机器学习的TFT-LCD多工序质量特性预测研究.docx'
$renderFolder = Join-Path $thesisFolder 'paper_render'
New-Item -ItemType Directory -Path $renderFolder -Force | Out-Null
$pdf = Join-Path $renderFolder 'thesis.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
    $doc = $word.Documents.Open($source, $false, $false)
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() | Out-Null }
    foreach ($toc in $doc.TablesOfContents) {
        $toc.Range.Font.Size = 10.5
        $toc.Range.ParagraphFormat.LineSpacingRule = 0
        $toc.Range.ParagraphFormat.SpaceBefore = 0
        $toc.Range.ParagraphFormat.SpaceAfter = 3
    }
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() | Out-Null }
    $doc.Save()
    $pages = $doc.ComputeStatistics(2)
    # Export only the document, without reviewer balloon markup.
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 1, 1, 0)
    Write-Output "Rendered pages: $pages"
    Write-Output $pdf
} finally {
    if ($null -ne $doc) { $doc.Close($false) }
    $word.Quit()
}
