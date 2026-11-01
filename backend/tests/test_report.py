from app.report import build_report
from app.synthetic import write_benchmark


def test_report_writes_audio_rolls_and_exports(tmp_path):
    benchmark = tmp_path / "benchmark"
    write_benchmark(
        benchmark,
        instrument="alto",
        only=["repeated-articulated"],
    )

    report = build_report(benchmark, mock=True, instrument="alto")

    page = report.read_text()
    assert "repeated-articulated" in page
    assert "<audio controls" in page
    assert '<canvas class="roll"' in page
    assert 'data-score-kind="predicted"' in page
    assert '"predictedXml":' in page
    assert '"referenceXml":' in page
    assert "Onset F" in page

    predictions = benchmark / "predictions"
    assert (predictions / "repeated-articulated.notes.json").exists()
    assert (predictions / "repeated-articulated.musicxml").exists()
    assert (predictions / "repeated-articulated.mid").exists()
    assert (predictions / "repeated-articulated.reference.musicxml").exists()


def test_report_title_is_not_reprocessed_as_a_template(tmp_path):
    benchmark = tmp_path / "benchmark"
    write_benchmark(benchmark, instrument="alto", only=["repeated-articulated"])

    report = build_report(benchmark, mock=True, title="__DATA__ $title")

    page = report.read_text()
    assert "<title>__DATA__ $title</title>" in page
    assert "<h1>__DATA__ $title</h1>" in page
