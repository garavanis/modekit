"""
The example-data download: files come from the data repository once, then from
the local folder.
"""

from modekit import datasets


def test_fetch_downloads_missing_files_once(tmp_path, capsys):
    repo = tmp_path / "repo" / "sets"  # a stand-in for the data repository
    repo.mkdir(parents=True)
    (repo / "a.csv").write_text("1\t2\n")
    (repo / "b.csv").write_text("3\t4\n")
    url = (tmp_path / "repo").as_uri()

    out = datasets.fetch("sets", ["a.csv", "b.csv"], tmp_path / "cache", url=url)
    assert out == tmp_path / "cache" / "sets"
    assert (out / "a.csv").read_text() == "1\t2\n" and (out / "b.csv").read_text() == "3\t4\n"
    assert not list(out.glob("*.part"))
    assert capsys.readouterr().out.count("downloading") == 2

    (repo / "a.csv").write_text("changed")  # a cached file is not downloaded again
    datasets.fetch("sets", ["a.csv"], tmp_path / "cache", url=url)
    assert (out / "a.csv").read_text() == "1\t2\n"
    assert "downloading" not in capsys.readouterr().out
