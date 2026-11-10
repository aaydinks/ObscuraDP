"""The example in the README runs as printed."""

import re
from pathlib import Path

import pytest

from obscuradp.cli import main

README = Path(__file__).resolve().parents[1] / "README.md"


def test_the_first_example_runs_and_its_report_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    if not README.is_file():
        pytest.skip("the README is not part of this installation")
    text = README.read_text(encoding="utf-8")
    section = text.split("## A first example", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert code is not None
    monkeypatch.chdir(tmp_path)
    exec(compile(code.group(1), "README example", "exec"), {"__name__": "readme_example"})  # noqa: S102
    printed = capsys.readouterr().out
    assert printed.startswith("PRIVACY REPORT")
    assert "(within budget; the largest spend of any scope)" in printed
    assert "'auc':" in printed
    assert (tmp_path / "privacy_report.json").is_file()
    command = re.search(r"```bash\n(obscuradp report verify .*?)\n```", section)
    assert command is not None
    assert main(command.group(1).split()[1:]) == 0
    assert "consistent" in capsys.readouterr().out
