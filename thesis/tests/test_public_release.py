"""Public distribution behavior, not scientific performance assertions."""
import json
import sys
from pathlib import Path
import pytest
from mad_etd.cli import main


def test_demo_is_explicitly_synthetic_and_offline(tmp_path, monkeypatch, capsys):
    target = tmp_path / 'new_demo'
    monkeypatch.setattr(sys, 'argv', ['mad-etd','demo','--output',str(target)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['scope'] == 'synthetic_rule_demo_not_model_evaluation'
    assert result['network'] is False
    assert len(result['cases']) == 2
    assert len(list(target.glob('*.audit.jsonl'))) == 2


def test_demo_never_overwrites_an_existing_directory(tmp_path, monkeypatch):
    target = tmp_path / 'exists'
    target.mkdir()
    sentinel = target / 'keep.txt'
    sentinel.write_text('keep', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['mad-etd','demo','--output',str(target)])
    with pytest.raises(FileExistsError):
        main()
    assert sentinel.read_text(encoding='utf-8') == 'keep'


def test_demo_fixture_has_documentation_addresses():
    rows = [json.loads(line) for line in Path('examples/sample_flows.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(row['context']['src_ip'].startswith('192.0.2.') for row in rows)
