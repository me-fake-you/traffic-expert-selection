from pathlib import Path

import pytest

from mad_etd.extract_ustc import _iter_packets, extract_dataset, extract_pcap


def test_missing_tshark_has_clear_error(monkeypatch):
    def missing():
        raise RuntimeError(
            "tshark is required for PCAP extraction. Install Wireshark/tshark "
            "and ensure tshark is on PATH."
        )

    monkeypatch.setattr("mad_etd.extract_ustc.resolve_tshark", missing)

    with pytest.raises(RuntimeError, match="tshark is required.*PATH"):
        list(_iter_packets(Path("missing.pcap")))


def test_extract_dataset_reports_when_no_pcaps_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="no PCAP files found"):
        list(extract_dataset(tmp_path))


def test_tcp_and_tls_display_protocol_share_one_stream(monkeypatch, tmp_path):
    rows = [
        [
            "1.0", "192.0.2.1", "", "50000", "", "192.0.2.2", "", "443", "",
            "100", "TCP", "7", "", "6", "", "",
        ],
        [
            "1.2", "192.0.2.2", "", "443", "", "192.0.2.1", "", "50000", "",
            "200", "TLSv1.3", "7", "", "6", "0x0304", "0x1301",
        ],
    ]
    monkeypatch.setattr("mad_etd.extract_ustc._iter_packets", lambda path: iter(rows))
    root = tmp_path / "USTC-TFC2016"
    pcap = root / "Benign" / "Gmail.pcap"
    pcap.parent.mkdir(parents=True)
    pcap.touch()

    records = list(extract_pcap(pcap, dataset_root=root))

    assert len(records) == 1
    assert records[0].stats["packet_count"] == 2
    assert records[0].sequence.directions == [1, -1]
    assert records[0].labels["application"] == "Gmail"


def test_ustc_sequence_is_truncated_without_corrupting_full_stats(
    monkeypatch, tmp_path
):
    rows = []
    for index in range(5):
        rows.append(
            [
                str(index),
                "192.0.2.1",
                "",
                "50000",
                "",
                "192.0.2.2",
                "",
                "443",
                "",
                "100",
                "TCP",
                "1",
                "",
                "6",
                "",
                "",
            ]
        )
    monkeypatch.setattr("mad_etd.extract_ustc._iter_packets", lambda path: iter(rows))
    root = tmp_path / "USTC-TFC2016"
    pcap = root / "Malware" / "Zeus.pcap"
    pcap.parent.mkdir(parents=True)
    pcap.touch()

    record = next(extract_pcap(pcap, dataset_root=root, sequence_limit=3))

    assert record.stats["packet_count"] == 5
    assert len(record.sequence.packet_lengths) == 3
    assert record.sequence.original_packet_count == 5
    assert record.sequence.truncated is True


def test_single_packet_udp_flow_is_preserved(monkeypatch, tmp_path):
    rows = [[
        "1.0", "192.0.2.1", "", "", "50000", "192.0.2.2", "", "", "443",
        "120", "UDP", "", "3", "17", "", "",
    ]]
    monkeypatch.setattr("mad_etd.extract_ustc._iter_packets", lambda path: iter(rows))
    root = tmp_path / "USTC-TFC2016"
    pcap = root / "Benign" / "Facetime.pcap"
    pcap.parent.mkdir(parents=True)
    pcap.touch()

    records = list(extract_pcap(pcap, dataset_root=root))

    assert len(records) == 1
    assert records[0].stats["packet_count"] == 1
    assert records[0].sequence.iats == []
