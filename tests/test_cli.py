"""Tests for the command-line interface.

The CLI is driven through ``cli.main(argv)`` which returns a process exit code.
We check that core commands run and print sensible output, that saved files are
actually created, and that bad input produces a clean error (exit code 1 and a
message) rather than a traceback.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourierlab import cli


def test_version_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_dft_command_runs(capsys) -> None:
    rc = cli.main(["dft", "--signal", "1,0,-1,0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DFT" in out
    assert "reconstruction" in out
    assert "Parseval" in out


def test_fft_command_runs(capsys) -> None:
    rc = cli.main(["fft", "--signal", "1,0,-1,0", "--sample-rate", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Cooley-Tukey" in out
    assert "max NumPy error" in out


def test_fft_non_power_of_two_reports_error(capsys) -> None:
    rc = cli.main(["fft", "--signal", "1,2,3"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "power of two" in err


def test_idft_and_ifft_run(capsys) -> None:
    assert cli.main(["idft", "--spectrum", "0,2,0,2"]) == 0
    assert cli.main(["ifft", "--spectrum", "0,2,0,2"]) == 0
    out = capsys.readouterr().out
    assert out.count("reconstruction") == 2


def test_dft_accepts_complex_input(capsys) -> None:
    rc = cli.main(["dft", "--signal", "1+1j,2-1j,0,3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "complex input      = yes" in out


def test_invalid_signal_reports_error(capsys) -> None:
    rc = cli.main(["dft", "--signal", "a,b,c"])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("error:")


def test_generate_creates_csv(tmp_path, capsys) -> None:
    out_csv = tmp_path / "sig.csv"
    rc = cli.main([
        "generate", "--kind", "mixed", "--freqs", "5,20,50",
        "--amps", "1,0.5,0.2", "--noise", "0.1", "--seed", "0",
        "--save", str(out_csv),
    ])
    assert rc == 0
    assert out_csv.exists()
    assert "mixed" in capsys.readouterr().out


def test_generate_is_deterministic(tmp_path) -> None:
    paths = []
    for i in range(2):
        p = tmp_path / f"sig{i}.csv"
        cli.main([
            "generate", "--kind", "sine", "--frequency", "10",
            "--noise", "0.2", "--seed", "3", "--save", str(p),
        ])
        paths.append(p)
    assert paths[0].read_text() == paths[1].read_text()


def test_spectrum_creates_png(tmp_path) -> None:
    csv = tmp_path / "sig.csv"
    png = tmp_path / "spec.png"
    cli.main([
        "generate", "--kind", "sine", "--frequency", "10",
        "--sample-rate", "128", "--duration", "1", "--save", str(csv),
    ])
    rc = cli.main(["spectrum", "--input", str(csv), "--save", str(png)])
    assert rc == 0
    assert png.exists()


def test_spectrum_finds_dominant_frequency(tmp_path, capsys) -> None:
    csv = tmp_path / "sig.csv"
    cli.main([
        "generate", "--kind", "sine", "--frequency", "10",
        "--sample-rate", "128", "--duration", "1", "--save", str(csv),
    ])
    capsys.readouterr()  # discard generate output
    cli.main(["spectrum", "--input", str(csv)])
    out = capsys.readouterr().out
    assert "10.000 Hz" in out


def test_filter_creates_outputs(tmp_path) -> None:
    csv = tmp_path / "sig.csv"
    png = tmp_path / "filtered.png"
    out_csv = tmp_path / "filtered.csv"
    cli.main([
        "generate", "--kind", "mixed", "--freqs", "5,50",
        "--amps", "1,1", "--sample-rate", "256", "--duration", "1",
        "--save", str(csv),
    ])
    rc = cli.main([
        "filter", "--input", str(csv), "--type", "lowpass",
        "--cutoff", "20", "--save", str(png), "--export", str(out_csv),
    ])
    assert rc == 0
    assert png.exists()
    assert out_csv.exists()


def test_filter_missing_file_reports_error(capsys) -> None:
    rc = cli.main(["filter", "--input", "nope.csv", "--type", "lowpass", "--cutoff", "5"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "error:" in err


def test_filter_bad_cutoff_reports_error(tmp_path, capsys) -> None:
    csv = tmp_path / "sig.csv"
    cli.main(["generate", "--kind", "sine", "--save", str(csv)])
    capsys.readouterr()
    rc = cli.main(["filter", "--input", str(csv), "--type", "bandpass", "--low", "30", "--high", "10"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "error:" in err


def test_series_creates_png(tmp_path, capsys) -> None:
    png = tmp_path / "square.png"
    rc = cli.main(["series", "--function", "square", "--terms", "25", "--save", str(png)])
    out = capsys.readouterr().out
    assert rc == 0
    assert png.exists()
    assert "mean sq. error" in out
    assert "max error" in out


def test_compare_runs(capsys) -> None:
    rc = cli.main(["compare", "--sizes", "32", "64", "128", "--repeats", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "speedup" in out


def test_gibbs_runs(capsys) -> None:
    rc = cli.main(["gibbs", "--terms", "5", "20"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "overshoot" in out


def test_no_command_errors() -> None:
    # argparse requires a subcommand.
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0
