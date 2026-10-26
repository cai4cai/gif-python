import logging
import stat

import pytest

from src.utils.process import ExternalToolError, run_tool


def make_tool(tmp_path, body, name="tool"):
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_successful_run_returns_output_and_logs_it_at_debug(tmp_path, caplog):
    tool = make_tool(tmp_path, "echo hello; echo world")
    with caplog.at_level(logging.DEBUG, logger="src.utils.process"):
        result = run_tool([tool, "arg"])
    assert result.stdout == "hello\nworld\n"
    assert any("hello" in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


def test_failure_raises_with_command_and_output_tail(tmp_path):
    tool = make_tool(tmp_path, "echo some context; echo 'bad things' >&2; exit 3")
    with pytest.raises(ExternalToolError) as exc:
        run_tool([tool, "-x", "a file with spaces"])
    message = str(exc.value)
    assert "exit status 3" in message and "bad things" in message and "a file with spaces" in message


def test_no_shell_is_involved(tmp_path):
    marker = tmp_path / "pwned"
    tool = make_tool(tmp_path, "exit 0")
    run_tool([tool, f"; touch {marker}"])
    assert not marker.exists()


def test_missing_executable_is_an_external_tool_error(tmp_path):
    with pytest.raises(ExternalToolError, match="Cannot execute"):
        run_tool([tmp_path / "does_not_exist"])


def test_niftyreg_warnings_are_summarised_not_failures(tmp_path, caplog):
    tool = make_tool(
        tmp_path,
        "echo '[NiftyReg WARNING] Function: reg_f3d<T>::ComputeJacobianBasedPenaltyTerm()';"
        "echo '[NiftyReg WARNING] The folding correction scheme failed'",
    )
    with caplog.at_level(logging.INFO, logger="src.utils.process"):
        run_tool([tool])
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1 and "2 warning line(s)" in info[0] and "informational" in info[0]
