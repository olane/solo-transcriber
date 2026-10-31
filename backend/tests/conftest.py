import os
import shutil
import tempfile

TEST_DATA_DIR = tempfile.mkdtemp(prefix="solo-transcriber-test-")
os.environ["ST_MOCK"] = "1"
os.environ["ST_DATA_DIR"] = TEST_DATA_DIR


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
