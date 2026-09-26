"""
validate_output.py: Wrapper to execute the official submission validator.
"""
import os
import sys
import subprocess
from src.config import BASE_DIR, MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH, TEST_DIR

def run_validation(check_ids: bool = False):
    validator_path = os.path.join(BASE_DIR, "utils", "validate_submission.py")
    if not os.path.exists(validator_path):
        print(f"Error: Validator script not found at {validator_path}")
        return 1

    cmd = [
        sys.executable,
        validator_path,
        "--matching", MATCHING_RESULTS_PATH,
        "--test-dir", TEST_DIR
    ]
    if os.path.exists(CANDIDATE_PAIRS_PATH):
        cmd.extend(["--candidate", CANDIDATE_PAIRS_PATH])
    if check_ids:
        cmd.append("--check-ids")

    print(f"Running validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode

if __name__ == "__main__":
    check_ids_flag = "--check-ids" in sys.argv
    ret = run_validation(check_ids=check_ids_flag)
    sys.exit(ret)
