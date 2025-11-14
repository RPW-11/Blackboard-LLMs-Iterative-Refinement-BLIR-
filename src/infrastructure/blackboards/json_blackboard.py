from domain.interface.blackboard import BlackboardInterface
import os
import json


class JsonBlackboard(BlackboardInterface):
    def __init__(self, storage_path):
        super().__init__(storage_path)


    def save_attempt(self, data: dict):
        filename = self._get_file_name()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    def get_attempt_results(self, n = 10) -> str:
        """Get last n attempts and format them into a string, separated by a new line"""
        trials = os.listdir(self.storage_path)
        trials.sort()
        trials = trials[-n:] if n < len(trials) else trials

        attempts_str = ""
        for trial in trials:
            with open(os.path.join(self.storage_path, trial), 'r') as file:
                attempt_no = trial.split("_")[0]
                attempts_str += "=" * 14 + f"ATTEMPT {attempt_no}" + "=" * 14 + "\n"
                attempts_str += file.read() + "\n"

        return attempts_str
    
    
    def _get_file_name(self) -> str:
        """Get the last name from the blackboard"""
        trials = os.listdir(self.storage_path)
        trials.sort()
        attempt_no = int(trials[-1].split("_")[0]) if len(trials) > 0 else 0

        return os.path.join(self.storage_path, f"{attempt_no + 1}_attempt.json")
