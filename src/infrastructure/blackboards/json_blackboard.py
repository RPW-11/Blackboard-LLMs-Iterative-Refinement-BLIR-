from domain.interface.blackboard import BlackboardInterface
from typing import List
import os
import json


class JsonBlackboard(BlackboardInterface):
    def __init__(self, storage_path):
        os.makedirs(storage_path, exist_ok=True)
        print(f"Directory '{storage_path}' created or already exists.")
        super().__init__(storage_path)


    def save_attempt(self, data: dict):
        filename = self._get_file_name()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    def get_attempt_results(self, n = 10) -> List[dict]:
        """Get last n attempts and format them into a dict"""
        trials = os.listdir(self.storage_path)
        trials_no = [int(trial.split("_")[0]) for trial in trials]
        trials_no.sort()

        attempts = []
        for no in trials_no[-n:]:
            with open(os.path.join(self.storage_path, f"{no}_attempt.json"), 'r') as file:
                data = json.load(file)
                attempts.append(data)

        return attempts
    
    
    def _get_file_name(self) -> str:
        """Get the last name from the blackboard"""
        trials = os.listdir(self.storage_path)
        trials_no = [int(trial.split("_")[0]) for trial in trials]
        trials_no.sort()
        attempt_no = trials_no[-1] if len(trials_no) > 0 else 0

        return os.path.join(self.storage_path, f"{attempt_no + 1}_attempt.json")
