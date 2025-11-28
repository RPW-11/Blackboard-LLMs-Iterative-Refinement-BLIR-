from domain.interface.blackboard import BlackboardInterface
from typing import List
import os
import json


class JsonBlackboard(BlackboardInterface):
    def __init__(self, storage_path):
        os.makedirs(storage_path, exist_ok=True)
        print(f"Directory '{storage_path}' created or already exists.")
        super().__init__(storage_path)


    def save_attempt(self, data: dict, filename: str = None):
        filename = self._get_file_name(filename)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    def get_attempt_results(self, n = 10) -> List[dict]:
        """Get last n attempts and format them into a dict"""
        trials = os.listdir(self.storage_path)
        if len(trials) == 0:
            return []
        
        trials = [trial for trial in trials if not trial.startswith('.')]
        trials.sort(key=lambda x: int(x.split("_")[0]))

        attempts = []
        for filename in trials[-n:]:
            with open(os.path.join(self.storage_path, filename), 'r') as file:
                data = json.load(file)
                attempts.append(data)

        return attempts
    
    
    def _get_file_name(self, file_name: str = None) -> str:
        """Get the last name from the blackboard"""
        trials = os.listdir(self.storage_path)
        trials_no = [int(trial.split("_")[0]) for trial in trials if not trial.startswith('.')]
        trials_no.sort()
        attempt_no = trials_no[-1] if len(trials_no) > 0 else 0

        return os.path.join(self.storage_path, file_name if file_name else f"{attempt_no + 1}_attempt.json")
