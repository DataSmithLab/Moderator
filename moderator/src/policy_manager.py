import json
import os
from moderator.src.configs.experiment_config import PolicyConfig

class PolicyManager:
    def __init__(self):
        self.database_path = "moderator/data/policy_database.json"
        self.policy_database = self.load_database()
        
    def check_policy(self, policy_name: str):
        return policy_name in self.policy_database

    def get_all_policies(self):
        return list(self.policy_database.keys())

    def load_database(self):
        with open(self.database_path, "r")as f:
            policy_database = json.load(f)
        return policy_database

    def write_database(self):
        with open(self.database_path, "w")as f:
            json.dump(self.policy_database, f)

    def add_policy(self, policy_config: PolicyConfig):
        self.policy_database[policy_config.task_name] = policy_config.to_dict()
        self.write_database()

    def delete_policy(self, policy_name: str):
        if policy_name in self.policy_database:
            del self.policy_database[policy_name]
            self.write_database()
            return True
        else:
            return False