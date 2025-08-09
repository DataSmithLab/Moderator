import json
import os
from moderator.src.configs.experiment_config import PolicyConfig

class PolicyManager:
    def __init__(self):
        self.database_path = "moderator/data/policy_database.json"
        self.policy_database = self.load_database()
        
    def get_all_policies(self):
        return list(self.policy_database.keys())

    def load_database(self):
        with open(self.database_path, "r")as f:
            policy_database = json.load(f)
        return policy_database

    def write_database(self):
        with open(self.database_path, "w")as f:
            json.dump(self.policy_database, f)

    def add_policy(self, new_policy_dict, new_policy_name):
        self.policy_database[new_policy_name] = new_policy_dict
        self.write_database()