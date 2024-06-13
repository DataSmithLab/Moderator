import os
import json
import time
def deep_dict_equal(dict1, dict2):
    if dict1 is dict2:
        return True
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        return False
    if len(dict1) != len(dict2):
        return False

    for key in dict1:
        if key not in dict2:
            return False

        value1 = dict1[key]
        value2 = dict2[key]

        if isinstance(value1, dict) and isinstance(value2, dict):
            if not deep_dict_equal(value1, value2):
                return False
        elif value1 != value2:
            return False

    return True

class EditDatabase:
    def __init__(
        self,
        database_filename="./database.json"
    ):
        self.database_filename = database_filename
        self.database_dict = self.load_database(self.database_filename)
        
        self.task_vectors = self.register_all_task_vectors(self.database_dict)
        self.task_vectors = self.remove_duplicate_task_vectors(self.task_vectors)
        
        self.tasks_history = self.database_dict["tasks_history"]
    
    def store_database(self):
        with open(self.database_filename, "w+") as f:
            json.dump(self.database_dict, f)
    
    def add_task(self, task):
        self.tasks_history.append(task)
        for task_vector in task["task_vectors"]:
            self.add_task_vector(task_vector)
    
    def add_task_vector(self, task_vector):
        self.task_vectors.append(task_vector)
        self.task_vectors = self.remove_duplicate_task_vectors(self.task_vectors)
        
    def load_database(self, database_filename):
        if os.path.exists(database_filename):
            with open(database_filename, "r")as f:
                database_dict = json.load(f)
            return database_dict
        else:
            database_dict = {"time":str(time.ctime()),"tasks_history":[],"task_vectors":[]}
            with open(database_filename, "w+")as f:
                json.dump(database_dict, f)
            return {}
    
    def register_all_task_vectors(self, database_dict):
        all_task_vectors = []
        for task in database_dict["tasks_history"]:
            task_vectors = task["task_vectors"]
            all_task_vectors += task_vectors
        return all_task_vectors
    
    def judge_equal_task_vectors(self, a_task_vector, b_task_vector):
        return deep_dict_equal(a_task_vector, b_task_vector)
    
    def remove_duplicate_task_vectors(self, task_vectors):
        duplicate_idxs = []
        for i in range( len(task_vectors)-1 ):
            for j in range(i+1, len(task_vectors) ):
                if self.judge_equal_task_vectors(task_vectors[i], task_vectors[j]):
                    duplicate_idxs.append(j)
        new_task_vectors = [item for idx, item in enumerate(task_vectors) if idx not in duplicate_idxs]
        return new_task_vectors