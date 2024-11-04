from parameter import Parameters
from process import Processer
from train import Trainer
from validate import Validator
from testing import Tester
import pandas as pd

class Main():
    def __init__(self):
        self.config_path = 'config/config.yaml'
        self.movies_path = 'data/ml-latest-small/movies.csv'
        self.ratings_path = 'data/ml-latest-small/ratings.csv'
        self.log_path = 'log/info.log'

    def run(self): 
        process = Processer(self.movies_path, self.ratings_path)
        trainer = Trainer(self.config_path, process.movies_num, process.users_num)
        
        with open(self.log_path, 'w', encoding='utf-8') as f:
            print('start training...\n', file = f)
            Rating_test = process.dataset_slices[0]
            for i in range(1, 10):
                print(f"the turn {i} start!", file = f)
                Rating_validate = process.dataset_slices[i]
                Rating_train = pd.concat(process.dataset_slices[1:i] +process.dataset_slices[i+1:])
                trainer.train(Rating_train)
                # trainer.save(trainer.name + f'turn{i}')
                validator = Validator(trainer, Rating_validate)
                result = validator.result
                print(f"the turn {i} finished, the validating result is: {result}", file = f)
            print('start tesing...\n', file = f)
            tester = Tester(trainer, Rating_test)
            result = tester.result
            print(f"the turn {i} finished, the validating result is: {result}", file = f)
            
        f.close()
        

if __name__ == "__main__":
    Main().run()