from process import Processer
from train import Trainer
from validate import Validator
from testing import Tester
import pandas as pd

class Small():
    def __init__(self):
        self.config_path = 'config/config.yaml'
        self.movies_path = 'data/small/movies.csv'
        self.ratings_path = 'data/small/ratings.csv'
        self.log_path = 'log/info.log'

    def run(self): 
        process = Processer(self.movies_path, self.ratings_path)
        trainer = Trainer(self.config_path, process.movies_num, process.users_num)
        
        print('start training...\n')
        Rating_test = process.dataset_slices[0]
        for i in range(1, 2):
            print(f"the turn {i} start!")
            Rating_validate = process.dataset_slices[i]
            Rating_train = pd.concat(process.dataset_slices[1:i] +process.dataset_slices[i+1:])
            trainer.train(Rating_train)
            print(f'the turn {i} training finished!')
            
            validator = Validator(trainer, Rating_validate)
            result = validator.result
            print(f"the turn {i} finished, the validating result is: {result}")
        print('start tesing...')
        tester = Tester(trainer, Rating_test)
        result = tester.result
        print(f"the testing result is: {result}")

if __name__ == "__main__":
    Small().run()