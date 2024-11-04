from src.parameter import Parameters
from src.process import Processer
from src.train import Trainer
from src.validate import Validator
from src.testing import Tester
import pandas as pd

class Draft():
    def __init__(self):
        self.config_path = 'config/config.yaml'
        self.movies_path = 'data/small/movies.csv'
        self.ratings_path = 'data/small/ratings.csv'
        self.log_path = 'log/info.log'

    def run(self): 
        process = Processer(self.movies_path, self.ratings_path)
        trainer = Trainer(process.movies_num , process.users_num, self.config_path)
        
        with open(self.log_path, 'w', encoding='utf-8') as f:
            print('start training...\n', file = f)
            Rating_test = process.rating_slices[0]
            for i in range(1, 10):
                print(f"the turn {i} start!", file = f)
                Rating_validate = process.rating_slices[i]
                Rating_train = pd.concat(process.rating_slices[1:i] +process.rating_slices[i+1:])
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
    Draft().run()