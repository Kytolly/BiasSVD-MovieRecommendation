from parameter import Parameters
from process import Processer
from train import Trainer
from testing import Tester
from env import Env
from SVDmodel import Model 

class Main():
    def __init__(self):
        self.env = Env()
        self.paras = Parameters()
        self.paras.loadConfig(self.env.config_path)
        self.pre = Processer(self.env.movies_path, self.env.ratings_path)
        self.model = Model(self.pre.movies_num, self.pre.users_num, self.paras.underlying_features_K, self.paras.learning_rate)
        self.trainer = Trainer()
        self.trainer.compile(self.paras)

    def run(self): 
        with open(self.env.log_path, 'w', encoding='utf-8') as f:
            print('start training...\n', file = f)
            Rating_test = self.pre.getTestSet()
            
            for i in range(1, 10):
                Rating_validate = self.pre.getValidateSet(i)
                Rating_train = self.pre.getTrainSet(i)
                
                print(f"the turn {i} start!", file = f)
                self.trainer.train(self.model, Rating_train)
                validator = Tester(self.model, Rating_validate)
                print(f"the turn {i} finished, the validating result is: {validator.result}", file = f)
            
            print('start tesing...\n', file = f)
            tester = Tester(self.model, Rating_test) 
            print(f"the testing result is: {tester.result}", file = f)
            
        f.close()
        

if __name__ == "__main__":
    Main().run()