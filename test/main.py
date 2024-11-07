from process import Processer
from parameter import Parameters
from env import Env
from mymodel import Model
from train import Trainer 
from testing import Tester

class Main():
    def __init__(self): 
        env = Env()
        self.paras = Parameters()
        self.paras.loadConfig(env.config_path)
        self.pre = Processer(env.movies_path, env.ratings_path)
        self.model = Model(self.pre.movies_num, self.pre.users_num, self.paras.underlying_features_K, self.paras.learning_rate)
        self.trainer = Trainer()
        self.trainer.compile(self.paras)
        
    def run(self): 
        print('start training...\n')
        Rating_test = self.pre.getTestSet()
        for i in range(1, 10):
            print(f"the turn {i} start!")
            Rating_validate = self.pre.getValidateSet(i)
            Rating_train = self.pre.getTrainSet(i)
            self.trainer.train(self.model, Rating_train)
            print(f'the turn {i} training finished!')
            
            validator = Tester(self.model, Rating_validate)
            result = validator.result
            print(f"the turn {i} finished, the validating result is: {result}")
        
        print('start tesing...')
        tester = Tester(self.trainer, Rating_test)
        result = tester.result
        print(f"the testing result is: {result}")

if __name__ == "__main__":
    Main().run()