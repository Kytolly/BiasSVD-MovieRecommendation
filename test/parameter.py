import yaml

class Parameters():
    def __init__(self, config_path: str):
        self.loadConfig(config_path)
    
    def __str__(self):
        return f'underlying_features_K: {self.underlying_features_K}\nsteps: {self.steps}\nlambda_r: {self.lambda_r}\nlearning_rate: {self.learning_rate}'
    
    def __repr__(self):
        return self.__str__()
    
    def name(self):
        return f'K{self.underlying_features_K}-lam{self.lambda_r}-learn{self.learning_rate}-'
    
    def loadConfig(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            paras = yaml.safe_load(f)
            self.underlying_features_K = paras['underlying_features_K']
            self.steps = paras['steps']
            self.lambda_r = paras['lambda_r'] 
            self.learning_rate = paras['learning_rate']
        f.close() 