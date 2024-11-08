import yaml

class Parameters():
    def __init__(self):
        self.underlying_features_K = 10
        self.steps = 10000
        self.lambda_r = 0.0001
        self.learning_rate = 0.01
    
    def __str__(self):
        return f'underlying_features_K: {self.underlying_features_K}\nsteps: {self.steps}\nlambda_r: {self.lambda_r}\nlearning_rate: {self.learning_rate}'
    
    def __repr__(self):
        return self.__str__()
    
    def name(self):
        return f'K{self.underlying_features_K}-lam{self.lambda_r}-learn{self.learning_rate}-'
    
    def loadConfig(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            paras = yaml.safe_load(f)
            self.setParas(
                paras['underlying_features_K'],
                paras['steps'],
                paras['lambda_r'],
                paras['learning_rate']
            )
        f.close() 
        
    def setParas(self, underlying_features_K, steps, lambda_r, learning_rate):
        self.underlying_features_K = underlying_features_K
        self.steps = steps
        self.lambda_r = lambda_r
        self.learning_rate = learning_rate