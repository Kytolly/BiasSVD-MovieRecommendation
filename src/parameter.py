import yaml

class Parameters():
    def __init__(self, config_path: str):
        self.loadConfig(config_path)
    
    def loadConfig(self, config_path: str):
        with open(config_path, 'r') as f:
            paras = yaml.load(f, Loader = yaml.FullLoader())
            self.underlying_features_K = paras['underlying_features_K']
            self.steps = paras['steps']
            self.lambda_r = paras['lambda_r'] 
            self.learning_rate = paras['learning_rate']
        f.close()
        
    def __str__(self):
        return f'underlying_features_K: {self.underlying_features_K}\nsteps: {self.steps}\nlambda_r: {self.lambda_r}\nlearning_rate: {self.learning_rate}'
    
    def __repr__(self):
        return self.__str__()