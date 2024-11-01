from parameter import Parameters

class Main():
    def __init__(self):
        self.config_path = 'config/config.yaml'
    
    def printParameters(self):
        p = Parameters(self.config_path) 
        print(p)

if __name__ == "__main__":
    model = Main()
    model.printParameters()