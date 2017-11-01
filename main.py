from dataset import *
from run import *
from model import *
class Config():
    def __init__(self):
        self.data_dir = './dataset'
        self.filename = 'qa3_three-supporting-facts_train.txt'
        self.glove_dim = 100
        
        self.hidden_dim = 300
        self.num_layer = 1
        self.batch_size = 5
        self.memory_depth = 5


def main():
    config = Config()
    
    # Tokenize raw data and create list for [i,q,a] pair
    Pairs = GetRawData(config.data_dir,config.filename)
    print(len(Pairs))
    
    glove = GloveVector(config.glove_dim)
    
    model = DMNmodel(config)
    run_train(Pairs,glove,model,config)

if __name__ == '__main__':
    main()
