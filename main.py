from dataset import *
from run import *
from model import *

class Config():
    def __init__(self):
        self.filename = 'qa1_single-supporting-fact_train.txt'
        #self.filename = 'qa2_two-supporting-facts_train.txt' 
        #self.filename = 'qa4_two-arg-relations_train.txt'
        #self.filename = 'qa5_three-arg-relations_train.txt'
        #self.filename = 'qa6_yes-no-questions_train.txt' 
        
        self.mode = 'train'
        self.more_train = False 
        self.train_mode = 'G'
        self.m_mode = 'GRU'
        self.load_model = 'H1000_GRUBatch4A%s'%self.filename[:4] 
        self.save_model = 'H1000_GRUBatch4A%s'%self.filename[:4] 
        self.train_embed = False

        self.epoch = 5 
        self.lr = 1e-5
        
        self.data_dir = './dataset2'
        self.glove_dim = 300
        
        self.shareH_dim = 1000 
        self.gW_dim = 1000
        self.num_layer = 1 
        self.batch_size = 5 
        

def main():
    config = Config()
    
    wordembed = pd.DataFrame.from_csv('wordembed.csv',header=None)
    pre_trained = np.array(wordembed)

    # Tokenize raw data and create list for [i,q,a] pair
    Pairs = GetRawData(config.data_dir,config.filename)
    
    model = DMNmodel(config)
    model.cuda()

    if 'tr' in config.mode:
        if config.more_train:
            model.load_state_dict(torch.load(
                '/home/raehyun/github/DMN/model/%s'%config.load_model))
            print('Model Loaded --- %s'%config.load_model)
        else:
            print('Train Model From Scratch')
       
        print(model)
        for i,param in enumerate(model.parameters()):
            print(param.data.shape)

        print('Start Training on %s'%config.filename)
        run_train(Pairs,wordembed,model,config)
    
    if 'te' in config.mode:
        model.eval()
        model.load_state_dict(torch.load(
            '/home/raehyun/github/DMN/model/%s'%config.load_model))
        print('Model Loaded -- %s'%config.load_model)
        
        run_test(Pairs,wordembed,model,config)

if __name__ == '__main__':
    main()
