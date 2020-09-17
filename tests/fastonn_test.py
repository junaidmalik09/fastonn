from fastonn import OpNetwork,utils,OpTier,OpBlock,Trainer,SelfONNLayer
from fastonn.osl import *
import numpy as np
import torch

def test_configure_oplib():
    nodal = [mul]
    pool = [summ]
    act = [tanh]
    try:
        OPLIB = getOPLIB(nodal,pool,act)
    except:
        raise 

def test_OpNetwork():
    try:
        nodal = [mul,cubic,sine,expp,sinh,chirp]
        pool = [summ,medd,maxx]
        act = [tanh,lincut]
        OPLIB = getOPLIB(nodal,pool,act)
        
        num_layers = np.random.randint(1,4)
        in_channels = np.random.randint(1,6)
        
        tier_sizes = np.random.randint(1,12,size=(num_layers)).tolist()
        kernel_sizes = np.random.randint(3,7,size=(num_layers)).tolist()
        sampling_factors = np.random.choice([-2,-1,1,2],size=(num_layers)).tolist()
        operators = [np.random.randint(len(OPLIB),size=(tier_sizes[i])).tolist() for i in range(num_layers)]
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = OpNetwork(in_channels,tier_sizes,kernel_sizes,operators,sampling_factors,OPLIB).to(device)
        return model
    except:
        raise

def test_SelfONNLayer():
    try:
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(0)
        inputt = torch.randn(4,1,60,60).to(device)
        model = torch.nn.Sequential(
            SelfONNLayer(1,16,21,padding=0,q=3),
            torch.nn.Tanh(),
            SelfONNLayer(16,16,21,padding=0,q=3),
            torch.nn.Tanh(),
            SelfONNLayer(16,1,20,padding=0,q=3),
            torch.nn.Tanh()
        ).to(device)
        output = model(inputt)
        assert output.shape == torch.Size([4, 1, 1, 1])
        assert torch.all(((output.cpu().flatten().data)*1000).round() == torch.tensor([ 153., -459., -108., -569.])).item()
    except:
        raise

def test_trainer():
    try:
        model = test_OpNetwork()
        train_size = val_size = test_size = 8
        batch_size = 4
        in_channels = model.in_channels
        in_size = 16
        out_size = model(torch.randn(1,in_channels,in_size,in_size).to(model.oper[0].oper[0].weights.device)).shape
        trainset = [(torch.randn(in_channels,in_size,in_size),torch.randn(out_size[1],out_size[2],out_size[3])) for _ in range(train_size)]
        valset = [(torch.randn(in_channels,in_size,in_size),torch.randn(out_size[1],out_size[2],out_size[3])) for _ in range(val_size)]
        testset = [(torch.randn(in_channels,in_size,in_size),torch.randn(out_size[1],out_size[2],out_size[3])) for _ in range(test_size)]
        train_dl = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
        val_dl = torch.utils.data.DataLoader(valset,batch_size=batch_size,shuffle=False)
        test_dl = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False) 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        optim_name = np.random.choice(['cgd','vanilla_adam','adamfast','sgd_momentum','sgd'])
        trainer = Trainer(model,train_dl,val_dl,test_dl,torch.nn.MSELoss(),optim_name,1e-3,{'snr':(utils.calc_snr,'max')},device,model.reset_parameters,"test")
        trainer.train(num_epochs=5,num_runs=3)
    except:
        raise


if __name__ == "__main__":
    test_SelfONNLayer()
    test_OpNetwork()
    test_trainer()
    