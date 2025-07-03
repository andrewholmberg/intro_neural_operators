import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
# create a multilayer perceptron
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
class Net(torch.nn.Module):

    def __init__(self,spatial_dim, hidden_structure:list):
        super(Net, self).__init__()
        assert len(hidden_structure)>=2

        self.spatial_dim = spatial_dim
        self.hidden_structure= hidden_structure
        self.hidden = torch.nn.ModuleList()        
        self.hidden.append(torch.nn.Linear(spatial_dim+1,hidden_structure[0]))
        for i in range(1,len(hidden_structure)):
            self.hidden.append(torch.nn.Linear(hidden_structure[i-1], hidden_structure[i]))
        self.hidden.append(torch.nn.Linear(hidden_structure[-1],1))
        self.apply(init_weights)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.001)
        self.sigmoid = torch.nn.Sigmoid()
        
        

    def forward(self, input_tensor):
        #t is the last column in the input tensor
        t = input_tensor[:,0:1]
    
        xt = self.leaky_relu(self.hidden[0](input_tensor))
        for i in range(1,len(self.hidden)-2):
            xt = xt + self.leaky_relu(self.hidden[i](xt))
        xt = xt + self.tanh(self.hidden[-2](xt))
        xt = self.hidden[-1](xt)

        return xt