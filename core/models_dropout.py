import torch
import torch.nn as nn
import torch.nn.functional as F
try : 
    from core.layers import Residual_module, New1_layer, New2_layer, New3_layer, Receptive_attention
except : 
    from layers import Residual_module, New1_layer, New2_layer, New3_layer, Receptive_attention
try:
    from core.utils import dropout_without_energy_perserving
except:
    from utils import dropout_without_energy_perserving
class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

def dropout_mask(x, p, training):
    if training:
        return (torch.rand_like(x) > p).float() 
    return torch.ones_like(x)
def apply_dropout(x, p, training):
    return (x * dropout_mask(x, p, training)).detach()
class New_model_dropout(nn.Module):
    def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10, mul = 1, case = 'FBI_Net', output_type='linear', sigmoid_value = 0.1,
                 blind=True, dropout_rate=0.0,dropout_type='input',dropout_RPM_mode=False,rescale_on_eval=False):
        super(New_model_dropout, self).__init__()
        
        self.case = case
        self.dropout_RPM_mode = dropout_RPM_mode
        self.dropout_type=dropout_type
        assert self.dropout_type in ['input','first','middle1','middle2','last'
                                    ,'input_np','first_np','middle1_np','middle2_np','last_np'
                                    , 'input_dropout', 'input_dropout_detach' 
                                     ]
        if 'input_dropout' in self.dropout_type :
            self.dropout_layer = nn.Dropout(p=dropout_rate)
            if dropout_RPM_mode:
                self.dropout_layer = lambda x : apply_dropout(x, dropout_rate, self.training)
        elif '_np' in self.dropout_type:
            self.dropout_layer = lambda x : dropout_without_energy_perserving(x.detach(),dropout_rate)
            self.dropout_type = self.dropout_type[:-3]
        else:
            self.dropout_layer = lambda x : dropout_without_energy_perserving(x,dropout_rate)
        
        self.rescale_on_eval = rescale_on_eval
        self.new1 = New1_layer(channel, filters, mul = mul, case = case, blind=blind).cuda()
        self.new2 = New2_layer(filters, filters, mul = mul, case = case, blind=blind).cuda()
        
        self.num_layers = num_of_layers
        self.output_type = output_type
        self.sigmoid_value = sigmoid_value
        dilated_value = 3
        
        for layer in range (num_of_layers-2):
            self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value, mul = mul, case = case, blind=blind).cuda())
            
        self.residual_module = Residual_module(filters, mul)
        self.activation = nn.PReLU(filters,0).cuda()
        self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
        if self.output_type == 'sigmoid':
            self.sigmoid=nn.Sigmoid().cuda()
        
        self.new = AttrProxy(self, 'new_')

    def forward(self, x):
        if self.dropout_type == 'input' and self.training:
            x = self.dropout_layer(x)
        output, output_new = self.new1(x)
        if self.dropout_type == 'first' and self.training:
            output_new = self.dropout_layer(output_new)
        output_sum = output
        output, output_new = self.new2(output, output_new)
        output_sum = output + output_sum
        for i, (new_layer)  in enumerate(self.new):

            output, output_new  = new_layer(output, output_new)
            output_sum = output + output_sum
            if self.dropout_type == 'middle1' and i == 1 and self.training:
                output_new = self.dropout_layer(output_new)
            if i == self.num_layers - 3:
                break
        if self.dropout_type == 'middle2' and self.training:
            output_sum = self.dropout_layer(output_sum)

        final_output = self.activation(output_sum/self.num_layers)
        final_output = self.residual_module(final_output)
        if self.dropout_type == 'last' and self.training:
            final_output = self.dropout_layer(final_output)
        final_output = self.output_layer(final_output)
            
        if self.output_type=='sigmoid':
               final_output[:,0]=(torch.ones_like(final_output[:,0])*self.sigmoid_value)*self.sigmoid(final_output[:,0])

        return final_output