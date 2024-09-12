import torch
import torch.nn as nn
import torch.nn.functional as F
try : 
    from core.layers import Residual_module, New1_layer, New2_layer, New3_layer, Receptive_attention
except : 
    from layers import Residual_module, New1_layer, New2_layer, New3_layer, Receptive_attention
class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class New_model(nn.Module):
    def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10, mul = 1, case = 'FBI_Net', output_type='linear', sigmoid_value = 0.1,
                 blind=True, input_dropout=False, dropout_rate=0.0,dropout_RPM_mode=False,rescale_on_eval=False):
        super(New_model, self).__init__()
        
        self.case = case
        self.input_dropout = None
        if input_dropout:
            self.input_dropout = nn.Dropout(dropout_rate).cuda()
            self.dropout_rate = dropout_rate
            self.dropout_RPM_mode = dropout_RPM_mode
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
        if self.input_dropout: 
            x = self.input_dropout(x)
            # print("1.",x[0,0,:5,:5])
            if self.dropout_RPM_mode and self.training == True:
                # print("RPM mode")
                # re-rescale input
                x *= (1-self.dropout_rate)
            if self.rescale_on_eval and self.training == False: # on evaluation time
                # print("rescale on eval")
                x *= (self.dropout_rate)
            # print("2.",x[0,0,:5,:5])
        output, output_new = self.new1(x)
        output_sum = output
        output, output_new = self.new2(output, output_new)
        output_sum = output + output_sum

        for i, (new_layer)  in enumerate(self.new):

            output, output_new  = new_layer(output, output_new)
            output_sum = output + output_sum

            if i == self.num_layers - 3:
                break

        final_output = self.activation(output_sum/self.num_layers)
        final_output = self.residual_module(final_output)
        final_output = self.output_layer(final_output)
            
        if self.output_type=='sigmoid':
               final_output[:,0]=(torch.ones_like(final_output[:,0])*self.sigmoid_value)*self.sigmoid(final_output[:,0])

        return final_output

class New_model_tiny(nn.Module):
    def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10, mul = 1, case = 'FBI_Net', output_type='linear', sigmoid_value = 0.1,
                 blind=True, input_dropout=False, dropout_rate=0.0,dropout_RPM_mode=False,rescale_on_eval=False):
        super(New_model_tiny, self).__init__()
        
        self.case = case
        self.input_dropout = None
        if input_dropout:
            self.input_dropout = nn.Dropout(dropout_rate).cuda()
            self.dropout_rate = dropout_rate
            self.dropout_RPM_mode = dropout_RPM_mode
            self.rescale_on_eval = rescale_on_eval
        self.new1 = New1_layer(channel, filters, mul = mul, case = case, blind=blind).cuda()
        
        self.num_layers = num_of_layers
        self.output_type = output_type
        self.sigmoid_value = sigmoid_value
        
        self.activation = nn.PReLU(filters,0).cuda()
        self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
        if self.output_type == 'sigmoid':
            self.sigmoid=nn.Sigmoid().cuda()
        
        self.new = AttrProxy(self, 'new_')

    def forward(self, x):
        if self.input_dropout:
            # print("0.",x[0,0,:5,:5])
            # print("input dropout")
            x = self.input_dropout(x)
            # print("1.",x[0,0,:5,:5])
            if self.dropout_RPM_mode and self.training == True:
                # print("RPM mode")
                # re-rescale input
                x *= (1-self.dropout_rate)
            if self.rescale_on_eval and self.training == False: # on evaluation time
                # print("rescale on eval")
                x *= (self.dropout_rate)
            # print("2.",x[0,0,:5,:5])
        output, output_new = self.new1(x)

        final_output = self.activation(output_new)
        final_output = final_output + output
        final_output = self.output_layer(final_output)
            
        if self.output_type=='sigmoid':
               final_output[:,0]=(torch.ones_like(final_output[:,0])*self.sigmoid_value)*self.sigmoid(final_output[:,0])

        return final_output
if __name__ == '__main__':
    # def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10, 
    #  mul = 1, case = 'FBI_Net', output_type='linear', sigmoid_value = 0.1,blind=True)
    model = New_model(channel = 1, output_channel = 1, filters = 64, num_of_layers=17, 
     mul = 1, case = 'FBI_Net', output_type='linear', sigmoid_value = 0.1,blind=True,
     input_dropout=True, dropout_rate=0.6,
     dropout_RPM_mode=True,rescale_on_eval=False).cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    tmp_img = torch.ones(1,1,256,256).cuda()
    print("========> training mode")
    out = model(tmp_img)
    print(out.shape)
    print("========>  eval mode")
    model.eval()
    out1 = model(tmp_img)
    print(out1.shape)