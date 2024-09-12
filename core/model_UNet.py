import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision.models import inception_v3
from torchvision import models
# from model_dcgan import Generator_DC_GAN, Discriminator_DC_GAN

# UNet style generator
class UNet(nn.Module):
    def __init__(self, dim,output_channel=1, ngf_factor = 1,depth=4):
        """
            dim : number of channels in the first layer (colors : 3, grayscale : 1)
        """
        super(UNet, self).__init__()
        self.dim = dim
        self.image_to_features = []
        self.depth = depth
        self.first_1x1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim * ngf_factor, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(self.dim*ngf_factor)
            )
        for i in range(depth):
            mul = 2**i * ngf_factor
            next_mul = 2**(i+1) * ngf_factor
            self.image_to_features.append(nn.Conv2d(mul * self.dim, next_mul * self.dim, kernel_size=4, stride=2,padding=1))
            # self.image_to_features.append(nn.LeakyReLU(0.2))
            self.image_to_features.append(nn.ReLU())
            self.image_to_features.append(nn.BatchNorm2d(next_mul * self.dim))
        self.image_to_features = torch.nn.ParameterList(self.image_to_features)

        self.features_to_image = []
        for i in range(depth):
            mul = 2**(depth-i) * ngf_factor
            next_mul = 2**(depth-1-i) * ngf_factor
            self.features_to_image.append(nn.ConvTranspose2d(mul * self.dim, next_mul * self.dim, kernel_size=4, stride=2, padding=1))
            # self.features_to_image.append(nn.LeakyReLU(0.2))
            self.features_to_image.append(nn.ReLU())
            self.features_to_image.append(nn.BatchNorm2d(next_mul* self.dim))
        self.features_to_image = torch.nn.ParameterList(self.features_to_image)
        self.last_1x1 = nn.Conv2d(next_mul*self.dim, output_channel, kernel_size=1, stride=1, padding=0)
        self.activation = nn.Tanh()
    def forward(self, input_data):
        # UNet style generator
        features = []
        x = self.first_1x1(input_data)
        for layer in self.image_to_features:
            if isinstance(layer, nn.Conv2d):
                features.append(x)
            # print(layer,x.size())
            x = layer(x)
        # Return generated image
        for layer in self.features_to_image:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d) :
                # print(layer,x.size(),features[-1].size())
                x = x + features.pop()
        x = self.last_1x1(x)
        x = self.activation(x)
        return x

class Generator_wgan_gp(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator_wgan_gp, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (self.img_size[0] // 16, self.img_size[1] // 16)
        
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, (8 * dim * self.feature_sizes[0] * self.feature_sizes[1])),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

class Discriminator(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size
        
        self.image_to_features_dis = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * dim * (img_size[0] // 16) * (img_size[1] // 16)
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )
    def get_activations(self,input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features_dis(input_data.cuda())
        x = x.view(batch_size, -1)
        return x
    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features_dis(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)

class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(weights='Inception_V3_Weights.DEFAULT').cuda()
        # for p in self.inception_network.named_parameters():
        #     print(p[0])
        # self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.layers = nn.Sequential()
        for idx, layer in enumerate(self.inception_network.children()):
            # print(idx,layer)
            if layer == self.inception_network.Mixed_5d:
                break
            else :
                self.layers.add_module(str(idx), layer)
        self.transform_input = transform_input

    # def output_hook(self, module, input, output):
    #     # N x 2048 x 8 x 8
    #     self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 224, 224) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        # self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        # activations = self.mixed_7c_output
        activations = self.layers(x)
        # print(activations.shape)
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        # print(activations.shape)

        activations = activations.view(x.shape[0], 288)
        # activations = activations.view(x.shape[0], 2048)
        return activations
    
    def get_activations(self,images, batch_size=None):
        """
        Calculates activations for last pool layer for all iamges
        --
            Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
            batch size: batch size used for inception network
        --
        Returns: np array shape: (N, 2048), dtype: np.float32
        """
        # print(type(images))
        num_images = images.shape[0]
        if batch_size is None:
            batch_size = num_images
        self.eval()
        n_batches = int(np.ceil(num_images  / batch_size))
        inception_activations = torch.zeros((num_images, 288), dtype=torch.float32)
        for batch_idx in range(n_batches):
            start_idx = batch_size * batch_idx
            end_idx = batch_size * (batch_idx + 1)

            ims = images[start_idx:end_idx].cuda()
            # print(ims.shape)
            activations = self.forward(ims)
            activations = activations # .detach().cpu().numpy()
            assert activations.shape == (ims.shape[0], 288), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 2048), activations.shape)
            inception_activations[start_idx:end_idx, :] = activations
        return inception_activations
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    

if __name__ == '__main__':
    print_param = False
    inceptionNetwork = PartialInceptionNetwork()
    input_size = (64,64,3)  # (256,256,3)
    # Test generator
    gen = Generator(3,7)
    
    # gen = Generator_wgan_gp((256,256,3),100,3)
    # gen = Generator_wgan_gp(input_size,100,3)
    
    
    random_img = torch.randn(1, input_size[2], input_size[0], input_size[1])
    print("img size : ",random_img.size())
    fake_out = gen(random_img)
    print("fake_img : ",fake_out.size())
    
    dis = Discriminator(input_size,1)
    # dis = Discriminator_DC_GAN(3,64)
    if print_param is True:
        print("===== Generator")
        for p in gen.named_parameters(): 
            print(p[0],p[1].requires_grad)
        print("==== Discriminator")
        for p in dis.named_parameters():
            print(p[0],p[1].requires_grad)
    out = dis(fake_out)
    
    print("discriminator decision :",out)
    
