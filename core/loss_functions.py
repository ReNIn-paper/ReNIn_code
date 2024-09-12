import torch
import torch.nn as nn
import sys

def mse_bias(output, target):
    
    b = output[:,0]
    Z = target[:,0]
    
    # E[(X - b)**2]
    loss = torch.mean((b - Z)**2)
    return loss

def estimated_bias(output, target):
    
    Z = target[:,0]
    b = output
    sigma = target[:,1]
    
    # E[(Z - b)**2 - sigma**2]
    loss = torch.mean((Z - b)**2 - sigma**2)
    
    return loss

def mse_affine(output, target):
    
    a = output[:,0]
    b = output[:,1]
    Z = target[:,0]
    X = target[:,1]
    
    # E[(X - (aZ+b))**2]
    x_hat = (a*Z+b)
    loss = torch.mean((X - x_hat)**2)
    return loss

def emse_affine(output, target):
    
    a = output[:,0]
    b = output[:,1]
    
    Z = target[:,0]
    sigma = target[:,1]
    # E[(Z - (aZ+b))**2 + 2a(sigma**2) - sigma**2]
    loss = torch.mean((Z - (a*Z+b))**2 + 2*a*(sigma**2) - sigma**2)
    
    return loss

# deprecated 
def total_variation_loss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


def mse_affine_with_tv(output,target,lambda_val=5):
    # print(output.shape, target.shape)
    a = output[:,0]
    b = output[:,1]
    Z = target[:,0]
    X = target[:,1]
    # print("in loss function, lambda val ",lambda_val)
    # sys.exit(0)
    # E[(X - (aZ+b))**2]
    x_hat = (a*Z+b)
    loss = torch.mean((X - x_hat)**2)
    # regularize gradient of x_hat
    # print(x_hat.shape)
    regularizer = torch.mean(torch.abs(x_hat[:,:,1:]-x_hat[:,:,:-1])) + torch.mean(torch.abs(x_hat[:,1:,:]-x_hat[:,:-1,:]))
    # print(loss, lambda_val *regularizer)
    loss = loss + lambda_val*regularizer
    return loss
