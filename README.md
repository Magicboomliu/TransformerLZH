# TransformerLZH
This is Transformer Repo for beiginers to use: Including Vit, Cross-vit, Swin Transformer and their blocks.  

### 1.1 Swin Tranformer blocks  
This is designed for those directly want to use Swin-Transformer(Local Window + Shift Window Attention) to do feature aggregation.  

* how to use? 
```

import torch
import torch.nn as nn
import sys
sys.path.append("../")
from Transformer.SwinTransformer.MySwinBlocks import MySwinFormerBlocks
'''
Usage Of SwinTransformerBlocks
'''

if __name__=="__main__":
    
    
    feature = torch.randn(3,128,40,80).cuda()
    
    # Example One: 
    # swinformer_blocks = MySwinFormerBlocks(input_feature_channels=128,
    #                                        window_size=7,
    #                                        embedd_dim=128,
    #                                        norm_layer=nn.LayerNorm,
    #                                        block_depths=[2,4],
    #                                        nums_head=[2,4],
    #                                        input_feature_size=(40,80),
    #                                        mlp_ratio=4.0,
    #                                        skiped_patch_embed=True,
    #                                        patch_size=(1,1),
    #                                        use_ape=True,
    #                                        use_prenorm=True,
    #                                        downsample=True,
    #                                        out_indices=(0,1),
    #                                        frozen_stage=-1).cuda()
    # out = swinformer_blocks(feature)
    # torch.Size([3, 128, 40, 80])
    # torch.Size([3, 256, 20, 40])
    
    
    # Example Two : Simple Block

    swinformer_blocks = MySwinFormerBlocks(input_feature_channels=128,
                                           window_size=7,
                                           embedd_dim=128,
                                           norm_layer=nn.LayerNorm,
                                           block_depths=[2],
                                           nums_head=[2],
                                           input_feature_size=(40,80),
                                           mlp_ratio=4.0,
                                           skiped_patch_embed=True,
                                           patch_size=(1,1),
                                           use_ape=True,
                                           use_prenorm=True,
                                           downsample=False,
                                           out_indices=[0],
                                           frozen_stage=-1).cuda()
    out = swinformer_blocks(feature)
    for o in out:
        print(o.shape)
    
```

### 1.2 Simple Vision Transformer (W/O ClS Token) --> Simply for Feature Extraction.  
* Absolute Positional Embedding  
   - Learnable Absolute Positional Embedding  
   - SinCos Positional Embedding 

How to use ? 
```
from Transformer.VIT.vit_ape import ViT

# Define the networks
vit = ViT(image_size=(40,80),patch_size=(1,1),heads=(2,4,4),dim_head=64,depths=3,
              embedd_dim=512,mlp_dim=256,input_channels=128,dropout_rate=0.,emb_dropout=0.,
              ape='sincos1d').cuda()
    
vit(image)
```
THE API: 
```
if self.ape =='learn':
    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.embedd_dim))
elif self.ape =='sincos2d':
    self.pos_embedding = positionalencoding2d(d_model=self.embedd_dim,height=H//patch_H,width=W//patch_W).cuda()
    self.pos_embedding = self.pos_embedding.permute(1,2,0).view(-1,self.embedd_dim).unsqueeze(0)
    self.pos_embedding.requires_grad = False
elif self.ape =='sincos1d':
    self.pos_embedding = positionalencoding1d(d_model=self.embedd_dim,length=num_patches).cuda().unsqueeze(0)
    self.pos_embedding.requires_grad = False
```

* Relative Positional Embedding  
```
from Transformer.VIT.vit_relative import ViT
if __name__=="__main__":
    
    image = torch.randn(1,128,40,80).cuda()
    
    vit = ViT(image_size=(40,80),patch_size=(1,1),heads=(2,4,4),dim_head=64,depths=3,
              embedd_dim=512,mlp_dim=256,input_channels=128,dropout_rate=0.,emb_dropout=0.).cuda()
    
    vit(image)

```

### 1.3 Cross Vision Transformer Feature Extraction. 

```
from Transformer.CrossVit.crossvit_ape import CrossVit

if __name__=="__main__":
    
    feature1 = torch.randn(1,24,40,80).cuda()
    feature2 = torch.randn(1,24,40,80).cuda()
    
    crossvit = CrossVit(image_size=[(40,80),(40,80)],
                        embedd_dim=[24,24],
                        input_dimension=(24,24),
                        patch_size=((1,1),(1,1)),
                        basic_depth=1,
                        cross_attention_dim_head=64,
                        cross_attention_depth=1,
                        cross_attention_head=[4],
                        enc_depths=[1,1],
                        enc_heads=[[4],[4]],
                        enc_head_dim=[64,64],
                        enc_mlp_dims=[128,128],
                        dropout_rate=0.1,
                        emb_dropout=0.1,
                        skiped_patch_embedding=False).cuda()
    
    feat_fusion = crossvit(feature1,feature2)
    
    print(feat_fusion.shape)

```