#!/usr/bin/env python
#
# @Author: PowersPope
# @About: script to house attention implemented modules
#

# Packages needed
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math


#### Implementation for a single input sequence (no MSA)
class RelativeAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_len: int = 1024,
                 dropout: float = 0.1,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 method: str = "random",
                 ):
        super().__init__()
        """Input for attention and cyclic offset positional encodings

        PARAMETERS
        ----------
        d_model : int
            Dimensionality of the model. In the sense of what is the desired
            out dimension. Also, what is the input dimension size

        num_heads: int
            The number of heads to run

        max_len: int
            Max length sequence. This can be larger then the given dataset, especially
            since the generated encodings are relative

        dropout : float
            The amount of data to dropout

        device: str
            Figures out the cuda or cpu method to use

        method: str
            Specify the method of relative positional encoding desired
        """
        assert d_model % num_heads == 0, "Model Dimensions is not divisible by num-heads"
        self.d_head = d_model // num_heads  # Dimension of each head
        self.max_len = max_len              # Max length of the strings allowed, currently 8
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        # No bias only weight matrix
        self.key = nn.Linear(d_model, d_model, bias=False, device=device)
        self.query = nn.Linear(d_model, d_model, bias=False, device=device)
        self.value = nn.Linear(d_model, d_model, bias=False, device=device)
        
        # Specify the particular relative position method you want to use
        if method == "random":
            # Specify the relative matrices aK and aV
            self.aK = nn.Parameter(
                torch.randn(size=(max_len, max_len, self.d_head)).to(device),
                requires_grad=True
            )
            self.aV = nn.Parameter(
                torch.randn(size=(max_len, max_len, self.d_head)).to(device),
                requires_grad=True
            )
        elif method == "symmetric":
            # First make the symmetric distance matrix
            i = torch.arange(max_len)
            dist = i[:, None] - i[None, :]
            R = torch.abs(dist).to(device)
            # Specify learning embedding functions
            self.aK_embed = nn.Embedding(max_len, self.d_head, device=device)
            self.aV_embed = nn.Embedding(max_len, self.d_head, device=device)
            # generate our relative matrices aK and aV
            self.aK = nn.Parameter(
                self.aK_embed(R),
                requires_grad=True,
            )
            self.aV = nn.Parameter(
                self.aV_embed(R),
                requires_grad=True,
            )
        else:
            # First we make an offset distance matrix
            i = torch.arange(max_len)
            ij = torch.stack((i, i+max_len), dim=-1)
            offset = (i[:, None] - i[None, :]).to(device)
            c_offset = torch.abs(
                ij[:, None, :, None] - ij[None, :, None, :]
            )
            R = torch.tensor(c_offset.numpy().min((2,3)), device=device)
            less_than_idx = R < torch.abs(offset)
            R[less_than_idx] = -R[less_than_idx]
            R = R * torch.sign(offset)
            # Specify learning embedding functions
            self.aK_embed = nn.Embedding(max_len+1, self.d_head, device=device)
            self.aV_embed = nn.Embedding(max_len+1, self.d_head, device=device)
            # generate our relative matrices aK and aV
            self.aK = nn.Parameter(
                self.aK_embed(R + max_len//2),
                requires_grad=True,
            )
            self.aV = nn.Parameter(
                self.aV_embed(R + max_len//2),
                requires_grad=True,
            )

        self.register_buffer(
            "mask",
            (torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)).to(device),
            persistent=True
        )

    def forward(self, x, y=None, set_mask=False, return_attention=False):
        # input x shape (Batch, Length, Dim)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "Sequence length exceeds model capacity"
            )

        # Generate our K, V, Q matrices
        # w.r.t K = XW^(k), Q = XW^(Q), V = XW^(V)
        # Then tranpose K and v, q so that when they multiply QK_t we get L as the output and not D_h
        if y != None:
            k_t  = self.key(y).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1) # (B, H, D_h, L)
            v = self.value(y).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)      # (B, H, L, D_h)
        else:
            k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1) # (B, H, D_h, L)
            v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)      # (B, H, L, D_h)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1,2)           # (B, H, L, D_h)

        # Calculate the edge information
        # Compute Numerator first. Full numerator is Q(K + aK)^{T}
        e_num1 = torch.matmul(q, k_t) # Q * K^T (B, H, L, L)

        # Second Numerator value
        ak_t = self.aK.transpose(1,2).unsqueeze(1) # (L, L, D_h) -> (L, 1, D_h, L)
        q_p = q.permute(2, 0, 1, 3) #(B, H, L, D_h) -> (L, B, H, D_h)
        e_num2 = torch.matmul(q_p, ak_t).permute(1, 2, 0, 3) # (L, B, H, L) -> (B, H, L, L)
        num = e_num1 + e_num2

        # Normalize the numerator by dimension length
        e = num / math.sqrt(self.d_head)

        # Also, compute mask
        if set_mask:
            mask = self.mask[:, :, :seq_len, :seq_len]
            e = e.masked_fill(mask == 0, float("-inf")) # (B, H, L, L)

        # Now we compute the alpha (softmax) attention score
        alpha = nn.functional.softmax(e, dim=-1) #torch.exp(e) / torch.sum( torch.exp(e), dim=1, keepdim=True)

        # Now we can compute the out value z
        z_1 = torch.matmul(
            alpha,
            v,
        )
        z_2 = torch.matmul(
            alpha.permute(2, 0, 1, 3),
            self.aV.unsqueeze(1),
        ).permute(1, 2, 0, 3) #(L, B, H, D) -> (B, H, L, D)
        z = z_1 + z_2

        # Now permute z for final output
        out = z.transpose(1,2).reshape(batch_size, seq_len, -1)

        if return_attention:
            return self.dropout(out), alpha
        else:
            return self.dropout(out)

# Implementation for an MSA Transformer type architecture
class RelativeMSAAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_len: int = 1024,
                 dropout: float = 0.1,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 msa_depth: int = 64,
                 method: str = "random",
                 ):
        super().__init__()
        """Input for attention and cyclic offset positional encodings

        PARAMETERS
        ----------
        d_model : int
            Dimensionality of the model. In the sense of what is the desired
            out dimension. Also, what is the input dimension size

        num_heads: int
            The number of heads to run

        max_len: int
            Max length sequence. This can be larger then the given dataset, especially
            since the generated encodings are relative

        dropout : float
            The amount of data to dropout

        device: str
            Figures out the cuda or cpu method to use

        msa_depth: int
            Amount of sequences within the MSA

        method: str
            Specify the method of relative positional encoding desired
        """
        assert d_model % num_heads == 0, "Model Dimensions is not divisible by num-heads"
        self.d_head = d_model // num_heads  # Dimension of each head
        self.max_len = max_len              # Max length of the strings allowed, currently 8
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.msa_depth = msa_depth
        
        # No bias only weight matrix
        self.key = nn.Linear(d_model, d_model, bias=False, device=device)
        self.query = nn.Linear(d_model, d_model, bias=False, device=device)
        self.value = nn.Linear(d_model, d_model, bias=False, device=device)
        
        # Specify the particular relative position method you want to use
        if method == "random":
            # Specify the relative matrices aK and aV
            self.aK = nn.Parameter(
                torch.randn(size=(max_len, max_len, self.d_head)).to(device),
                requires_grad=True
            )
            self.aV = nn.Parameter(
                torch.randn(size=(max_len, max_len, self.d_head)).to(device),
                requires_grad=True
            )
        elif method == "symmetric":
            # First make the symmetric distance matrix
            i = torch.arange(max_len)
            dist = i[:, None] - i[None, :]
            R = torch.abs(dist).to(device)
            # Specify learning embedding functions
            self.aK_embed = nn.Embedding(max_len, self.d_head, device=device)
            self.aV_embed = nn.Embedding(max_len, self.d_head, device=device)
            # generate our relative matrices aK and aV
            self.aK = nn.Parameter(
                self.aK_embed(R),
                requires_grad=True,
            )
            self.aV = nn.Parameter(
                self.aV_embed(R),
                requires_grad=True,
            )
        else:
            # First we make an offset distance matrix
            i = torch.arange(max_len)
            ij = torch.stack((i, i+max_len), dim=-1)
            offset = (i[:, None] - i[None, :]).to(device)
            c_offset = torch.abs(
                ij[:, None, :, None] - ij[None, :, None, :]
            )
            R = torch.tensor(c_offset.numpy().min((2,3)), device=device)
            less_than_idx = R < torch.abs(offset)
            R[less_than_idx] = -R[less_than_idx]
            R = R * torch.sign(offset)
            # Specify learning embedding functions
            self.aK_embed = nn.Embedding(max_len+1, self.d_head, device=device)
            self.aV_embed = nn.Embedding(max_len+1, self.d_head, device=device)
            # generate our relative matrices aK and aV
            self.aK = nn.Parameter(
                self.aK_embed(R + max_len//2),
                requires_grad=True,
            )
            self.aV = nn.Parameter(
                self.aV_embed(R + max_len//2),
                requires_grad=True,
            )

        self.register_buffer(
            "mask",
            (torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0).unsqueeze(0)).to(device),
            persistent=True
        )

    def forward(self, x, y=None, set_mask=False, return_attention=False):
        # input x shape (Batch, Length, Dim)
        batch_size, MSA, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "Sequence length exceeds model capacity"
            )

        # Generate our K, V, Q matrices
        # w.r.t K = XW^(k), Q = XW^(Q), V = XW^(V)
        # Then tranpose K and v, q so that when they multiply QK_t we get L as the output and not D_h
        if y != None:
            k_t  = self.key(y).reshape(batch_size, MSA, seq_len, self.num_heads, -1).permute(0, 3, 1, 4, 2)     # (B, H, M, D_h, L)
            v = self.value(y).reshape(batch_size, MSA, seq_len, self.num_heads, -1).permute(0, 3, 1, 2, 4)      # (B, H, M, L, D_h)
        else:
            k_t  = self.key(x).reshape(batch_size, MSA, seq_len, self.num_heads, -1).permute(0, 3, 1, 4, 2)     # (B, H, M, D_h, L)
            v = self.value(x).reshape(batch_size, MSA, seq_len, self.num_heads, -1).permute(0, 3, 1, 2, 4)      # (B, H, M, L, D_h)
        q = self.query(x).reshape(batch_size, MSA, seq_len, self.num_heads, -1).permute(0, 3, 1, 2, 4)          # (B, H, M, L, D_h)

        # Calculate the edge information
        # Compute Numerator first. Full numerator is Q(K + aK)^{T}
        e_num1 = torch.matmul(q, k_t) # Q * K^T (B, H, M, L, L)

        # Second Numerator value
        ak_t = self.aK.transpose(1,2).unsqueeze(1).unsqueeze(1) # (L, L, D_h) -> (L, 1, 1, D_h, L)
        q_p = q.permute(3, 0, 1, 2, 4) #(B, H, M, L, D_h) -> (L, B, H, M, D_h)
        e_num2 = torch.matmul(q_p, ak_t).permute(1, 2, 3, 0, 4) # (L, B, H, M, L) -> (B, H, L, L)
        num = e_num1 + e_num2

        # Normalize the numerator by dimension length
        e = num / math.sqrt(self.d_head)

        # Also, compute mask
        if set_mask:
            mask = self.mask[:, :, :, :seq_len, :seq_len]
            e = e.masked_fill(mask == 0, float("-inf")) # (B, H, M, L, L)

        # Now we compute the alpha (softmax) attention score
        alpha = nn.functional.softmax(e, dim=-1) #torch.exp(e) / torch.sum( torch.exp(e), dim=1, keepdim=True)

        # Now we can compute the out value z
        z_1 = torch.matmul(
            alpha,
            v,
        )
        z_2 = torch.matmul(
            alpha.permute(3, 0, 1, 2, 4),  # (B, H, M, L, L) -> (L, B, H, M, L)
            self.aV.unsqueeze(1).unsqueeze(1),  # (B, L, L) -> (B, 1, 1, L, L)
        ).permute(1, 2, 3, 0, 4) #(L, B, H, M, D) -> (B, H, M, L, D)
        z = torch.sum(z_1 + z_2, axis=2)

        # Now permute z for final output
        out = z.transpose(1,2).reshape(batch_size, seq_len, -1)

        if return_attention:
            return self.dropout(out), alpha
        else:
            return self.dropout(out)
