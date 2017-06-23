# UCSB ArchLab OpenTPU Project

The OpenTPU project provides a programmable framework for creating machine learning acceleration hardware.

This project is a an attempt at an open-source version of Google's Tensor Processing Unit (TPU). The TPU is Google's custom ASIC for accelerating the inference phase of neural network computations.

We used details from Google's paper titled "In-Datacentre Performance Analysis of a Tensor Processing Unit" (https://arxiv.org/abs/1704.04760) which is to appear at ISCA2017.

#### The OpenTPU is powered by PyRTL (http://ucsbarchlab.github.io/PyRTL/).

## FAQs:
### What are the parts of the OpenTPU programmable hardware?
#### The OpenTPU hardware consists of:
- A weight FIFO
- A matrix multiply unit
- An activation unit (with ReLU and sigmoid activation functions)
- An accumulator
- A unified buffer

### What does the OpenTPU do?
#### This hardware is able to support:
- Inference phase of neural network computations

### What CAN'T the OpenTPU do?
#### The OpenTPU does not support:
- Convolution operations

### Does the OpenTPU implement all the instructions from the paper?
#### No, the OpenTPU currently supports the following instructions only:
- RHM
- WHM
- RW
- MMC
- ACT
- NOP
- HLT

### I'm a Distinguished Hardware Engineer at Google and the Lead Architect of the TPU. I see many inefficiencies in your implementation.
Hi Norm! Tim welcomes you to Santa Barbara to talk about all things TPU :)
