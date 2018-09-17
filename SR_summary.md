# Super resolution paper summary

### Cascaded SR-GAN for Scale-Adaptive Low Resolution Person Re-identification (IJCAI 2018)
paper url : [https://www.ijcai.org/proceedings/2018/0541.pdf](https://www.ijcai.org/proceedings/2018/0541.pdf)




## Video super resolution
### Frame-Recurrnt Video Super-Resolution (2018)
paper url : [https://arxiv.org/pdf/1801.04590.pdf](https://arxiv.org/pdf/1801.04590.pdf)

summary : Current state-of-the-art methods process a batch of LR frames to generate a single high-resolution (HR) frame and run this scheme in a sliding window fashion over the entire video, effectively treating the problem as a large number of separate multi-frame super-resolution tasks. This approach has two main weaknesses: 1) Each input frame is processed and warped multiple  times, increasing  the  computational
cost, and 2) each output frame is estimated independently conditioned on the input frames, limiting the system’s ability to produce temporally consistent results. This work proposes an end-to-end trainable frame-recurrent video super-resolution framework that uses the previously inferred HR estimate to super-resolve the subsequent frame. This naturally encourages temporally consistent results and reduces the computational cost by warping only one image in each step. Furthermore, due to its recurrent nature, the proposed method has the ability to assimilate a large number of previous frames without increased computational demands.

The latest state-of-the-art video super-resolution methods approach the problem by combining a batch of LR frames to estimate a single HR frame, effectively dividing the task of video super-resolution into a large number of separate multi-frame super-resolution subtasks. However, this approach is computationally expensive since each input frame needs to be processed several times. Furthermore, generating each output frame separately reduces the system’s ability to produce temporally consistent frames, resulting in unpleasing flickering artifacts.


### Video Super-Resolution With Convolutional Neural Networks (2015)


# Author
Dohyun Kim