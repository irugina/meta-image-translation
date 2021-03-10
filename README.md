# meta-image-translation

Repository for few-shot image translation applications, in particular SEVIR but also miniimagenet for fast prototyping. 
Dataloaders are borrowed from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the UNet model architecture from [here](https://github.com/milesial/Pytorch-UNet). 

## TODO

- [x] train joint reconstruction
- [ ] train MAML reconstruction
- [ ] train joint GAN
- [ ] train MAML GAN
- [x] eval joint reconstruction
- [ ] eval MAML reconstruction
- [ ] eval joint GAN
- [ ] eval MAML GAN
- [ ] load pretrained encoder
- [ ] miniimagenet

## SEVIR dataset issue:

I am currently manipulating tensors in ```__getitem__``` when sampling from sevir because I needed to preprocess the data: its spec has changed since when I saved SEVIR to disk. I think all intermediate results get moved to CUDA though - so this might be an issue worth addressing, because it might improve speed, the allowed batch size, as well as code simplicity. We should generate SEVIR for few-shot once again and make it match our setting.
