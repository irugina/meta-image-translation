# meta-image-translation

Repository for few-shot image translation applications, in particular SEVIR but also miniimagenet for fast prototyping. 
Dataloaders are borrowed from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the UNet model architecture from [here](https://github.com/milesial/Pytorch-UNet). 

## TODO

- [x] train joint reconstruction
- [x] train MAML reconstruction
- [ ] train joint GAN
- [ ] train MAML GAN
- [x] eval joint reconstruction
- [x] eval MAML reconstruction
- [ ] eval joint GAN
- [ ] eval MAML GAN
- [ ] load pretrained encoder
- [ ] save opt to disk
- [ ] sweep number_inner_steps for sevir
- [ ] add miniimagenet dataloader
- [ ] redo SEVIR dataset
