# meta-image-translation

Repository for few-shot image translation applications, in particular SEVIR but also miniimagenet for fast prototyping. 
Dataloaders and patch discriminator are borrowed from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), the UNet model architecture from [here](https://github.com/milesial/Pytorch-UNet), and MAML gradient calculations from [here](https://github.com/learnables/learn2learn).

## TODO

- [ ] test pretrained backbones from multiple epochs
- [ ] test new pretraining
- [ ] do not adapt discriminator
- [ ] plot discriminator losses throughout training
- [x] improve bookkeping: create results dir; use os.path.join()
- [x] train joint reconstruction
- [x] train MAML reconstruction
- [x] train joint GAN
- [x] train MAML GAN
- [x] eval joint reconstruction
- [x] eval MAML reconstruction
- [x] eval joint GAN
- [x] eval MAML GAN
- [x] load pretrained encoder
- [x] save opt to disk
- [x] sweep number_inner_steps for sevir
- [x] test on MAML (e.g. implement inner loop in eval\_unet)
- [x] test contrastive pretraining after limiting train set

## Lower priority
- [ ] add miniimagenet dataloader
- [ ] MAML + pretraining
- [x] redo SEVIR dataset
