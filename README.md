# meta-image-translation

Repository for few-shot image translation applications, in particular SEVIR but also miniimagenet for fast prototyping. 
Dataloaders and patch discriminator are borrowed from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), the UNet model architecture from [here](https://github.com/milesial/Pytorch-UNet), and MAML gradient calculations from [here](https://github.com/learnables/learn2learn).

## TODO

- [x] improve bookkeping: create results dir; use os.path.join()
- [x] train joint reconstruction
- [x] train MAML reconstruction
- [x] train joint GAN
- [x] train MAML GAN
- [x] eval joint reconstruction
- [x] eval MAML reconstruction
- [ ] eval joint GAN
- [ ] eval MAML GAN
- [x] load pretrained encoder
- [x] save opt to disk
- [x] sweep number_inner_steps for sevir
- [ ] add miniimagenet dataloader
- [ ] redo SEVIR dataset
- [ ] test on MAML (e.g. implement inner loop in eval\_unet)
- [ ] test contrastive pretrainig after limiting train set
- [ ] MAML + pretraining
- [ ] test pretrained backbones from multiple epochs (?)
