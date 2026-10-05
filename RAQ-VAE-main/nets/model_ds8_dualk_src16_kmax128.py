"""Source-K=16, target-K<=128 variant of the dual-K DS8 RAQ-VAE."""

from nets.model_ds8_dualk import RAQVAE_TWO_DS8_DUALK


class RAQVAE_TWO_DS8_DUALK_SRC16_KMAX128(RAQVAE_TWO_DS8_DUALK):
    """Keep the dual-K architecture while extending trained target sizes to 128.

    The source codebook size is supplied by ``args.num_embeddings``.  The
    dedicated training entry point for this class fixes that value at 16 and
    fixes ``args.num_embeddings_max`` at 128.
    """

    K_CHOICES = (2, 4, 8, 16, 32, 64, 128)

    def validation_step(self, val_batch, batch_idx):
        """Keep validation metrics unchanged while bounding TensorBoard growth.

        The inherited implementation writes two 64-image grids for every
        validation batch.  Logging only the first batch preserves a stable
        visual sample and prevents multi-gigabyte event files.
        """
        x = val_batch[0]
        k_bottom = self.sample_target_k()
        k_top = self.sample_target_k()
        trg_bottom = self.target_tensor(k_bottom)
        trg_top = self.target_tensor(k_top)
        (x_hat_src, _, _, _, x_hat_trg, _, _, _) = self.forward(
            x, trg_bottom, trg_top
        )

        recon_loss_src = self.criterion(x_hat_src, x)
        recon_loss_trg = self.criterion(x_hat_trg, x)
        self.log('val_recon_loss', recon_loss_src + recon_loss_trg)
        self.log('val_recon_loss(src)', recon_loss_src)
        self.log('val_recon_loss(trg)', recon_loss_trg)

        if batch_idx == 0:
            import torchvision
            tensorboard = self.logger.experiment
            tensorboard.add_image(
                'val_source_imnages',
                torchvision.utils.make_grid(x),
                self.global_step,
            )
            tensorboard.add_image(
                'val_generated_images',
                torchvision.utils.make_grid(x_hat_trg),
                self.global_step,
            )
