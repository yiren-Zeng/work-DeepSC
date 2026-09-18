"""Dual-target-codebook extension of the isolated 1/8 + 1/16 RAQVAE_TWO."""
import random

import torch
import torch.nn.functional as F

from nets.model_ds8 import RAQVAE_TWO_DS8


class RAQVAE_TWO_DS8_DUALK(RAQVAE_TWO_DS8):
    """Generate and quantize bottom/top target codebooks independently.

    The base EMA codebook and all Seq2Seq parameters remain shared. Two Seq2Seq
    calls are always made, including when bottom and top request the same K.
    """

    K_CHOICES = (2, 4, 8, 16, 32, 64)

    @classmethod
    def sample_target_k(cls):
        return random.choice(cls.K_CHOICES)

    @staticmethod
    def target_tensor(k):
        return torch.arange(k).unsqueeze(1)

    def forward(self, x, trg_bottom, trg_top):
        encoded = self.encode(x, trg_bottom, trg_top)
        (quant_src_t, quant_src_b, diff_src, ind_src_t, ind_src_b,
         quant_trg_t, quant_trg_b, diff_trg, ind_trg_t, ind_trg_b) = encoded
        x_hat_src = self.decode(quant_src_t, quant_src_b)
        x_hat_trg = self.decode(quant_trg_t, quant_trg_b)
        return (x_hat_src, diff_src, ind_src_t, ind_src_b,
                x_hat_trg, diff_trg, ind_trg_t, ind_trg_b)

    def encode(self, input, trg_bottom, trg_top):
        enc_b = self.encoder_b(input)
        enc_t = self.encoder_t(enc_b)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_src_b, diff_src_b, ind_src_b = self.quantize(quant_b)
        quant_src_b = quant_src_b.permute(0, 3, 1, 2)
        diff_src_b = diff_src_b.unsqueeze(0)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_src_t, diff_src_t, ind_src_t = self.quantize(quant_t)
        quant_src_t = quant_src_t.permute(0, 3, 1, 2)
        diff_src_t = diff_src_t.unsqueeze(0)

        # Intentionally call the one shared Seq2Seq twice. Do not combine these
        # calls when K_bottom == K_top: each spatial branch owns its target path.
        embed_weight_bottom = self.cbk2cbk(
            self.src.to(self.device), trg_bottom.to(self.device)
        ).squeeze(1)
        embed_weight_top = self.cbk2cbk(
            self.src.to(self.device), trg_top.to(self.device)
        ).squeeze(1)

        quant_trg_b, diff_trg_b, ind_trg_b = self.quantize_cbk(
            quant_b, embed_weight_bottom.to(self.device)
        )
        quant_trg_b = quant_trg_b.permute(0, 3, 1, 2)
        diff_trg_b = diff_trg_b.unsqueeze(0)
        quant_trg_t, diff_trg_t, ind_trg_t = self.quantize_cbk(
            quant_t, embed_weight_top.to(self.device)
        )
        quant_trg_t = quant_trg_t.permute(0, 3, 1, 2)
        diff_trg_t = diff_trg_t.unsqueeze(0)

        return (quant_src_t, quant_src_b, diff_src_b + diff_src_t,
                ind_src_t, ind_src_b, quant_trg_t, quant_trg_b,
                diff_trg_b + diff_trg_t, ind_trg_t, ind_trg_b)

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        k_bottom = self.sample_target_k()
        k_top = self.sample_target_k()
        trg_bottom = self.target_tensor(k_bottom)
        trg_top = self.target_tensor(k_top)
        (x_hat_src, latent_loss_src, _, _, x_hat_trg,
         latent_loss_trg, _, _) = self.forward(x, trg_bottom, trg_top)

        recon_loss = self.criterion(x_hat_src, x) + self.criterion(x_hat_trg, x)
        latent_loss = latent_loss_src + latent_loss_trg
        loss = recon_loss + latent_loss
        self.log('train_recon_loss', recon_loss)
        self.log('train_latent_loss', latent_loss)
        self.log('train_total_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
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

        # Keep the original validation image logging behavior.
        import torchvision
        tensorboard = self.logger.experiment
        tensorboard.add_image(
            'val_source_imnages', torchvision.utils.make_grid(x), self.global_step
        )
        tensorboard.add_image(
            'val_generated_images', torchvision.utils.make_grid(x_hat_trg), self.global_step
        )

    def decode_latent(self, code_t, code_b, trg_bottom, trg_top):
        # As in encode(), these are deliberately two independent generator calls.
        embed_weight_bottom = self.cbk2cbk(
            self.src.to(self.device), trg_bottom.to(self.device)
        ).squeeze(1)
        embed_weight_top = self.cbk2cbk(
            self.src.to(self.device), trg_top.to(self.device)
        ).squeeze(1)
        quant_t = F.embedding(code_t, embed_weight_top).permute(0, 3, 1, 2)
        quant_b = F.embedding(code_b, embed_weight_bottom).permute(0, 3, 1, 2)
        return self.decode(quant_t, quant_b)

