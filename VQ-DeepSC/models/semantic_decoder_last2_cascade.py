import torch

from .semantic_decoder import SemanticDecoder


class SemanticDecoderLast2Cascade(SemanticDecoder):
    """Four-stage decoder driven only by scales 2 and 3.

    Scale 2 remains the sole encoder-to-decoder skip. At the two shallower
    skip locations, the current decoder feature is concatenated with itself.
    This preserves the original decoder channel dimensions without exposing
    scale-0 or scale-1 encoder features.
    """

    def forward(self, quantized_features):
        if len(quantized_features) != 2:
            raise ValueError(
                "Last-two cascade decoder expects [scale_2, scale_3] features"
            )

        quantized_scale_2, quantized_scale_3 = quantized_features
        if quantized_scale_2.shape[1] != self.embedding_dims[2]:
            raise ValueError(
                f"Scale 2 must have {self.embedding_dims[2]} channels, "
                f"got {quantized_scale_2.shape[1]}"
            )
        if quantized_scale_3.shape[1] != self.embedding_dims[3]:
            raise ValueError(
                f"Scale 3 must have {self.embedding_dims[3]} channels, "
                f"got {quantized_scale_3.shape[1]}"
            )

        x = self.init(quantized_scale_3)

        # The only real U-Net skip: scale 2 at 32x32.
        x = self.up_blocks[0](x)
        if x.shape[-2:] != quantized_scale_2.shape[-2:]:
            raise ValueError(
                "Scale-2 feature resolution does not match the first decoder output"
            )
        x = torch.cat((x, quantized_scale_2), dim=1)

        # Scale-1 skip is absent. Copy the current deep decoder information.
        x = self.up_blocks[1](x)
        x = torch.cat((x, x), dim=1)

        # Scale-0 skip is absent. Copy the current decoder information again.
        x = self.up_blocks[2](x)
        x = torch.cat((x, x), dim=1)

        x = self.up_blocks[3](x)
        return self.final(x)
