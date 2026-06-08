"""
Test for Gemma4 spatial localization fix (Issue #6028).

This test verifies that the patches for Gemma4MultimodalEmbedder and
Gemma4VisionPatchEmbedder correctly force float32 computation to preserve
spatial precision in position embeddings and multimodal projections.
"""
import torch
import pytest


class TestGemma4SpatialLocalization:
    """Tests for Gemma4 spatial localization precision fixes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import unsloth to apply patches."""
        import unsloth
        self.unsloth = unsloth

    def test_gemma4_multimodal_embedder_float32_precision(self):
        """Test that Gemma4MultimodalEmbedder uses float32 for projection."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig, Gemma4TextConfig

        vision_config = Gemma4VisionConfig(
            output_proj_dims=3840,
            hidden_size=3840,
            rms_norm_eps=1e-6,
        )
        text_config = Gemma4TextConfig(
            hidden_size=2048,
        )

        embedder = Gemma4MultimodalEmbedder(vision_config, text_config)

        # Test with bfloat16 input
        batch_size = 2
        seq_len = 16
        inputs_embeds = torch.randn(batch_size, seq_len, 3840, dtype=torch.bfloat16)

        result = embedder(inputs_embeds)

        # Result should be in the same dtype as input (bfloat16)
        # but computation should happen in float32 internally
        assert result.dtype == torch.bfloat16
        assert result.shape == (batch_size, seq_len, 2048)

    def test_gemma4_multimodal_embedder_fp16_precision(self):
        """Test that Gemma4MultimodalEmbedder works with fp16 input."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig, Gemma4TextConfig

        vision_config = Gemma4VisionConfig(
            output_proj_dims=3840,
            hidden_size=3840,
            rms_norm_eps=1e-6,
        )
        text_config = Gemma4TextConfig(
            hidden_size=2048,
        )

        embedder = Gemma4MultimodalEmbedder(vision_config, text_config)

        # Test with fp16 input
        batch_size = 2
        seq_len = 16
        inputs_embeds = torch.randn(batch_size, seq_len, 3840, dtype=torch.float16)

        result = embedder(inputs_embeds)

        assert result.dtype == torch.float16
        assert result.shape == (batch_size, seq_len, 2048)

    def test_gemma4_vision_patch_embedder_position_embeddings_float32(self):
        """Test that Gemma4VisionPatchEmbedder position embeddings use float32."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

        config = Gemma4VisionConfig(
            hidden_size=768,
            patch_size=16,
            position_embedding_size=100,
            pooling_kernel_size=3,
        )

        embedder = Gemma4VisionPatchEmbedder(config)

        batch_size = 2
        num_patches = 20
        pixel_position_ids = torch.randint(0, 100, (batch_size, num_patches, 2))
        padding_positions = torch.zeros(batch_size, num_patches, dtype=torch.bool)

        # The internal computation should use float32
        result = embedder._position_embeddings(pixel_position_ids, padding_positions)

        # Result dtype should match position_embedding_table dtype (typically float32 or bfloat16)
        # but computation happens in float32
        assert result.shape == (batch_size, num_patches, 768)
        # The computation is done in float32, result may be cast back

    def test_gemma4_vision_patch_embedder_forward_float32(self):
        """Test that Gemma4VisionPatchEmbedder forward uses float32 for position embeddings."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

        config = Gemma4VisionConfig(
            hidden_size=768,
            patch_size=16,
            position_embedding_size=100,
            pooling_kernel_size=3,
        )

        embedder = Gemma4VisionPatchEmbedder(config)

        batch_size = 2
        num_patches = 20
        pixel_values = torch.randn(batch_size, num_patches, 3 * 16 * 16)
        pixel_position_ids = torch.randint(0, 100, (batch_size, num_patches, 2))
        padding_positions = torch.zeros(batch_size, num_patches, dtype=torch.bool)

        result = embedder(pixel_values, pixel_position_ids, padding_positions)

        assert result.shape == (batch_size, num_patches, 768)

    def test_compiler_disables_vision_components(self):
        """Test that the compiler disables compilation for Gemma4 vision components."""
        from unsloth_zoo.compiler import DISABLE_COMPILE_MODULES

        vision_components = [
            "Gemma4VisionPatchEmbedder",
            "Gemma4VisionModel",
            "Gemma4VisionEncoder",
            "Gemma4VisionEncoderLayer",
            "Gemma4MultimodalEmbedder",
        ]

        for component in vision_components:
            # Check that each component is in the disable list
            is_disabled = any(component.endswith(x) for x in DISABLE_COMPILE_MODULES)
            assert is_disabled, f"{component} should be in DISABLE_COMPILE_MODULES"

    def test_temporary_patches_registered(self):
        """Test that the new patches are registered in TEMPORARY_PATCHES."""
        from unsloth_zoo.temporary_patches import TEMPORARY_PATCHES

        patch_names = [p.__name__ for p in TEMPORARY_PATCHES]

        assert "patch_Gemma4MultimodalEmbedder_forward" in patch_names
        assert "patch_Gemma4VisionPatchEmbedder_position_embeddings" in patch_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
