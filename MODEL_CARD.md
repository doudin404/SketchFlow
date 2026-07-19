# SketchFlow Model Card

## Model

SketchFlow maps CLIP text or image embeddings into the sketch-embedding
distribution learned from QuickDraw, then decodes the transported latent into a
256-point vector sketch.

- Conditioner: GMM-prior optimal-transport conditional flow matching
- Decoder: one-dimensional U-Net/Transformer diffusion model
- Conditioning encoder: OpenCLIP ViT-B/32 with OpenAI weights
- Training data: the 345-category QuickDraw Sketch-RNN full archives
- Output: `(256, 3)` arrays containing `x`, `y`, and pen-state values

## Intended Use

The model is intended for research and creative vector-sketch generation. It is
particularly useful for concise concepts with a strong CLIP representation,
including concepts outside the discrete QuickDraw label vocabulary.

## Limitations

SketchFlow is not a general text-to-image model. Long prompts, multi-object
layouts, detailed attributes, text rendering, and precise spatial relations may
be ignored or projected to a simpler sketch concept. Sampling is stochastic, so
multiple seeds are often useful.

QuickDraw may contain inappropriate or biased samples despite moderation. Model
outputs inherit limitations from QuickDraw and CLIP. Review outputs before using
them in public-facing or sensitive contexts.

## License

The released code and weights use the MIT License. QuickDraw data is not
redistributed and remains under CC BY 4.0.
