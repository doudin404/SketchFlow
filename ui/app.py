"""Gradio interface for SketchFlow."""

from __future__ import annotations

import gradio as gr
from PIL import Image, ImageOps

from ui.inference import InferenceEngine


SELECTION_COLOR = "#de5b43"
SELECTION_WIDTH = 6


def create_app(
    engine: InferenceEngine,
    default_save_dir: str = "ui_results",
) -> gr.Blocks:
    def decorate(image: Image.Image, selected: bool) -> Image.Image:
        if not selected:
            return image
        return ImageOps.expand(
            image,
            border=SELECTION_WIDTH,
            fill=SELECTION_COLOR,
        )

    def selection_outputs(images, selected):
        images = images or []
        selected = sorted(set(selected or []))
        selected_set = set(selected)
        items = [
            (decorate(image, index in selected_set), f"#{index}")
            for index, image in enumerate(images)
        ]
        choices = [str(index) for index in range(len(images))]
        return (
            items,
            selected,
            f"{len(selected)} selected",
            gr.update(
                choices=choices,
                value=[str(index) for index in selected],
            ),
        )

    def generate(
        text,
        image,
        second_text,
        second_image,
        batch_size,
        variance,
        flow_alpha,
        seed_value,
        steps,
        sigma_txt,
    ):
        seed = None if seed_value < 0 else int(seed_value)
        images, strokes = engine.generate(
            text=text,
            image=image,
            B=int(batch_size),
            seed=seed,
            varscale=float(variance),
            second_text=second_text,
            second_image=second_image,
            flow_alpha=float(flow_alpha),
            denoising_steps=int(steps),
            sigma_txt_override=float(sigma_txt),
        )
        gallery, selected, label, checkboxes = selection_outputs(images, [])
        return gallery, strokes, images, selected, label, checkboxes

    def select_gallery(images, selected, event: gr.SelectData):
        index = event.index
        if isinstance(index, (tuple, list)):
            index = index[0]
        try:
            index = int(index)
        except (TypeError, ValueError):
            return selection_outputs(images, selected)
        selected_set = set(selected or [])
        if index in selected_set:
            selected_set.remove(index)
        else:
            selected_set.add(index)
        return selection_outputs(images, sorted(selected_set))

    def change_checkboxes(images, values):
        return selection_outputs(
            images,
            sorted(int(value) for value in (values or [])),
        )

    def select_all(images):
        return selection_outputs(images, list(range(len(images or []))))

    def clear_selection(images):
        return selection_outputs(images, [])

    def save(
        strokes,
        selected,
        save_dir,
        text,
        batch_size,
        seed_value,
        variance,
        flow_alpha,
    ):
        if not selected:
            return "Select at least one result."
        seed = None if seed_value < 0 else int(seed_value)
        return engine.save_strokes(
            raw_strokes=strokes,
            save_dir=save_dir or default_save_dir,
            prompt=text,
            b=int(batch_size),
            seed=seed,
            varscale=float(variance),
            flow_alpha=float(flow_alpha),
            selected_indices=selected,
        )

    theme = gr.themes.Base(
        primary_hue="orange",
        secondary_hue="blue",
        neutral_hue="slate",
    )
    with gr.Blocks(theme=theme, title="SketchFlow", fill_width=True) as app:
        strokes_state = gr.State(None)
        images_state = gr.State([])
        selected_state = gr.State([])

        gr.Markdown("# SketchFlow")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Concept",
                    placeholder="Kirby, Mickey Mouse, ghost, rocket...",
                )
                reference = gr.Image(
                    type="pil",
                    label="Reference image",
                    height=220,
                )
            with gr.Column(scale=1):
                second_prompt = gr.Textbox(
                    label="Interpolation endpoint",
                    placeholder="Optional second concept",
                )
                second_reference = gr.Image(
                    type="pil",
                    label="Second reference image",
                    height=220,
                )

        with gr.Row():
            batch_size = gr.Slider(
                1,
                32,
                value=8,
                step=1,
                label="Samples",
            )
            variance = gr.Slider(
                0,
                2,
                value=1,
                step=0.05,
                label="Variation",
            )
            sigma_txt = gr.Slider(
                0,
                0.05,
                value=0.025,
                step=0.001,
                label="GMM noise",
            )

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                steps = gr.Slider(
                    1,
                    1000,
                    value=60,
                    step=1,
                    label="Denoising steps",
                )
                flow_alpha = gr.Slider(
                    0,
                    1,
                    value=1,
                    step=0.05,
                    label="Flow strength",
                )
                seed_value = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed (-1 is random)",
                )

        generate_button = gr.Button("Generate", variant="primary")
        gallery = gr.Gallery(
            columns=[8],
            object_fit="contain",
            show_label=False,
            allow_preview=False,
        )
        with gr.Row():
            select_all_button = gr.Button("Select all", size="sm")
            clear_button = gr.Button("Clear", size="sm")
            selection_label = gr.Markdown("0 selected")
        selection_checkboxes = gr.CheckboxGroup(
            choices=[],
            label="Selection",
        )

        with gr.Row():
            save_dir = gr.Textbox(
                value=default_save_dir,
                label="Export directory",
                scale=3,
            )
            save_button = gr.Button("Export .npy", scale=1)
        save_status = gr.Textbox(label="Export status", interactive=False)

        selection_targets = [
            gallery,
            selected_state,
            selection_label,
            selection_checkboxes,
        ]
        generate_button.click(
            fn=generate,
            inputs=[
                prompt,
                reference,
                second_prompt,
                second_reference,
                batch_size,
                variance,
                flow_alpha,
                seed_value,
                steps,
                sigma_txt,
            ],
            outputs=[
                gallery,
                strokes_state,
                images_state,
                selected_state,
                selection_label,
                selection_checkboxes,
            ],
        )
        gallery.select(
            fn=select_gallery,
            inputs=[images_state, selected_state],
            outputs=selection_targets,
        )
        selection_checkboxes.change(
            fn=change_checkboxes,
            inputs=[images_state, selection_checkboxes],
            outputs=selection_targets,
        )
        select_all_button.click(
            fn=select_all,
            inputs=[images_state],
            outputs=selection_targets,
        )
        clear_button.click(
            fn=clear_selection,
            inputs=[images_state],
            outputs=selection_targets,
        )
        save_button.click(
            fn=save,
            inputs=[
                strokes_state,
                selected_state,
                save_dir,
                prompt,
                batch_size,
                seed_value,
                variance,
                flow_alpha,
            ],
            outputs=save_status,
        )
    return app
