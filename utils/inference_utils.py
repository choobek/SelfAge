import os
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDIMScheduler

from configs.transforms_config import AgingTransforms
from dataset.inference_dataset import InferenceDataset
from utils.p2p import *
from utils.null_inversion import *


def get_person_placeholder(age=None, predicted_gender=None):
    if age < 5:
        person_placeholder = "baby"
    elif 5 <= age < 15:
        if predicted_gender is None:
            person_placeholder = "child"
        else:
            person_placeholder = ['boy', 'girl'][predicted_gender == 'Female' or predicted_gender == 1]
    elif 15 <= age < 65:
        if predicted_gender is None:
            person_placeholder = "person"
        else:
            person_placeholder = ['man','woman'][predicted_gender == 'Female' or predicted_gender == 1]
    elif age >= 65:
        person_placeholder = "elderly"
    return person_placeholder


def load_dataset(args):
    print(f'Loading dataset')
    transforms_dict = AgingTransforms(args).get_transforms()
    dataset = InferenceDataset(root=args.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=args)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=int(args.test_workers),
                            drop_last=False)
    return dataset, dataloader


def load_diffusers(specialized_path, base_model):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                            clip_sample=False, set_alpha_to_one=False,
                            steps_offset=1)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device=device)

    model = StableDiffusionPipeline.from_pretrained(base_model,
        scheduler=scheduler,
        safety_checker=None).to(device)
    model.load_lora_weights(specialized_path)
    tokenizer = model.tokenizer
    
    return model, generator, tokenizer


def invert(model, image_path, inversion_prompt):
    null_inversion = NullInversion(model)
    
    (_, _), x_t, uncond_embeddings = null_inversion.invert(
        image_path, inversion_prompt, offsets=(0,0,0,0), verbose=True)
    
    return x_t, uncond_embeddings


def prompt_to_prompt(
        inversion_prompt,
        prompts, model,
        generator, tokenizer,
        x_t,
        uncond_embeddings
):
    prompt_before, prompt_after = prompts
    new_prompt = inversion_prompt
    for i in range(len(prompt_before)):
        new_prompt = new_prompt.replace(prompt_before[i], prompt_after[i])
    blend_word = ((prompt_before, prompt_after))
    is_replace_controller = True

    prompts = [inversion_prompt, new_prompt]

    cross_replace_steps = {'default_': .8,}
    self_replace_steps = .5

    eq_params = {"words": (prompt_after[0]), "values": (1,)}

    controller = make_controller(
        prompts,
        is_replace_controller,
        cross_replace_steps,
        self_replace_steps,
        tokenizer,
        blend_word,
        eq_params
    )

    images, _ = p2p_text2image(
        model,
        prompts,
        controller,
        generator=generator.manual_seed(0),
        latent=x_t,
        uncond_embeddings=uncond_embeddings
    )
    
    return images


def save_output(new_img, save_aged_dir, output_img_name):
    new_img_pil = Image.fromarray(new_img)
    new_img_pil.save(os.path.join(save_aged_dir, output_img_name))
