import os
import argparse
from pytorch_lightning import seed_everything

import sys
sys.path.append(".")
sys.path.append("..")

from utils import inference_utils as util
from utils.p2p import *
from utils.null_inversion import *
from criteria.aging_loss import AgingLoss
import numpy as np


UNIQUE_TOKEN = "sks"


def run():
    args = parse_args()
    seed_everything(args.seed)
    base_model = args.base_model
    gender = args.gender
    gt_gender = None
    if gender is not None:
        gt_gender = int(gender == 'female')
    personalized_path = args.personalized_path
    target_age = [int(age) for age in args.target_age.split(',')]
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ldm_stable, g_cuda, tokenizer = util.load_diffusers(personalized_path, base_model)
    dataset, dataloader = util.load_dataset(args)
    aging_loss = AgingLoss()
    
    print(f"Running on {os.path.basename(os.path.normpath(args.data_path))}, gender: {gender}...")

    global_i = 0
    for input_batch in tqdm(dataloader):
        for input_img in input_batch:
            print(f"Inverting img no.{global_i}...")
            
            im_path = dataset.paths[global_i]
            img_name = os.path.basename(im_path)
        
            age_init_ = aging_loss.extract_ages(input_img.unsqueeze(0).to(device))
            age_init = age_init_.item()
            
            person_placeholder = util.get_person_placeholder(age_init, gt_gender)
            inversion_prompt = f"photo of {UNIQUE_TOKEN} {person_placeholder} as {age_init}-year-old"
            x_t, uncond_embeddings = util.invert(ldm_stable, input_img, inversion_prompt)
            
            edited_images = []
            for age_new in target_age:
                print(f'Age editing with target age {age_new}...')
                
                new_person_placeholder = util.get_person_placeholder(age_new, gt_gender)
                prompt_before_after = (
                    (f"{age_init}-year-old", person_placeholder),
                    (f"{age_new}-year-old", new_person_placeholder)
                )
                
                image = util.prompt_to_prompt(
                    inversion_prompt, prompt_before_after,
                    ldm_stable, g_cuda, tokenizer,
                    x_t, uncond_embeddings
                )
                image = image[-1]
                
                exp_dir_age = os.path.join(exp_dir, str(age_new))
                os.makedirs(exp_dir_age, exist_ok=True)
                edited_images.append(image)
                util.save_output(image, exp_dir_age, img_name)
            
            if args.side_by_side:
                combined_image = np.hstack(edited_images)
                exp_dir_combined = os.path.join(exp_dir, 'side-by-side')
                os.makedirs(exp_dir_combined, exist_ok=True)
                combined_img_name = f'{os.path.splitext(img_name)[0]}_side-by-side.png'
                util.save_output(combined_image, exp_dir_combined, combined_img_name)
            
            global_i += 1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_model', default='runwayml/stable-diffusion-v1-5', type=str,
                        help='Base model for diffusion')
    parser.add_argument('--data_path', required=True,
                        help='Path to dir with input images')
    parser.add_argument('--gender', choices=["female", "male"], default=None,
                        help='Specify the gender ("female" or "male")')
    parser.add_argument('--personalized_path', required=True,
                        help='Path to personalized diffusion model')
    parser.add_argument('--exp_dir', default='./experiment',
                        help='Path to save outputs')
    parser.add_argument('--target_age', default=None, type=str,
                        help='Target age for inference. Can be comma-separated list for multiple ages.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for seed_everything')
    parser.add_argument('--side_by_side', action='store_true',
                        help='Save outputs side-by-side')
    
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help='Batch size for testing and inference')
    parser.add_argument('--test_workers', default=2, type=int,
                        help='Number of test/inference dataloader workers')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
	run()