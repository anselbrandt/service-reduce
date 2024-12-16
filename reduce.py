import os

from safetensors.torch import save_file
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_and_save_ema_model(
    checkpoint_path: str, new_checkpoint_path: str, safetensors: bool
) -> str:
    try:
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location=torch.device(device)
        )
        print("Original Checkpoint Keys:", checkpoint.keys())

        ema_model_state_dict = checkpoint.get("ema_model_state_dict", None)
        if ema_model_state_dict is None:
            return "No 'ema_model_state_dict' found in the checkpoint."

        if safetensors:
            new_checkpoint_path = new_checkpoint_path.replace(".pt", ".safetensors")
            save_file(ema_model_state_dict, new_checkpoint_path)
        else:
            new_checkpoint_path = new_checkpoint_path.replace(".safetensors", ".pt")
            new_checkpoint = {"ema_model_state_dict": ema_model_state_dict}
            torch.save(new_checkpoint, new_checkpoint_path)

        return f"New checkpoint saved at: {new_checkpoint_path}"

    except Exception as e:
        return f"An error occurred: {e}"


ROOT = os.getcwd()

checkpoint_path = os.path.join(ROOT, "model_last.pt")

reduced_checkpoint = os.path.join(ROOT, "reduced.pt")

result = extract_and_save_ema_model(checkpoint_path, reduced_checkpoint, False)

print(result)
