import torch
from bpd_ui.models.model_util import load_pretrained_model, get_preprocessing_transforms


class ModelManager:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._models: dict[str, torch.nn.Module] = {}
        self._transforms: dict[str, object] = {}

    def get(self, model_name: str):
        if model_name not in self._models:
            m = load_pretrained_model(model_name, device=self.device)
            self._models[model_name] = m
        if model_name not in self._transforms:
            self._transforms[model_name] = get_preprocessing_transforms(model_name)
        return self._models[model_name], self._transforms[model_name]

    @torch.inference_mode()
    def predict(self, model_name: str, image):
        model, transform = self.get(model_name)
        x = transform(image).unsqueeze(0).to(self.device)
        logits = model(x)
        prob = torch.sigmoid(logits).flatten()[0].item()
        return prob, float(logits.flatten()[0].item())
