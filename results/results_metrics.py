import os
import pandas as pd
import time

import logging  # Logging added

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("Starting program and setting up environment.")


class ResultMetrics:
    """
    The results are saved in a csv file with the following header:
    Model,Attribution Method,Layer,Metric,Upscale Method,Value
    """

    def __init__(self, path: str, save_each_time: float = 30.0):
        self.path = path
        self.HEADER = [
            "Image Index",
            "Label",
            "Predicted Label",
            "Model",
            "Dataset",
            "Attribution Method",
            "Layer",
            "Metric",
            "Upscale Method",
            "Mixing Method",
            "Value",
        ]

        self.results = pd.DataFrame(columns=self.HEADER)
        self.save_each_time = save_each_time
        self.last_save_time = 0
        self.load_results()

    def load_results(self):
        if os.path.exists(self.path):
            self.results = pd.read_csv(self.path)
            print(f"Results loaded from {self.path}.")
        else:
            print(f"Results file not found. Creating new results file {self.path}.")
            self.results = pd.DataFrame(columns=self.HEADER)

    def add_result(
        self,
        model,
        attribution_method,
        dataset,
        layer,
        metric,
        upscale_method,
        mixing_method,
        value,
        label=-1,
        predicted_label=-1,
        image_index=-1,
    ):
        self.results.loc[len(self.results)] = {
            "Image Index": image_index,
            "Label": label,
            "Predicted Label": predicted_label,
            "Model": model,
            "Dataset": dataset,
            "Attribution Method": attribution_method,
            "Layer": layer,
            "Metric": metric,
            "Upscale Method": upscale_method,
            "Mixing Method": mixing_method,
            "Value": value,
        }

        self.save_results(force_save=False)

    def save_results(self, force_save=True):
        if force_save:
            self.results.to_csv(self.path, index=False)
            logger.info(f"Results saved to {self.path}.")
            return
        now = time.time()
        if now - self.last_save_time > self.save_each_time:
            self.last_save_time = now
            self.results.to_csv(self.path, index=False)
            logger.info(f"Results saved to {self.path}.")

    def get_last_image_index(self):
        if self.results.empty:
            return -1
        else:
            return self.results["Image Index"].max()
