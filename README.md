
# DDA5001-25fall Final Project

The final project code for the DDA5001-25fall. Please refer to the project manual for more detailed information.

## How to start

### Part 1

All files related to part 1 of the final project are located in `p1`, and thus, before you run any code, please make sure you are in the `p1` directory.

For part 1, if you do not have enough computing resources, you'd better follow the instruction for using free GPU resources from Kaggle, please refer to [this](./Kaggle_training.md).

The main entry point of part 1 is `p1/main.ipynb`, which is a Jupyter notebook. You can run it in your local machine or in Kaggle.

### Part 2

All files for Part 2 are located in the `p2` directory. Before running the code, please ensure you are in the correct working directory. For instance, as shown in `p2/main.ipynb`, you may need to change your working directory to `p2/src`.

As with Part 1, if you have limited computing resources, we recommend following the instructions for using free GPU resources on Kaggle, which can be found [here](./Kaggle_training.md).

For users with a compatible local machine, you can refer to `vllm_rollout.py` for accelerated inference using vLLM. (Please note that running vLLM on Kaggle can be challenging due to library and hardware compatibility issues.)

**Note:** `p2/example.ipynb` provides a minimal example of running LoRA on Kaggle. For simplicity, many features, such as detailed plots, have been omitted. Your final results should be more comprehensive.

While the core training logic should be preserved, you are encouraged to experiment with other components if you wish to explore further!

## Acknowledgements

The project is based on [nanoGPT](https://github.com/karpathy/nanoGPT). Thanks for the great work!