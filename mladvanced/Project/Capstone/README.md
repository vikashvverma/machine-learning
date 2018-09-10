# Flowers Recognition

I have published the kernel on [Kaggle](https://www.kaggle.com) here: [https://www.kaggle.com/vikashvverma/flowers-classification](https://www.kaggle.com/vikashvverma/flowers-classification)

## Setup

There are two ways to run this IPython notebook:



1. The best way to run this project is to fork this Kernel [https://www.kaggle.com/vikashvverma/flowers-classification](https://www.kaggle.com/vikashvverma/flowers-classification) on Kaggle and execute it there.
   - You may change the epochs before running the model as the higher number of epochs may take longer to learn.

2. Alternatively, it can be run locally as follows:

- Download the dataset from Kaggle: [https://www.kaggle.com/alxmamaev/flowers-recognition](https://www.kaggle.com/alxmamaev/flowers-recognition)
- Extract the dataset.
- Open the IPython Notebook and change the `base_dir` in 3rd `code` block.
- The `base_dir` is where all the flowers are present. This is directory above the sub-directory for each flower category, e.g. `/foo/bar/.../flowers`. The sub-directories(tulip, daisy etc.) are inside `flowers` directory.
- Preferably Use Linux based environment as some Linux commands such as `mkdir`, `ls` etc. have been used to create directories and reorganize the images.