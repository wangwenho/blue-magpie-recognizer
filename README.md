# 🦜 Blue Magpie Recognizer
**Final Project For Course: `Introduction to Computer Vision and Its Applications`.**

## 🔖 Abstract
- **台灣藍鵲 Taiwan Blue Magpie** (*Urocissa caerulea*) is the national bird of Taiwan, known for its striking blue plumage and long tail.
- The suspected smuggling of the **紅嘴藍鵲 Red-billed Blue Magpie** (*Urocissa erythroryncha*) into Taiwan has led to a gradual expansion of its population, threatening the habitat of the Taiwan Blue Magpie.
- This project aims to raise awareness of this issue and help people recognize and distinguish between the two species.

> ![Difference](assets/difference.png)

### Blue Magpie Species
There are five kinds of blue magpies in the world:
| Chinese Name   | English Name               |
|----------------|----------------------------|
| 台灣藍鵲         | Taiwan Blue Magpie         |
| 紅嘴藍鵲         | Red-billed Blue Magpie     |
| 黃嘴藍鵲         | Yellow-billed Blue Magpie  |
| 白翅藍鵲         | White-winged Magpie        |
| 斯里蘭卡藍鵲      | Sri Lanka Blue Magpie      |

In this project, we selected three species with similar appearances: **Taiwan Blue Magpie**, **Red-billed Blue Magpie**, and **Yellow-billed Blue Magpie** as the dataset, aiming to train a model that can distinguish between them.

## 📋 TODO List
- [x] Dataset Preparation (using Selenium)
- [x] Model Building (using PyTorch)
- [x] Training and Evaluation (using PyTorch)
- [x] Hyperparameter Tuning (using TensorBoard)
- [x] Model Visualization (using Captum)
- [x] UI Wrapping (using Gradio)

## 👥 Contributors
- **[王文和](https://github.com/wangwenho)** - Dataset Preparation, Frontend Interface Development
- **[李崑銘](https://github.com/PuiPui32071)** - Model Selection, Fine-tuning, and Optimization

## 🚀 Getting Started
### Prerequisites
- Python >= 3.10
- PyTorch >= 2.5
- CUDA

### Environment Setup
1. Clone this repository:
    ```bash
    git clone https://github.com/wangwenho/blue-magpie-recognizer.git
    ```
2. Create a conda environment:
    ```bash
    conda create -n bmr python==3.10.16
    conda activate bmr
    ```
3. Install required packages:
   - macOS
    ```bash
    conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 scikit-learn -c pytorch
    conda install captum==0.7 ipykernel ipywidgets matplotlib selenium tensorboard tqdm
    pip install gradio
    ```
    - Linux and Windows
    ```bash
    conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install captum==0.7 ipykernel ipywidgets matplotlib selenium tensorboard tqdm
    pip install gradio
    ```

## 📁 Dataset Preparation
### Option 1: Use Our Dataset
- If you want to use our proposed dataset, please [click here](https://drive.google.com/drive/folders/1E_pRJGIzvn5IInmIfg55CrSge5gsOnGE?usp=drive_link) to download `dataset_1500.zip`.
- Extract `dataset_1500.zip` to the root directory.
- It contains three different types of blue magpies images, with each class having `1500` images sized `256x256`.
- We also provide `raw_images.zip` for additional sampling options.

```
├── dataset_1500
│   ├── red-billed-blue-magpie
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── taiwan-blue-magpie
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── yellow-billed-blue-magpie
│       ├── 1.png
│       ├── 2.png
│       └── ...
```

### Option 2: Use Custom Dataset
- To run Selenium properly, you need a driver that matches your browser, e.g., [Chrome](https://developer.chrome.com/docs/chromedriver/downloads?hl=zh-tw).
- Once downloaded, place the driver in the root directory.
- Then, modify `prepare_dataset.ipynb` and run the script to get the bird dataset you want.
- Note that you must pass the [media.ebird.org](https://media.ebird.org/catalog?taxonCode=formag1&mediaType=photo) URL with the specific species to the `EbirdCrawler` class so that the crawler can work properly.

```
├── dataset_<num_of_images>
│   ├── <species-1>
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── <species-2>
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── <species-3>
│       ├── 1.png
│       ├── 2.png
│       └── ...
```

## 🏋️ Training
### Model We Used
- To lower the cost of training and achieve maximum accuracy, we use the Pre-Trained **EfficientNetV2_s** from the paper: [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298), and fine-tune it with our dataset.

### Start Training
- The `train_model.ipynb` includes all the steps to construct and train our blue magpie recognizer. Just run it and have fun.
- Additionally, you can build your own model and save it in the `models` folder. Then, you can import it in the block and train it.
- During the program's execution, it automatically saves the TensorBoard logs in the `runs` folder, which you can use for better inspection.

### Model Checkpoints
- If you just want to do testing, we provide two checkpoints in the `ckpts/effv2s_bn_si_0.001_10_0.5` folder.
- In there, one has the minimum validation loss, another has the highest validation accuracy. You can directly load the pre-trained model and use it.

## 🧪 Testing
- When training is done, you can run `eval_model.ipynb` to check the performance of your model.

## 📈 Evaluation
- Our trained model achieved 95% accuracy on the test dataset, which is beyond our expectation.
- **Minimum loss:**
    | Class                        | Precision | Recall | F1-Score | Support |
    |------------------------------|-----------|--------|----------|---------|
    | red-billed-blue-magpie       | 0.9607    | 0.9205 | 0.9402   | 239     |
    | taiwan-blue-magpie           | 0.9494    | 0.9740 | 0.9615   | 231     |
    | yellow-billed-blue-magpie    | 0.9378    | 0.9561 | 0.9469   | 205     |
    | **Accuracy**                 |           |        | 0.9496   | 675     |
    | **Macro Avg**                | 0.9493    | 0.9502 | 0.9495   | 675     |
    | **Weighted Avg**             | 0.9499    | 0.9496 | 0.9495   | 675     |
- **Highest accuracy:**
    | Class                        | Precision | Recall | F1-Score | Support |
    |------------------------------|-----------|--------|----------|---------|
    | red-billed-blue-magpie       | 0.9696    | 0.9331 | 0.9510   | 239     |
    | taiwan-blue-magpie           | 0.9567    | 0.9567 | 0.9567   | 231     |
    | yellow-billed-blue-magpie    | 0.9346    | 0.9756 | 0.9547   | 205     |
    | **Accuracy**                 |           |        | 0.9541   | 675     |
    | **Macro Avg**                | 0.9536    | 0.9551 | 0.9541   | 675     |
    | **Weighted Avg**             | 0.9545    | 0.9541 | 0.9540   | 675     |


## 👀 Model Interpretability
- As mentioned in the Abstract, our goal is to differentiate three types of blue magpies. The question is, does the model distinguish them as we do?
- You can run the `captum.ipynb` to see how the model interprets images. We use the method called **Occlusion Attribution** to show the important areas in the image.
- The darker green areas indicate that the model relies more on those areas to distinguish the bird.
> ![captum](assets/captum.png)

## 🖥️ Gradio App
- It is inconvenient to change the image path in the code and run it every time we want to make a prediction. Therefore, we built a Gradio app that allows you to drag an image and immediately see the prediction and Captum heatmap.
- By running `gradio_app.ipynb`, you can access the testing interface in your browser.
> ![gradio](assets/gradio.png)

## 🌟 Acknowledgement
- Thanks to [eBird.org](https://ebird.org/home) for providing such a platform with many valuable statistics for educational purposes :)
- Special thanks to the eBird team for their continuous efforts in maintaining and updating the database.
- We also appreciate the contributions of all the bird watchers and photographers who have shared their observations and images on eBird.