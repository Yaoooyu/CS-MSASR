🎯 CS-MSASR: 
## Multimodal Sentiment Analysis and Speech Recognition Dataset for Video Based on Changsha Dialects

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ffb7f87-aa79-453e-926c-1cfaab9d5e9f" alt="CS-MSASR Dataset Preview" width="400"/>
</p>

---

### 📌 Background

Intelligent human-computer interaction systems should not only focus on mainstream languages but also understand **regional dialects** that are rich in emotional and cultural characteristics. Changsha is the capital of Hunan province in China, and the **Changsha dialect**, as one of the significant dialects in southern China, features **variable intonation** and **vivid expressions**, but is **almost absent** in current Artificial Intelligence (AI) corpora.

---

### 📂 Dataset Overview

We introduce **CS-MSASR**, the **first multimodal video dataset** for the Changsha dialect, aimed at **sentiment analysis** and **speech recognition** research.

- 🎥 **1085 video clips** covering diverse real-life scenarios
- 🗣️ Speakers ranging from **8 to 93 years old**, ensuring diversity
- 🧾 Each video is **manually transcribed** with authentic Changsha dialect text
- ❤️ **5 categories of multimodal sentiment labels**:
  - `Negative`
  - `Weakly Negative`
  - `Neutral`
  - `Weakly Positive`
  - `Positive`
- 🧠 **Unimodal sentiment annotations** for:
  - Text
  - Audio
  - Visual
- ✂️ **Fine-grained temporal segmentation**

---
### 🔗 Dataset

- **Google Drive**  
  https://drive.google.com/drive/folders/1g5zbyc6ZMVdqC95yfTl4lZZSIkK9V_E5?usp=drive_link

- **百度网盘**  
  https://pan.baidu.com/s/1b3NqWo1ZfqJXgjk5GavE7Q  
  提取码: `w69i`

- The dataset after feature extraction is in the file CS-MSASR_fulldata.pkl.
- The file contains all the data and does not differentiate between the training set, testing set and validation set. This is convenient for users to divide the dataset by themselves. If you want to synchronize with the article, please use 8:1:1 division, random_state=42, or contact the author.

---

### 📊 Benchmark

We evaluated:

- **12 mainstream multimodal sentiment analysis models**
- **5 speech recognition models** using:
  - Direct inference
  - Fine-tuning on CS-MSASR

---

### 📎 Citation

For detailed citation information, please refer to our [citations.json](https://github.com/Yaoooyu/CS-MSASR/blob/main/citations.json) file.

