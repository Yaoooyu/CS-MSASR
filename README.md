🎯 CS-MSASR: 
 ## Video Multimodal Dataset： Sentiment Analysis + Speech Recognition
 ## Base on Changsha --- Representative dialects of southern China
<p align="center">
  <img src="https://github.com/user-attachments/assets/3ffb7f87-aa79-453e-926c-1cfaab9d5e9f" alt="CS-MSASR Dataset Preview" width="400"/>
</p>

---

### 📌 Background

Intelligent human-computer interaction systems should extend beyond mainstream languages to embrace **regional dialects**, which carry rich emotional nuances and cultural heritage. As the capital of Hunan Province, **Changsha** is home to the **Changsha dialect**—a prominent variety in southern China known for its **dynamic intonation** and **expressive vocabulary**. However, despite its linguistic value, the Changsha dialect remains **largely underrepresented** in existing Artificial Intelligence (AI) corpora.

This project aims to fill that gap by providing resources and tools tailored for the Changsha dialect, contributing to more inclusive and culturally aware AI systems.

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
     https://pan.baidu.com/s/1lYznkyVZ0GsaDKosHb9fKQ
  提取码: 2cbi 

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

