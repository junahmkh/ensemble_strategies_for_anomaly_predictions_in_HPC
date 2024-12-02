# Ensemble strategies for anomaly prediction in High-performance computing (HPC) systems

Download data from the following links and store in the data directory:
- https://zenodo.org/records/7541722/files/10.tar?download=1
- https://zenodo.org/records/7541722/files/22.tar?download=1

## Introduction

Supercomputers play a central role in driving technological advancements in our society. These complex machines comprise heterogeneous components, including compute nodes and cooling infrastructure. Over the years, the complexity of supercomputing systems has increased rapidly, presenting significant challenges for facility managers, engineers, and operators—especially as we approach the exascale era. A critical challenge is the unavailability of compute nodes, which can lead to reduced productivity and financial losses for stakeholders. These issues can be mitigated through predictive maintenance, leveraging Machine Learning (ML) and Artificial Intelligence (AI) for anomaly detection and prediction.
Instances of node unavailability are classified as anomalous, as they deviate from expected operational norms. Promptly detecting and addressing these anomalies is crucial to minimizing their impact on system operations and maintaining the overall reliability of the facility.

Existing solutions for anomaly detection in HPC facilities, including rule-based systems and machine learning models, have demonstrated their potential to identify deviations from normal operational behaviour [1][2]. However, the challenge of anomaly prediction remains largely unresolved. Despite advancements in predictive analytics and machine learning, the inherent complexity and dynamic nature of HPC environments continue to pose significant challenges to achieving accurate predictions.

Graph neural networks (GNNs) have demonstrated significant potential in predicting anomalies within HPC facilities. Molan et al. [3] developed an anomaly prediction framework utilizing per-rack models to forecast anomalies across nine future windows, ranging from one hour ahead to 72 hours ahead. These GNN models are built using Graph Convolutional Networks (GCNs), which employ graph convolutional layers to perform message passing between neighbouring nodes in the graph, enabling each node to aggregate information from its local neighbourhood. The study shows that these GNN models consistently outperform other state-of-the-art approaches for predictive maintenance. The models achieve an area-under-the-curve (AUC) evaluation ranging from approximately 0.91 for predicting anomalies one hour ahead to 0.75 for predicting anomalies up to eight hours in advance. This performance significantly surpasses that of other models, including Dense Neural Networks (DNN), Gradient Boosting (GB), Random Forests (RF), Decision Trees (DT), and Markov Chains (MC). 

In this project, we aim to answer the following question: Can the performance of Graph Neural Networks (GNNs) in anomaly detection be enhanced by using an ensemble of models, compared to relying on GNNs alone? To address this question, we will evaluate the performance of an ensemble of models, including GNNs, alongside other machine learning algorithms. The goal is to determine whether combining multiple models results in improved predictive accuracy and robustness, particularly in the context of anomaly detection in HPC facilities.

## Methodology

### Graph convolutional network (GCN) models
The GCN models are trained as per-rack models, meaning each rack in the data center has its own separately trained model. These GCN models are trained for a supervised classification task. Labels are obtained by considering a future time window FW; label 1 indicates anomalies, and 0 indicates normal samples. For each node and at any point in time, a label of value 1 (anomaly) is assigned if the node encounters any anomaly in the future window FW; otherwise, the label will be 0.

The trained model consisted of a line graph where each compute node was connected to the node above and below. The structure of the rack-level model is depicted in following figure.

![image](gcn_schema.pdf)

### Ensemble strategies
An ensemble of models combines predictions from multiple individual models to improve overall performance. The primary goal is to leverage the strengths of different models, reducing the likelihood of errors and increasing the robustness and accuracy of predictions. Ensembles are particularly effective when individual models are diverse, meaning they make different kinds of errors. There are several techniques to create ensembles, including bagging, boosting, and stacking, each with its unique methodology for combining models.

One popular way to aggregate predictions in an ensemble is through voting strategies, which come in two main forms: soft voting and hard voting.

•	Hard voting: Each model in the ensemble casts a "vote" for a specific class label based on its prediction. The class with the majority votes across all models is chosen as the final prediction. 
•	Soft voting: Each model in the ensemble outputs its prediction as a probability, specifically for class 1 (anomaly) in our case. The final predicted probability is calculated as the average of these probabilities. This approach leverages the confidence levels of individual models, providing a more nuanced prediction compared to hard voting.

### Ensemble models
In this project, we consider an ensemble of five different models: GNN models, Random Forest (RF), Decision Tree (DT), Support Vector Machines (SVM), and Logistic Regression (LR). Since the GNN models are already trained, our first step will be to train the remaining four models. Each model will be trained on a per-node basis, using an 80:20 split for training and testing (consistent with the GNN training), with a separate model for each compute node in the HPC facility.

## Results

### Experimental setting
The experiments are conducted on a system with an NVIDIA Quadro RTX 6000 GPU with CUDA version 12.0 and an Intel Xeon(R) Gold 5220 CPU @2.20Ghz.

The dataset used in this study is the largest openly available dataset from a supercomputing facility, specifically the Marconi100 system at CINECA. It consists of 31 months of telemetry data collected from the facility. For this work, we utilized the annotated, 15-minute aggregated version of the dataset, focusing on data from racks 10 and 22, as compiled by Borghesi et al. [4].

The area under the Receiver-Operator characteristic (ROC) curve (AUC) is selected as the primary evaluation metric for the experimental analysis. The predictive models output the probability of the anomaly for a given future window (instead of just predicting class 0 or 1).

### Base GNN model performances (Baseline)
The following table outlines the performance of the GCN models for all nine future windows. With the ensemble of models, we aim to determine whether we can improve upon the GNN, which serves as the baseline.
| FW  | Time ahead (hr) | GNN   |
|:---:|:---------------:|:-----:|
| 4   | 1               | 0.902 |
| 6   | 1.5             | 0.88  |
| 12  | 3               | 0.878 |
| 24  | 6               | 0.75  |
| 32  | 8               | 0.796 |
| 64  | 16              | 0.757 |
| 96  | 24              | 0.699 |
| 192 | 48              | 0.63  |
| 288 | 72              | 0.544 |

The area-under-the-curve (AUC) metric is commonly used to evaluate the performance of binary classification models, where higher AUC values indicate better model performance.
From the table, we see that at a time-ahead interval of 1 hour ahead FW of 4, the GNN model achieved an AUC score of 0.902, indicating strong predictive performance. The model's performance decreases as we increase the future window. Conversely, at longer time-ahead intervals such as 72 hours (FW of 288), the AUC score decreases to 0.544, suggesting decreased predictive accuracy over longer forecast horizons.

### Base performance of other ensemble models

| FW  | RF    | DT    | SVM   | LR    |
|:---:|:-----:|:-----:|:-----:|:-----:|
| 4   | 0.608 | 0.691 | 0.272 | 0.408 |
| 6   | 0.413 | 0.491 | 0.278 | 0.413 |
| 12  | 0.512 | 0.471 | 0.276 | 0.425 |
| 24  | 0.555 | 0.488 | 0.312 | 0.444 |
| 32  | 0.583 | 0.451 | 0.345 | 0.46  |
| 64  | 0.519 | 0.543 | 0.441 | 0.485 |
| 96  | 0.515 | 0.477 | 0.464 | 0.491 |
| 192 | 0.475 | 0.507 | 0.467 | 0.493 |
| 288 | 0.463 | 0.487 | 0.484 | 0.493 |
The table shows the base performance results of the four ensemble models, excluding GNN. From the observed results, it is evident that these machine learning models perform poorly, with very low AUC scores approaching 0.5, which is the AUC of a random classifier. Among these models, the SVM performs the worst, with the average AUC score of 0.371. The best performers among the four models are RF and DT, with average AUCs over all nine future windows (FW) of 0.516 and 0.511 respectively. These results highlight the complexity of predicting anomalies in HPC environments.

### Performance using ensemble strategies
| FW  | Hard voting | Hard-GNN | Soft voting | Soft-GNN |
|:---:|:-----------:|:--------:|:-----------:|:--------:|
| 4   | 0.743       | -0.159   | 0.932       | 0.03     |
| 6   | 0.767       | -0.113   | 0.907       | 0.027    |
| 12  | 0.753       | -0.125   | 0.889       | 0.011    |
| 24  | 0.702       | -0.048   | 0.817       | 0.067    |
| 32  | 0.679       | -0.117   | 0.789       | -0.007   |
| 64  | 0.618       | -0.139   | 0.704       | -0.053   |
| 96  | 0.593       | -0.106   | 0.653       | -0.046   |
| 192 | 0.540       | -0.09    | 0.604       | -0.026   |
| 288 | 0.542       | -0.002   | 0.566       | 0.022    |

The table presents the performance of ensemble models using hard voting and soft voting strategies. The hard voting strategy has AUC scores ranging from 0.540 to 0.767. The highest performance is observed at FW = 6, where it achieves a score of 0.767, and performance tends to decrease as the FW increases, reaching its lowest at FW = 192 with a score of 0.540. The column labelled "Hard-GNN" highlights the difference in performance between the hard voting strategy and the baseline GNN model. The consistently negative values across all FWs suggest that the hard voting strategy fails to enhance the performance, indicating that it does not offer any improvement over the base GNN model. 

The soft voting strategy, on the other hand, achieves the highest AUC score of 0.932 at FW = 4, with an improvement of about 3 percent over the baseline GNN models at the same FW. Soft voting not only outperforms hard voting in all FWs, but it does also outperform baseline performance of GNN at five FWs as evident by the positive values in the column “Soft-GNN”. This demonstrates that with the superior soft voting ensemble strategy, we can improve upon the baseline GNN model's performance, leveraging the strengths of multiple models to achieve better results.

## Conclusion
In this project, we investigated the use of ensemble strategies, specifically hard voting and soft voting, to enhance the performance of baseline GNN models in predicting anomalies within HPC systems. The findings emphasize that while ensemble methods have the potential to improve predictive performance, the strategy chosen is crucial to their success. Among the two approaches, soft voting demonstrates a clear advantage, achieving improvements over the baseline GNN models’ performance, even when individual models in the ensemble exhibit poor anomaly prediction capabilities. In contrast, the hard voting strategy significantly reduces overall predictive performance, yielding an approximately 0.2 lower AUC at FW = 4 compared to the baseline GNN models. Notably, soft voting achieves the highest AUC score across all base models and delivers a three percent improvement over the baseline GNN at FW = 4, underscoring its effectiveness in leveraging ensemble methods to enhance prediction accuracy.

## References
[1]  Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, Luca Benini, A semisupervised autoencoder-based approach for anomaly detection in high performance computing systems, Engineering Applications of Artificial Intelligence, Volume 85, 2019, Pages 634-644, ISSN 0952-1976, https://doi.org/10.1016/j.engappai.2019.07.008.

[2] Martin Molan, Andrea Borghesi, Daniele Cesarini, Luca Benini, Andrea Bartolini, RUAD: Unsupervised anomaly detection in HPC systems, Future Generation Computer Systems, Volume 141, 2023, Pages 542-554, ISSN 0167-739X, https://doi.org/10.1016/j.future.2022.12.001.

[3] Martin Molan, Mohsen Seyedkazemi Ardebili, Junaid Ahmed Khan, Francesco Beneventi, Daniele Cesarini, Andrea Borghesi, Andrea Bartolini, GRAAFE: GRaph Anomaly Anticipation Framework for Exascale HPC systems, Future Generation Computer Systems, Volume 160, 2024, Pages 644-653, ISSN 0167-739X, https://doi.org/10.1016/j.future.2024.06.032.

[4] Borghesi, A., Di Santi, C., Molan, M. et al. M100 ExaData: a data collection campaign on the CINECA’s Marconi100 Tier-0 supercomputer. Sci Data 10, 288 (2023). https://doi.org/10.1038/s41597-023-02174-3
