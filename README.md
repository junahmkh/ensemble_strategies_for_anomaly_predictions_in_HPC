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

The trained model consisted of a line graph where each compute node was connected to the node above and below. The structure of the rack-level model is depicted in Fig.1.

### Ensemble strategies
An ensemble of models combines predictions from multiple individual models to improve overall performance. The primary goal is to leverage the strengths of different models, reducing the likelihood of errors and increasing the robustness and accuracy of predictions. Ensembles are particularly effective when individual models are diverse, meaning they make different kinds of errors. There are several techniques to create ensembles, including bagging, boosting, and stacking, each with its unique methodology for combining models.

One popular way to aggregate predictions in an ensemble is through voting strategies, which come in two main forms: soft voting and hard voting.

•	Hard voting: Each model in the ensemble casts a "vote" for a specific class label based on its prediction. The class with the majority votes across all models is chosen as the final prediction. 
•	Soft voting: Each model in the ensemble outputs its prediction as a probability, specifically for class 1 (anomaly) in our case. The final predicted probability is calculated as the average of these probabilities. This approach leverages the confidence levels of individual models, providing a more nuanced prediction compared to hard voting.

### Ensemble models
In this project, we consider an ensemble of five different models: GNN models, Random Forest (RF), Decision Tree (DT), Support Vector Machines (SVM), and Logistic Regression (LR). Since the GNN models are already trained, our first step will be to train the remaining four models. Each model will be trained on a per-node basis, using an 80:20 split for training and testing (consistent with the GNN training), with a separate model for each compute node in the HPC facility.

## References
[1]  Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, Luca Benini, A semisupervised autoencoder-based approach for anomaly detection in high performance computing systems, Engineering Applications of Artificial Intelligence, Volume 85, 2019, Pages 634-644, ISSN 0952-1976, https://doi.org/10.1016/j.engappai.2019.07.008.

[2] Martin Molan, Andrea Borghesi, Daniele Cesarini, Luca Benini, Andrea Bartolini, RUAD: Unsupervised anomaly detection in HPC systems, Future Generation Computer Systems, Volume 141, 2023, Pages 542-554, ISSN 0167-739X, https://doi.org/10.1016/j.future.2022.12.001.

[3] Martin Molan, Mohsen Seyedkazemi Ardebili, Junaid Ahmed Khan, Francesco Beneventi, Daniele Cesarini, Andrea Borghesi, Andrea Bartolini, GRAAFE: GRaph Anomaly Anticipation Framework for Exascale HPC systems, Future Generation Computer Systems, Volume 160, 2024, Pages 644-653, ISSN 0167-739X, https://doi.org/10.1016/j.future.2024.06.032.

[4] Borghesi, A., Di Santi, C., Molan, M. et al. M100 ExaData: a data collection campaign on the CINECA’s Marconi100 Tier-0 supercomputer. Sci Data 10, 288 (2023). https://doi.org/10.1038/s41597-023-02174-3
