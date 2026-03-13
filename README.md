# Federated-learning-Classifier
For Undergraduate Research

## Overview
Traditional machine learning pipelines require centralized datasets, which can create privacy risks and large data transfer overhead.
This project implements a Federated Learning framework where:
- Clients train models locally on their own data
- Only model parameters are shared with the server
- The server aggregates parameters using Federated Averaging (FedAvg)
This design allows decentralized training while maintaining data privacy.

## Key Idea
Traditional machine learning requires centralizing datasets.
Federated learning instead allows models to be trained across decentralized data sources.

## System Architecture
                +----------------------+
                |     FL Server        |
                |  Model Aggregation   |
                |      (FedAvg)        |
                +----------+-----------+
                           |
                Model Parameter Exchange
                           |
            +--------------+--------------+
            |                             |
    +-------+--------+            +-------+--------+
    |    Client 1    |            |    Client 2    |
    | Local Dataset  |            | Local Dataset  |
    | CNN Training   |            | CNN Training   |
    +-------+--------+            +-------+--------+
            |                             |
            +--------------+--------------+
                           |
                    Updated Global Model
## Federated Learning Workflow
Initialize Global Model
        |
        v
Distribute Model to Clients
        |
        v
Clients Perform Local Training
        |
        v
Upload Model Parameters
        |
        v
Server Aggregates Models (FedAvg)
        |
        v
Update Global Model
        |
        v
Next Communication Round


## Dataset
Handwriting archives of English exam papers from thirty children in rural areas provided by The BoYo Foundation.

### Training process:
1. Each client trains a model locally
2. Model parameters are uploaded to a central server
3. The server aggregates parameters using Federated Averaging
4. The updated global model is redistributed to clients
5. The process repeats for multiple communication rounds

## Technologies
### Machine Learning
- PyTorch
- CNN classifier
- Fully connected neural network
### Federated Learning
- Federated Averaging (FedAvg)
- Multi-client training simulation
- Global model synchronization
### Backend
- Flask REST API server
- HTTP communication
### Data Handling
- Pickle for model serialization
- Gzip compression for network transfer

## System Components
### Client
- Performs local training
- Uploads model weights
- Downloads updated global model
### Server
- Receives model parameters
- Performs weight aggregation
- Distributes global model
