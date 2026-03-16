# Federated-learning-Classifier
For Undergraduate Research (Python >= 3.8)

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
### Distributed System
- Federated Averaging (FedAvg)
- Multi-client training simulation
- Global model synchronization
- Parameter server architecture 
### Backend
- Flask REST API server
- HTTP communication
### Data Handling
- Pickle (model serialization)
- Gzip compression

## System Components
### Client
- Performs local training
- Uploads model weights
- Downloads updated global model
### Server
- Receives model parameters
- Performs weight aggregation
- Distributes global model

## Project Structure
    Federated-learning-Classifier
    │
    ├──requirements.txt
    │
    ├── server
    │   └── app.py
    │
    ├── client
    │   ├── traint.py
    │   └── train2.py
    │
    ├── models
    │   ├── CNN
    │   └── SimpleNN
    │
    ├── CV
    │   └── extract.py
    │
    ├── result
    │
    └── README.md

## Installation
Clone the repository
```
git clone https://github.com/yourname/Federated-learning-Classifier.git
cd Federated-learning-Classifier
```
Install dependencies
```
pip install -r requirements.txt
```
Dependencies typically include:
```
torch
flask
numpy
pickle
gzip
```

## Usage
### 1. Start the Federated Learning Server
```
python server/app.py
```
The server will start a Flask API responsible for:
- receiving model parameters
- aggregating models
- distributing the global model
### 2. Run a Client
```
python client/train.py
python client/train2.py
```
Each client will:
1. download the global model
2. train locally on its dataset
3. upload updated model parameters
4. receive the updated global model

### 3. Training Loop
The federated training process repeats for multiple communication rounds.
Example:
```
Round 1
Client Training → Upload → Server Aggregation

Round 2
Client Training → Upload → Server Aggregation
```

## Training Configuratio
## Training Configuration

The following configuration was used for federated training.

| Parameter | Value |
|-----------|------|
| Number of Clients | 2 |
| Local Epochs | 10 |
| Batch Size | 64 |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Model Architecture | CNN / SimpleNN |

Each client trains the model locally on its private dataset before sending updated parameters to the server for aggregation.

## Federated Averaging
The global model is computed by averaging client parameters.
$w_{global} = \frac{1}{N} \sum_{i=1}^{N} w_i$
Where
- w_{i} is the model from client i
- 𝑁 is the number of participating clients
This allows the server to update the model without accessing raw training data.

## Example Use Cases
Federated learning is useful in domains where data privacy is critical:
- Healthcare AI
- Mobile device learning
- Financial data modeling
- Edge AI systems

## Future Improvements
Possible extensions of this project include:
- Differential Privacy integration
- Secure aggregation
- Asynchronous federated learning
- Cross-device federated training
- Scaling to larger client networks
