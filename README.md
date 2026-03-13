# Federated-learning-Classifier
For Undergraduate Research

## Project Overview

This project implements a Federated Learning–based classification system where multiple clients collaboratively train a machine learning model without sharing their raw data.

The system simulates a typical federated learning pipeline with a central server coordinating model aggregation and multiple clients performing local training.

## Key Idea

Traditional machine learning requires centralizing datasets.
Federated learning instead allows models to be trained across decentralized data sources.

### Training process:

Each client trains a model locally #FFFF00

Model parameters are uploaded to a central server

The server aggregates parameters using Federated Averaging

The updated global model is redistributed to clients

The process repeats for multiple communication rounds

Technologies

Machine Learning

PyTorch

CNN classifier

Fully connected neural network

Federated Learning

Federated Averaging (FedAvg)

Multi-client training simulation

Global model synchronization

Backend

Flask REST API server

HTTP communication

Data Handling

Pickle for model serialization

Gzip compression for network transfer

## System Components

Client

Performs local training

Uploads model weights

Downloads updated global model

Server

Receives model parameters

Performs weight aggregation

Distributes global model
