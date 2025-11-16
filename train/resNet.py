# ===========================================================================
# UFPR-ALPR Dataset Trainer for Classification Model

# This script trains ResNet18 with the folowing variations of the dataset:
# 1. Unsesored (Raw dataset)
# 2. Blur censored (Only faces censored)
# 3. Black box censored (Only faces censored)

# All variations tested on the unsensored dataset
#
# The codes used to complete this work were adapted 
# from those seen during the course 'Tópicos em aprendizado de máquina'
# ===========================================================================
import os
import math
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from Config import Config
from datetime import datetime
from torchvision import models

# ========================
# CONFIGURAÇÕES e LOGGING
# ========================
def config(project):
    date = datetime.now().strftime("%Y%m%d_%H%M")
    rede = "ResNet"
    config = Config(project, date, rede)

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=f"logs/{config.name}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("Conectado em um ambiente com", config.device)
    logging.info(f"Iniciando experimento no ambiente: {config.device}")
    logging.info(f"Classes: {config.class_names}")

    return config, logging

# ========================
# MODELO
# ========================
def confModel(config, random):
    logging.info(f'seed = {random}')
    torch.manual_seed(random)
    model = models.resnet18()
    model.fc = nn.Linear(512, 2)
    
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.0001) 
    
    logging.info("Modelo = ResNet18")
    logging.info(f'Épocas = {config.epochs}')
    logging.info(f'Batch = {config.batch_size}')
    logging.info(f'Paciência = {config.patience}')
    logging.info(f'Dados de treino = {config.train_data_path}')
    logging.info(f'Dados de validação = {config.val_data_path}')
    logging.info(f'Dados de teste = {config.test_data_path}')
    logging.info(f"lr opt = {optimizer_ft.param_groups[0]['lr']}")
    logging.info("Modelo configurado")
    
    return model, criterion, optimizer_ft

# ========================
# TREINO
# ========================
def train_model(model, criterion, config, logging, optimizer, max_epochs, grace_period):
    best_loss = math.inf
    curr_grace_period = 0
    best_model = copy.deepcopy(model.state_dict())

    logging.info(f"Iniciando treinamento {config.name}")

    for epoch in range(max_epochs):
        print(f"Época {epoch+1}/{max_epochs}")
        print('-' * 10)

        for phase in ["treino", "validacao"]:
            if phase == "treino":
                model.train()
                data = config.train_loader
                data_size = config.train_size
            else:
                model.eval()
                data = config.val_loader
                data_size = config.val_size

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in data:
                inputs,labels = inputs.to(config.device), labels.to(config.device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "treino"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "treino":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects.double() / data_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print()

            if phase == "validacao":
              if epoch_loss < best_loss:
                best_loss = epoch_loss
                curr_grace_period = 0
                best_model = copy.deepcopy(model.state_dict())
              else:
                curr_grace_period += 1
                if curr_grace_period >= grace_period:
                  print("Early stopping")
                  logging.info(f"Early stopping at {epoch}")
                  model.load_state_dict(best_model)
                  return

    model.load_state_dict(best_model)
    return

# ========================
# TESTE
# ========================
def test_model(model, config, logging):
    logging.info("Avaliando no conjunto de teste!")
    model.eval()
    test_corrects = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for inputs, labels in config.test_loader:
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)
        true_positives += torch.sum((preds == 1) & (labels.data == 1)).item()
        false_positives += torch.sum((preds == 1) & (labels.data == 0)).item()
        false_negatives += torch.sum((preds == 0) & (labels.data == 1)).item()
        true_negatives += torch.sum(((preds == 0) & (labels.data == 0))).item()

    test_acc = test_corrects.double() / config.test_size
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    logging.info("-----------------MÉTRICAS--------------------------")
    logging.info(f'Test Acc: {test_acc:.4f}')
    logging.info(f'Precision = {precision:.4f}')
    logging.info(f'Recall = {recall:.4f}')
    logging.info(f'F1 = {f1:.4f}')    
    logging.info("---------------------------------------------------")
    logging.info(f'corretos = {test_corrects:.2f}, tamanho dos dados = {config.test_size:.2f}')
    logging.info(f'TN = {true_negatives:.2f}')
    logging.info(f'FN = {false_negatives:.2f}')
    logging.info(f'TP = {true_positives:.2f}')
    logging.info(f'FP = {false_positives:.2f}')
    logging.info("==================================================")

# ========================
# MAIN
# ========================

if __name__ == '__main__':
    project = np.array(["Uncensored", "BBox", "Blur"])
    ran = random.randint(0, 100)

    print("=======================================")
    print("Experimento - Iniciando")
    print("=======================================")

    print("=======================================")
    print("Experimento - Uncensored")
    print("=======================================")
    cfg0, log0 = config(project[0])
    model0, criterion0, optimizer_ft0 = confModel(cfg0, ran)
    train_model(model0, criterion0, cfg0, log0, optimizer_ft0, max_epochs=cfg0.epochs, grace_period=cfg0.patience)
    test_model(model0, cfg0, log0)

    print("=======================================")
    print("Experimento - BBox")
    print("=======================================")
    cfg1, log1 = config(project[1])
    model1, criterion1, optimizer_ft1 = confModel(cfg1, ran)
    train_model(model1, criterion1, cfg1, log1, optimizer_ft1, max_epochs=cfg1.epochs, grace_period=cfg1.patience)
    test_model(model1, cfg1, log1)

    print("=======================================")
    print("Experimento - Blur")
    print("=======================================")
    cfg2, log2 = config(project[2])
    model2, criterion2, optimizer_ft2 = confModel(cfg2, ran)
    train_model(model2, criterion2, cfg2, log2, optimizer_ft2, max_epochs=cfg2.epochs, grace_period=cfg2.patience)
    test_model(model2, cfg2, log2)

    print("=======================================")
    print("Experimento - Finalizado")
    print("=======================================")
