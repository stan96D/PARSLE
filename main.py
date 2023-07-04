import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import stanza
import csv

import os
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.1'

from asyncio.windows_events import NULL
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import plotly.express as px
from rpy2.robjects import pandas2ri
import pandas as pd

class Feedback:

    def __init__(self, text, name, role, nameFor, assessment, scale):
        self.text = text
        self.name = name
        self.nameFor = nameFor
        self.role = role
        self.assessment = assessment
        self.scale = scale
        self.feedback = None
        self.feedup = None
        self.feedforward = None

    def setFeedup(self, value):
        self.feedup = value

    def setFeedback(self, value):
        self.feedback = value

    def setFeedforward(self, value):
        self.feedforward = value


def createModel(trainingSet, amount):
    # Read the Excel file, skipping the first row with column names
    df = pd.read_excel(trainingSet + '.xlsx', skiprows=1)

    nlp = stanza.Pipeline(processors='tokenize,pos,lemma', lang='en')

    # Define sentences and labels
    sentences = df.iloc[0:int(amount), 0].tolist()
    labels = df.iloc[0:int(amount), 1].tolist()

    # Preprocess the sentences using Stanza
    preprocessed_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        preprocessed_sentence = ' '.join([word.lemma for sent in doc.sentences for word in sent.words])
        preprocessed_sentences.append(preprocessed_sentence)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split the data into training and validation sets
    sentences_train, sentences_val, labels_train, labels_val = train_test_split(
        preprocessed_sentences, labels, test_size=0.2, random_state=42
    )

    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize and encode the sentences
    inputs_train = tokenizer(sentences_train, padding=True, truncation=True, return_tensors="pt")
    inputs_val = tokenizer(sentences_val, padding=True, truncation=True, return_tensors="pt")

    # Encode the labels
    label_mapping = {label: i for i, label in enumerate(set(labels))}
    labels_train_encoded = [label_mapping[label] for label in labels_train]
    labels_val_encoded = [label_mapping[label] for label in labels_val]

    # Create dataset and dataloaders
    train_dataset = torch.utils.data.TensorDataset(inputs_train["input_ids"], inputs_train["attention_mask"], torch.tensor(labels_train_encoded))
    val_dataset = torch.utils.data.TensorDataset(inputs_val["input_ids"], inputs_val["attention_mask"], torch.tensor(labels_val_encoded))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

    # Load pre-trained BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))
    model.to(device)

    print("Model created")

    return train_dataloader, val_dataloader, model, device


def trainModel(model, device, train_dataloader, val_dataloader):

    # Set optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 1
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0
        num_correct = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                _, predicted_labels = logits.max(1)
                num_correct += (predicted_labels == labels).sum().item()
                num_samples += labels.size(0)
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = num_correct / num_samples * 100
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {val_accuracy:.2f}%")

    model.save_pretrained("saved_model/model")
    print("Model saved in " + "saved_model/model")
    return 

def extractExcelData():

    # Extract the Excel data
    df = pd.read_excel('data_set.xlsx', skiprows=1)

    index = 0
    feedback_data = []
    input_sentences = []

    while index < df.shape[0]:

        feedback = Feedback(df.iloc[index].values[0],
                            df.iloc[index].values[1],
                            df.iloc[index].values[2],
                            df.iloc[index].values[3],
                            df.iloc[index].values[4],
                            df.iloc[index].values[5])
        
        feedback_data.append(feedback)
        input_sentences.append(df.iloc[index].values[0])
        index += 1 

    return feedback_data, input_sentences

def evaluateModel(input_sentences, loaded_model):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # For multiple sentences
    input_ids = tokenizer.batch_encode_plus(input_sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move input tensors to the device
    input_ids = input_ids.to(device)

    # Set the model to evaluation mode
    loaded_model.eval()

    # Make predictions
    with torch.no_grad():
        logits = loaded_model(input_ids)[0]  # Access the logits from the model output
        probabilities = torch.softmax(logits, dim=1)
        print(probabilities)
        predicted_labels = torch.argmax(probabilities, dim=1)
        print(predicted_labels)

    # Convert predicted labels to class names
    label_mapping = {0: "feedback", 1: "feedforward", 2: "feedup"}
    predicted_labels = [label_mapping[label.item()] for label in predicted_labels]
    print(label_mapping, predicted_labels)
    return predicted_labels

def exportCSV(predicted_labels, feedback_data):

    index = 0
    dataCSV = []
    while index < len(feedback_data):

        fb = feedback_data[index]
        fb.setFeedback(int(predicted_labels[index] == 'feedback'))
        fb.setFeedup(int(predicted_labels[index] == 'feedup'))
        fb.setFeedforward(int(predicted_labels[index] == 'feedforward'))

        index += 1

    dataCSV = [
        ['user', 'role', 'scale', 'assessment', 'for', 'text', 'feedback', 'feedup', 'feedforward'],
    ]

    for fb in feedback_data:

        dataCSV.append([fb.name, fb.role, fb.scale, fb.assessment, fb.nameFor, fb.text, fb.feedback, fb.feedup, fb.feedforward])

    # Specify the file path and name for the CSV file
    csv_file = 'data_ENA.csv'

    # Open the file in write mode and create a CSV writer object
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the data to the CSV file
        writer.writerows(dataCSV)

    print("CSV file created successfully in data_ENA.csv.")

    return

def createDataFrame():
    df = pd.read_csv("./data_ENA.csv")

    # df
    with (robjects.default_converter + pandas2ri.converter).context():
        r_from_pd_df = robjects.conversion.get_conversion().py2rpy(df)

    return r_from_pd_df

def createENA(r_from_pd_df):

    base = importr('base')
    rena = importr('rENA')

    units = robjects.StrVector(['Role', 'UserName'])
    convs = robjects.StrVector(['Role', 'Assessment'])
    codes = robjects.StrVector(['Feedback', 'Feedup', 'Feedforward'])
    metadata = robjects.StrVector(['Scale', 'Text', 'For'])

    unit_cols_i = robjects.IntVector((2,1))
    units_sub_df = r_from_pd_df.rx(True, unit_cols_i)
    convs_cols_i = robjects.IntVector((2,4))
    convs_sub_df = r_from_pd_df.rx(True, convs_cols_i)
    codes_cols_i = robjects.IntVector((7,8,9))
    codes_sub_df = r_from_pd_df.rx(True, codes_cols_i)
    metadata_cols_i = robjects.IntVector((3,6,5))
    metadata_sub_df = r_from_pd_df.rx(True, metadata_cols_i)

    ena_accum = robjects.r('ena.accumulate.data')
    accum = ena_accum(units_sub_df, convs_sub_df, codes_sub_df, metadata = metadata_sub_df)

    ena_set = robjects.r('ena.make.set')
    model = ena_set(accum)

    ena_plot = robjects.r('ena.plot')
    ena_plot_points = robjects.r('ena.plot.points')

    model_points = model.rx('points')[0]
    model_points_mtx = robjects.r('as.matrix')(model_points)
    model_plot = ena_plot(model)
    model_plot = ena_plot_points(model_plot, model_points_mtx)

    return model_plot


def visualizeENA(model_plot):
    html_print = robjects.r('htmltools::html_print')
    as_tags = robjects.r('htmltools::as.tags')
    model_html = html_print(as_tags(model_plot['plot'], standalone = True), viewer = robjects.r("NULL"))[0]
    with open('output.html', 'w') as file:
        file.write(model_html)
    import webbrowser
    webbrowser.open_new_tab(model_html)

def main():
    trainModelAnswer = input("Do you want to create a new model? It will override the current one. (Y/N).")
    if (trainModelAnswer == "Y" or trainModelAnswer == "y"):

        createModelSetName = input("What is the name of the dataset?")
        createModelRowCount = input("What is the amount of data in rows?")

        print("Preprocessing data....")
        train_dataloader, val_dataloader, model, device = createModel(createModelSetName, createModelRowCount)

        print("Training started....")
        trainModel(model, device, train_dataloader, val_dataloader)


    loadModelAnswer = input("Do you want to test the model? It will generate CSV data for the ENA. (Y/N).")
    if (loadModelAnswer == "Y" or loadModelAnswer == "y"):

        # Load the Model
        loaded_model = BertForSequenceClassification.from_pretrained("saved_model/model")

        feedback_data, input_sentences = extractExcelData()
        predicted_labels = evaluateModel(input_sentences, loaded_model)
        exportCSV(predicted_labels, feedback_data)

    showENAAnswer = input("Do you want to plot the CSV in an ENA? It will open a html web browser. (Y/N).")
    if (showENAAnswer == "Y" or showENAAnswer == "y"):
        df = createDataFrame()
        model = createENA(df)
        visualizeENA(model)

main()












