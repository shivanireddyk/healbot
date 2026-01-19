import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names*")


# Load training and testing data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
clf.feature_names = None

# Extract feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


# Initialize dictionaries for severity, description, and precaution
severityDictionary = {}
description_list = {}
precautionDictionary = {}

# Define reduced_data
reduced_data = training.groupby(training['prognosis']).max()

# Defining calc_condition function
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum += severityDictionary[item]
    if ((sum * days) / (len(exp) + 1) > 13):
        print("You should take the consultation from a doctor.")
    else:
        print("It might not be that bad, but you should take precautions.")


# Initialize symptoms dictionary
symptoms_dict = {symptom: index for index, symptom in enumerate(cols)}

# Load symptom descriptions
def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

# Load symptom severity
def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Ensure row has at least 2 elements
                severityDictionary[row[0]] = int(row[1])


# Load symptom precautions
def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# Initialize chatbot
def getInfo():
    print("---------------------------------------HealBot---------------------------------------")
    name = input("\nYour Name? \t\t\t\t-> ")
    while True:
        age_input = input("\nYour Age? \t\t\t\t-> ")
        if age_input.isdigit():
            age = int(age_input)
            if age in range(1, 101):
                break  
            else:
                print("Invalid age. Please enter a number between 1 and 100.")
        else:
            print("Invalid input. Please enter a valid integer for age.")

    while True:
        gender = input("Please enter your gender (male/female/other): ").lower()
        if gender in ['male', 'female', 'other']:
            break  
        else:
            print("Invalid input. Please enter 'male', 'female', or 'other'.")

    print("Hello,", name)
    print("Your age is:", age)
    print("Your gender is:", gender)


# Pattern matching for symptom input
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return pred_list

# Secondary prediction based on symptoms
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

# Print disease prediction
def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

# Decision tree traversal to predict disease
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    chk_dis = ",".join(feature_names).split(",")

    symptoms_present = []
    while True:
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")
        pred_list = check_pattern(chk_dis, disease_input)
        if pred_list:
            print("searches related to input: ")
            for num, it in enumerate(pred_list):
                print(num, ")", it)
            conf_inp = int(input(f"Select the one you meant (0 - {len(pred_list) - 1}):  "))
            disease_input = pred_list[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. From how many days? : "))
            break
        except:
            print("Enter valid input.")

    # Recursive function for tree traversal
    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0

            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = input(f"{syms}? (yes/no): ")
                while inp.lower() not in ['yes', 'no']:
                    print("Provide proper answers (yes/no): ", end="")
                    inp = input()
                if inp.lower() == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)

            if present_disease[0] == second_prediction[0]:
                print("You may have ", present_disease[0])
                print(description_list.get(present_disease[0], "Description not available"))
            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list.get(present_disease[0], "Description not available"))
                print(description_list.get(second_prediction[0], "Description not available"))

            precution_list = precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(precution_list):
                print(i + 1, ")", j)

            print("\n Having flucating health is not uncommon. It can be stressful, but I want you to remember that things will improve and I am always here for you. Get Well Soon!")


    recurse(0, 1)

# Loading symptom descriptions, severity, and precautions
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols)
print("----------------------------------------------------------------------------------------")