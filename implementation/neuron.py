import re
import math
import argparse


class Instance:
    # DO NOT CHANGE THIS CLASS
    def __init__(self, attributes, label):
        self.attributes = attributes
        self.label = label


def read_data(file_name):
    # DO NOT CHANGE THIS METHOD
    fl = open(file_name, 'r')
    data_read = []
    fl.readline()
    for inst in fl.readlines():
        if not re.search('\t', inst):
            continue
        items = list(map(int, inst.strip().split('\t')))
        attr, label = items[:-1], items[-1]
        data_read.append(Instance(attr, label))
    return data_read


def sigmoid(x):
    # Sigmoid function
    return 1 / (1 + math.exp(-x))


def train(instances, lr, epochs, error_type="sigmoid"):
    wt = [0] * (len(instances[0].attributes))
    for epoch in range(epochs):
        for inst in instances:
            dot_pr = sum(w * a for w, a in zip(wt, inst.attributes))
            output = sigmoid(dot_pr)
            if error_type == "classifier":
                predicted_label = 1 if output >= 0.5 else 0
                error = inst.label - predicted_label
            elif error_type == "sigmoid":
                error = inst.label - output
            for j, xj in enumerate(inst.attributes):
                wt[j] += lr * error * output * (1 - output) * xj
    return wt


def predict(weights, attributes):
    # YOUR CODE GOES HERE
    #   Use weights to predict the label of instances with attributes
    dot_pr = sum(w * a for w, a in zip(weights, attributes))
    output = sigmoid(dot_pr)
    return 1 if output >= 0.5 else 0


def make_predictions(weights, instances):
    # DO NOT CHANGE THIS METHOD
    for instance in instances:
        yield predict(weights, instance.attributes)


def get_accuracy(weights, instances):
    # DO NOT CHANGE THIS METHOD
    error = 0
    predictions = make_predictions(weights, instances)

    for instance, prediction in zip(instances, predictions):
        error += abs(instance.label - prediction)

    accuracy = float(len(instances) - error) / len(instances)
    return accuracy * 100


def train_and_test(file_tr, file_te, lr, epochs, error_type="sigmoid"):
    instances_tr = read_data(file_tr)
    instances_te = read_data(file_te)

    weights = train(instances_tr, lr, epochs, error_type)
    return get_accuracy(weights, instances_te)


def main(file_tr, file_te, lr, epochs):
    for error_type in ["classifier", "sigmoid"]:
        accuracy = train_and_test(file_tr, file_te, lr, epochs, error_type)
        print(f"Result: Accuracy on test set using {error_type} error: {round(accuracy, 2)}%")
        print("**************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR", help="Path to train file with POS annotations")
    parser.add_argument("PATH_TE", help="Path to test file (POS tags only used for evaluation)")
    parser.add_argument("lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    print(' '.join(map(str, [args.PATH_TR, args.PATH_TE, args.lr, args.epochs])))

    main(args.PATH_TR, args.PATH_TE, args.lr, args.epochs)
