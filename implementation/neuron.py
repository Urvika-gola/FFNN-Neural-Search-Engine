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
    f = open(file_name, 'r')

    data = []
    f.readline()
    for inst in f.readlines():
        if not re.search('\t', inst): continue
        items = list(map(int, inst.strip().split('\t')))
        attributes, label = items[:-1], items[-1]
        data.append(Instance(attributes, label))
    return data




def train(instances, lr, epochs):
    weights = [0] * (len(instances[0].attributes))

    # YOUR CODE GOES HERE
    #   Use the weight update rule to update the weights so that the error is minimized (with instances)
    #   lr: learning rate
    #   epochs: number of epochs (passes through instances) to train for


    return weights


def predict(weights, attributes):
    # YOUR CODE GOES HERE
    #   Use weights to predict the label of instances with attributes
    prediction = 0

    return prediction


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


def train_and_test(file_tr, file_te, lr, epochs):
    # DO NOT CHANGE THIS METHOD
    instances_tr = read_data(file_tr)
    instances_te = read_data(file_te)

    weights = train(instances_tr, lr, epochs)
    return get_accuracy(weights, instances_te)


def main(file_tr, file_te, lr, epochs):
    # DO NOT CHANGE THIS METHOD
    accuracy = train_and_test(file_tr, file_te, lr, epochs)
    print(f"Accuracy on test set {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR", help="Path to train file with POS annotations")
    parser.add_argument("PATH_TE", help="Path to test file (POS tags only used for evaluation)")
    parser.add_argument("lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    print(' '.join(map(str, [args.PATH_TR, args.PATH_TE, args.lr, args.epochs])))

    main(args.PATH_TR, args.PATH_TE, args.lr, args.epochs)
