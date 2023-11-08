from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer
from get_data import get_dataset, get_dataset_split
import numpy as np
import tensorflow as tf
#print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import numpy as np


class FeedForwardNeuralNetwork:
    def __init__(self, input_shape):
        # Initialize the Feedforward Neural Network (FeedForwardNeuralNetwork) class with a given input shape.
        self.input_shape = input_shape  # input_shape would be the shape of a single sample of the input data
        # Define a Sequential model with 3 hidden layers and an output layer.
        self.model = Sequential([
            # First hidden layer with 128 neurons and ReLU activation function. It expects an input as per input shape.
            Dense(128, activation='relu', input_shape=(input_shape,)),  # First hidden layer with 128 neurons
            # Dropout layer to reduce overfitting by randomly setting input units to 0 during training.
            Dropout(0.5),
            # Second hidden layer with 64 neurons and ReLU activation function.
            Dense(64, activation='relu'),  # Second hidden layer with 64 neurons
            # Another dropout layer for regularization.
            Dropout(0.5),
            # Third hidden layer with 64 neurons, also using ReLU activation function.
            Dense(64, activation='relu'),  # Third hidden layer with 64 neurons
            # Final dropout layer.
            Dropout(0.5),  # Dropout layer to prevent overfitting
            # Output layer with 3 neurons for the classification task & softmax activation func for prob distribution.
            Dense(3, activation='softmax')  # Output layer with 3 neurons for the classes
        ])
        # Compile the model with 'adam' optimizer, 'sparse_categorical_crossentropy' loss function
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, training_inputs, labels):
        # Train the model on the provided training data and labels.
        # It runs for 100 epochs with a batch size of 512 and uses 20% of the data for validation.
        self.model.fit(training_inputs, labels, epochs=50, batch_size=512, validation_split=0.2)
        print("Training Complete!\n\n")

    def predict(self, inputs):
        # Predict the class for given inputs.
        # It reshapes the inputs to match the model's input shape, predicts the output and
        # returns the class with the highest probability.
        sample = np.array(inputs).reshape(1, self.input_shape)
        preds = self.model.predict(sample, verbose=True)
        return preds.argmax(axis=1)[0]


class Parser(ABC):
    # DO NOT CHANGE THIS CLASS
    def __init__(self):
        self.train = get_dataset_split("train")
        self.val = get_dataset_split("validation")
        self.test = get_dataset_split("test")
        self._train()

    @abstractmethod
    def _train(self):
        pass

    def predict(self):
        correct_unlabeled, correct_labeled = 0, 0
        num_tokens = 0
        for i, sentence in enumerate(self.test):
            sentence_copy = sentence.copy()
            sentence_copy["head"] = [] * len(sentence["tokens"])
            sentence_copy["deprel"] = [] * len(sentence["tokens"])
            sentence_copy["deps"] = [] * len(sentence["tokens"])
            heads, deprels = self.parse_sentence(sentence_copy)
            sent_correct_unlabeled, sent_correct_labeled = Parser.get_correct(sentence, heads, deprels)
            correct_unlabeled += sent_correct_unlabeled
            correct_labeled += sent_correct_labeled
            num_tokens += len(sentence["tokens"])
            # if i > 10:
            #    break

        return correct_unlabeled / num_tokens, correct_labeled / num_tokens, num_tokens

    @abstractmethod
    def parse_sentence(self, sentence):
        pass

    @classmethod
    def get_correct(self, sentence, heads, deprels):
        assert len(sentence["tokens"]) == len(heads) == len(deprels)
        assert len([h for h in heads if h == 0]) == 1, f"{sentence['tokens']}\n" \
                                                       f"{sentence['xpos']}\n" \
                                                       f"{heads}"
        correct_unlabeled, correct_labeled = 0, 0
        for g_head, g_deprel, p_head, p_deprel in zip(sentence["head"], sentence["deprel"], heads, deprels):
            if g_head == str(p_head):
                correct_unlabeled += 1
                if g_deprel == p_deprel:
                    correct_labeled += 1
        return correct_unlabeled, correct_labeled


class FancyArcStandardParser(Parser):
    # The get_vectors method is responsible for converting the state of the stack and buffer into a fixed-size vector
    # that can be used as input to a neural network.
    def get_vectors(self, stack_upos, buffer_upos, test_param=2):
        # Initialize zero vectors for one-hot encoding of UPOS tags.
        stack_features = [[0] * 18 for _ in range(5)]
        buffer_features = [[0] * 18 for _ in range(5)]

        # Helper function to safely set feature value
        def set_feature(features, pos, upos, param):
            if pos < len(upos):
                features[pos][upos[-(pos + 1)][param]] = 1

        # Set the appropriate index to 1 based on the UPOS tag for each position in the stack and buffer.
        for i in range(5):
            set_feature(stack_features, i, stack_upos, test_param)
            set_feature(buffer_features, i, buffer_upos, test_param)

        # Combine the vectors for all stack and buffer positions into a single feature vector.
        feature_vector = sum(buffer_features[::-1] + stack_features, [])
        return feature_vector

    # The _train method is used to train the neural network model.
    def _train(self):
        # The number of features is set to 10, with each feature being an 18-length vector.
        # Define the size of the input feature vector for the neural network.
        # Each feature represents an 18-dimensional one-hot encoded UPOS tag.
        # 5 features each from the stack and buffer make up the input.

        number_of_feature = 10  # 5 stack + 5 buffer positions
        length_of_feature = 18 * number_of_feature  # total length of the input vector to the neural network
        # Instantiate the feedforward neural network with the input shape.
        self.model = FeedForwardNeuralNetwork(input_shape=length_of_feature)

        # Lists to store the one-hot encoded features and corresponding labels for training.
        one_hot_encoding_upos = []
        labels = []
        # Iterate over each training instance (a parsed sentence with annotations).
        for sentnc in self.train:
            #print("Training in progress..")
            # Initialize the stack for storing UPOS tags during parsing.
            upos_stack = []
            # Prepare the buffer with heads and UPOS tags, reversing them to start from the end of the sentence.
            heads_list = list(reversed(sentnc["head"]))
            upos_list = list(reversed(sentnc["upos"]))
            for idx, pos in enumerate(heads_list):
                # Clean up the 'None' entries from heads and UPOS tags.
                if pos == 'None':
                    heads_list.pop(idx)
                    upos_list.pop(idx)

            # Initialize the buffer which will hold information about the words (index, head, and UPOS tag).
            rows, cols = (len(heads_list), 3)
            buffer_of_upos = [[[0] for _ in range(cols)] for _ in range(rows)]

            # Populate the buffer with the word index, head, and UPOS tag.
            for r in range(rows):
                for c in range(cols):
                    if c == 0:
                        buffer_of_upos[r][c] = len(heads_list) - r
                    elif c == 1:
                        buffer_of_upos[r][c] = heads_list[r]
                    else:
                        buffer_of_upos[r][c] = upos_list[r]

            # Calculate how many transitions we need to predict to parse the sentence.
            remaining_assignments = len(buffer_of_upos) - 1

            # Prime the stack with the first two elements from the buffer if available.
            if remaining_assignments > 0:
                upos_stack.append(buffer_of_upos.pop())
                upos_stack.append(buffer_of_upos.pop())
            # Setting a limit for while loop
            count = 2 * len(buffer_of_upos)

            # Iterate and build the transition-based parse until all words have a head (i.e., a dependency arc).
            while remaining_assignments > 0:
                # If count reaches 0, break out to avoid infinite looping.
                count -= 1
                if count < 1:
                    break

                # Check if the top of the stack has any dependents left in the buffer.
                no_dependents = True
                for elem in buffer_of_upos:
                    if str(elem[1]) == str(upos_stack[-1][0]):
                        no_dependents = False
                        break

                # Left Arc operation: Assign the second item on the stack as a dependent of the first item.
                if len(upos_stack) > 1 and (str(upos_stack[-2][1]) == str(upos_stack[-1][0])):
                    feature_vector = self.get_vectors(upos_stack, buffer_of_upos)
                    one_hot_encoding_upos.append(feature_vector)
                    labels.append(0)
                    upos_stack.pop(-2)
                    remaining_assignments -= 1

                # Right Arc operation: Assign the first item on the stack as a dependent of the second item.
                elif len(upos_stack) > 1 and (
                        str(upos_stack[-2][0]) == str(upos_stack[-1][1]) and no_dependents):
                    feature_vector = self.get_vectors(upos_stack, buffer_of_upos)
                    one_hot_encoding_upos.append(feature_vector)
                    labels.append(1)
                    upos_stack.pop()
                    remaining_assignments -= 1

                # Shift operation: Move the next item from the buffer onto the stack.
                elif len(buffer_of_upos) > 0:
                    feature_vector = self.get_vectors(upos_stack, buffer_of_upos)
                    one_hot_encoding_upos.append(feature_vector)
                    labels.append(2)
                    upos_stack.append(buffer_of_upos.pop())
        # Convert lists to numpy arrays for the model training.
        x_train = np.array(one_hot_encoding_upos)
        y_train = np.array(labels)
        # Train the model with the feature vectors and labels.
        self.model.train(x_train, y_train)

    # The parse_sentence method uses the trained model to parse a new sentence.
    def parse_sentence(self, sentence):
        # Finalize the parsing by assigning heads and dependency relations to each token in the sentence.
        # Adding heads related to None and ensuring there is a root in the tree.
        heads = [None] * len(sentence["tokens"])
        deprels = ["unk"] * len(sentence["tokens"])
        stack = []      # Initialize the stack for parsing.
        # Reverse the list of UPOS tags for processing and remove specific tags (e.g., tag with index 13).
        upos_list = list(reversed(sentence["upos"]))
        for j, pos in enumerate(upos_list):
            if pos == 13:
                upos_list.pop(j)
        # Initialize the buffer that will store indices and UPOS tags.
        rows, cols = (len(upos_list), 2)
        buffer = [[[0] for _ in range(cols)] for _ in range(rows)]

        # Populate the buffer with index and UPOS information.
        for r in range(rows):
            for c in range(cols):
                if c == 0:
                    buffer[r][c] = len(upos_list) - r
                elif c == 1:
                    buffer[r][c] = upos_list[r]

        # Create a list to store temporary heads during the parsing process.
        temporary_heads = [-1] * len(buffer)

        # Determine the number of transitions required for parsing the sentence.
        work_left = len(buffer) - 1

        # Push the first two elements onto the stack if they are available.
        if work_left > 0:
            stack.append(buffer.pop())
            stack.append(buffer.pop())
        # Set a counter to avoid infinite loops.
        count = 2 * len(buffer)

        # Perform parsing actions until all words have been assigned a head.
        while work_left > 0:
            # Prevent infinite looping by breaking if the count reaches zero.
            count -= 1
            if count < 1:
                break

            # Ensure that there are always at least two elements on the stack.
            if len(stack) < 2:
                for fill in range(len(stack), 2):
                    stack.append(buffer.pop())

            # Generate feature vectors for the current configuration and predict the next transition.
            feature_vector = self.get_vectors(stack, buffer, test_param=1)
            transition = self.model.predict(feature_vector)

            # Execute the Left Arc transition, assigning the second item on the stack as a dependent of the first.
            if transition == 0:
                temporary_heads[stack[-2][0] - 1] = stack[-1][0]
                stack.pop(-2)
                work_left -= 1

            # Execute the Right Arc transition, assigning the first item on the stack as a dependent of the second.
            elif transition == 1:
                temporary_heads[stack[-1][0] - 1] = stack[-2][0]
                stack.pop(-1)
                work_left -= 1

            # Execute the Shift transition, moving the next item from the buffer onto the stack.
            elif transition == 2:
                if len(buffer) > 0:
                    stack.append(buffer.pop())

        # If there's one element left on the stack, assign it as the root.
        if len(stack) == 1:
            temporary_heads[stack[-1][0] - 1] = 0

        # Reverse the temporary heads to align with the original sentence order.
        temporary_heads = list(reversed(temporary_heads))
        # Assign heads to tokens, ignoring specific UPOS tags.
        upos_list = list(sentence["upos"])
        for idx, pos in enumerate(upos_list):
            if pos == 13:
                heads[idx] = None
            else:
                if len(temporary_heads) > 0:
                    heads[idx] = temporary_heads.pop()

        # Ensure there is at least one root in the parsed sentence; if not, assign one.
        verb_found = False
        for idx in range(len(heads)):
            if heads[idx] == 0:
                verb_found = True
                break
        if not verb_found:
            if heads[0] is None or heads[0] == -1:
                heads[0] = 0
            else:
                # If the first token is not suitable, assign root to the head of the first token.
                heads[heads[0] - 1] = 0
        return heads, deprels


if __name__ == "__main__":
    for parser in [FancyArcStandardParser]:
        parser = parser()
        uas, las, num_tokens = parser.predict()
        print(f"{parser.__class__.__name__:20} Unlabeled Accuracy (UAS): {uas:.3f} [{num_tokens} tokens]")
        print(f"{parser.__class__.__name__:20} Labeled Accuracy (UAS):   {las:.3f} [{num_tokens} tokens]")
        print()
