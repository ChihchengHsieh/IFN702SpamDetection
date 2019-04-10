
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocessingInputData, matchingURL, mapFromWordToIdx, CreateDatatset
import itertools
import pickle
from Constants import specialTokenList
import nltk


def loadingData(args):

    if not os.path.isfile(args.pickle_name):

        if not os.path.isfile(args.pickle_name_beforeMapToIdx):

            print("Loading Origin Data and do the Proprocessing")

            if args.full_data:
                df = pd.read_html('FullDataFromSQLHSpam14.html')[0].iloc[1:, :]
            else:
                df_noSPammer = pd.read_html(
                    'textMaliciousMark_10th_NotSpammer.html')
                df_noSPammer = df_noSPammer[0][1:]
                df_Spammer = pd.read_html(
                    'textMaliciousMark_10th_Spammer.html')
                df_Spammer = df_Spammer[0][1:]
                df = pd.concat([df_Spammer, df_noSPammer])
                del df_noSPammer, df_Spammer

            print("Data Splitation")

            X_train, X_test, Y_train, Y_test = train_test_split(
                df[0], df[1], test_size=args.validation_portion, stratify=df[1], random_state=64)
            X_validation, X_test, Y_validation, Y_test  = train_test_split(
                X_test, Y_test, test_size=args.test_portion, stratify=Y_test, random_state=64)

            print("Number of Training Data: ", X_train)
            print("Number of Validation Data: ", X_validation)
            print("Number of Test Data: ", X_test)

            print("Preprocessing X_train")

            # if I change the vocab size, I will not change this one..
            X_train = preprocessingInputData(X_train)

            print("Preprocessing X_validation")

            X_validation = preprocessingInputData(X_validation)

            print("Preprocessing X_test")

            X_test = preprocessingInputData(X_test)

            print("Generating text")

            # Preparing the dictionary
            text = nltk.Text(list(itertools.chain(*X_train)))

            with open(args.pickle_name_beforeMapToIdx, "wb") as fp:  # Pickling
                pickle.dump([X_train, X_validation, X_test,
                             Y_train, Y_validation, Y_test,  text], fp)
                print("The Pickle Data beforeMapToIdx Dumped")
        else:
            print("Loading Existing BeforeMapToIdx file: ",
                  args.pickle_name_beforeMapToIdx)
            with open(args.pickle_name_beforeMapToIdx, "rb") as fp:   # Unpickling
                [X_train, X_validation, X_test, Y_train,
                    Y_validation, Y_test, text] = pickle.load(fp)

        # The vocab_size will start to affect the data from here
        args.vocab_size = args.vocab_size or len(text.tokens)
        if args.vocab_size:  # and this if expression
            text.tokens = specialTokenList + \
                [w for w, _ in text.vocab().most_common(
                    args.vocab_size - len(specialTokenList))]
        else:
            text.tokens = specialTokenList + text.tokens
        args.vocab_size = len(text.tokens)  # change the vacab_size

        print("Generating Datasets")

        print("Training set map to Idx")

        training_dataset = CreateDatatset(
            X_train, mapFromWordToIdx(X_train, text), list(map(int, list(Y_train))))

        print("Validation set map to Idx")

        validation_dataset = CreateDatatset(
            X_validation, mapFromWordToIdx(X_validation, text), list(map(int, list(Y_validation))))

        print("Test set map to Idx")

        test_dataset = CreateDatatset(
            X_test, mapFromWordToIdx(X_test, text), list(map(int, list(Y_test))))

        print("Dumping Data")

        with open(args.pickle_name, "wb") as fp:   # Pickling
            pickle.dump([training_dataset, validation_dataset,
                         test_dataset, text], fp)
            print("The Pickle Data Dumped")

    else:
        print("Loading Existing File: ", args.pickle_name)
        with open(args.pickle_name, "rb") as fp:   # Unpickling
            training_dataset, validation_dataset, test_dataset, text = pickle.load(
                fp)
            args.vocab_size = len(text.tokens)

    return training_dataset, validation_dataset, test_dataset, text
