
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocessingInputData, matchingURL, mapFromWordToIdx, CreateDatatset
import itertools
import pickle
from Constants import specialTokenList
import nltk


def loadingData(args):

    if not os.path.isfile(os.path.join(args.dataset, args.pickle_name)):

        if not os.path.isfile(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx)):

            print("Loading Origin Data and do the Proprocessing")

            if args.dataset == "HSpam14":
                print("Loading HSpam14 dataset")
                df = pd.read_html(os.path.join(args.dataset, "FullDataFromSQLHSpam14.html"))[
                    0].iloc[1:, :]
                df.columns = ['text', 'maliciousMark']
            elif args.dataset == "Honeypot":
                df_Nonspammer = pd.read_csv(
                    "./Honeypot/nonspam_tweets.csv", encoding="ISO-8859-1")[['text', 'maliciousMark']]
                df_Spammer = pd.read_csv(
                    "./Honeypot/spam_tweets.csv", encoding="ISO-8859-1")[['text', 'maliciousMark']]
                df = pd.concat([df_Nonspammer, df_Spammer])
                del df_Nonspammer, df_Spammer
                print("Loading Honeypot dataset")
            else:
                print("Please input a valid dataset name: HSpam14, Honeypot")
                raise ValueError

            print("Data Splitation")

            X_train, X_test, Y_train, Y_test = train_test_split(
                df['text'], df['maliciousMark'], test_size=args.validation_portion, stratify=df['maliciousMark'], random_state=64)
            X_validation, X_test, Y_validation, Y_test = train_test_split(
                X_test, Y_test, test_size=args.test_portion, stratify=Y_test, random_state=64)

            print("Number of Training Data: ", len(X_train))
            print("Number of Validation Data: ", len(X_validation))
            print("Number of Test Data: ", len(X_test))

            print("Preprocessing X_train")

            X_train = preprocessingInputData(X_train)

            print("Preprocessing X_validation")

            X_validation = preprocessingInputData(X_validation)

            print("Preprocessing X_test")

            X_test = preprocessingInputData(X_test)

            print("Generating text")

            # Preparing the dictionary
            text = nltk.Text(list(itertools.chain(*X_train)))

            print("Original Vocab Size: ", len(text.tokens))

            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "wb") as fp:  # Pickling
                pickle.dump([X_train, X_validation, X_test,
                             Y_train, Y_validation, Y_test,  text], fp)
                print("The Pickle Data beforeMapToIdx Dumped to:", os.path.join(
                    args.dataset, args.pickle_name_beforeMapToIdx))
        else:
            print("Loading Existing BeforeMapToIdx file: ",
                  os.path.join(args.dataset, args.pickle_name_beforeMapToIdx))
            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "rb") as fp:   # Unpickling
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

        with open(os.path.join(args.dataset, args.pickle_name), "wb") as fp:   # Pickling
            pickle.dump([training_dataset, validation_dataset,
                         test_dataset, text], fp)
            print("The Pickle Data Dumped")

    else:
        print("Loading Existing File: ", args.pickle_name)
        with open(os.path.join(args.dataset, args.pickle_name), "rb") as fp:   # Unpickling
            training_dataset, validation_dataset, test_dataset, text = pickle.load(
                fp)
            args.vocab_size = len(text.tokens)

    return training_dataset, validation_dataset, test_dataset, text

























####################################################################################


def TkloadingData(args, resultTextbox, window):

    if not os.path.isfile(os.path.join(args.dataset, args.pickle_name)):

        if not os.path.isfile(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx)):

            resultTextbox.insert("end", ("Loading Origin Data and do the Proprocessing\n"))
            window.update_idletasks()

            if args.dataset == "HSpam14":
                resultTextbox.insert("end", ("Loading HSpam14 dataset\n"))
                df = pd.read_html(os.path.join(args.dataset, "FullDataFromSQLHSpam14.html"))[
                    0].iloc[1:, :]
                df.columns = ['text', 'maliciousMark']
            elif args.dataset == "Honeypot":
                df_Nonspammer = pd.read_csv(
                    "./Honeypot/nonspam_tweets.csv", encoding="ISO-8859-1")[['text', 'maliciousMark']]
                df_Spammer = pd.read_csv(
                    "./Honeypot/spam_tweets.csv", encoding="ISO-8859-1")[['text', 'maliciousMark']]
                df = pd.concat([df_Nonspammer, df_Spammer])
                del df_Nonspammer, df_Spammer
                resultTextbox.insert("end", ("Loading Honeypot dataset\n"))
            else:
                resultTextbox.insert("end", ("Please input a valid dataset name: HSpam14, Honeypot\n"))
                raise ValueError

            window.update_idletasks()
            resultTextbox.insert("end", ("Data Splitation\n"))
            window.update_idletasks()       
            X_train, X_test, Y_train, Y_test = train_test_split(
                df['text'], df['maliciousMark'], test_size=args.validation_portion, stratify=df['maliciousMark'], random_state=64)
            X_validation, X_test, Y_validation, Y_test = train_test_split(
                X_test, Y_test, test_size=args.test_portion, stratify=Y_test, random_state=64)

            resultTextbox.insert("end", ("Number of Training Data: " + str(len(X_train)) + "\n" ))
            resultTextbox.insert("end", ("Number of Validation Data: "+ str(len(X_validation)) + "\n"))
            resultTextbox.insert("end", ("Number of Test Data: " + str(len(X_test)) + "\n" ))
            window.update_idletasks()

            resultTextbox.insert("end", ("Preprocessing X_train\n"))
            window.update_idletasks()

            X_train = preprocessingInputData(X_train)

            resultTextbox.insert("end", ("Preprocessing X_validation\n"))
            window.update_idletasks()

            X_validation = preprocessingInputData(X_validation)

            resultTextbox.insert("end", ("Preprocessing X_test\n"))
            window.update_idletasks()

            X_test = preprocessingInputData(X_test)

            resultTextbox.insert("end", ("Generating text\n"))
            window.update_idletasks()

            # Preparing the dictionary
            text = nltk.Text(list(itertools.chain(*X_train)))

            resultTextbox.insert("end", ("Original Vocab Size: " + str(len(text.tokens)) + "\n" ))
            window.update_idletasks()

            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "wb") as fp:  # Pickling
                pickle.dump([X_train, X_validation, X_test,
                             Y_train, Y_validation, Y_test,  text], fp)
                resultTextbox.insert("end", ("The Pickle Data beforeMapToIdx Dumped to:" + str(os.path.join(
                    args.dataset, args.pickle_name_beforeMapToIdx)) + "\n"))
                window.update_idletasks()
                
        else:
            resultTextbox.insert("end", ("Loading Existing BeforeMapToIdx file: " + str(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx)) + "\n"))
            window.update_idletasks()            
            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "rb") as fp:   # Unpickling
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

        resultTextbox.insert("end", ("Generating Datasets\n"))

        resultTextbox.insert("end", ("Training set map to Idx\n"))
        window.update_idletasks()

        training_dataset = CreateDatatset(
            X_train, mapFromWordToIdx(X_train, text), list(map(int, list(Y_train))))

        resultTextbox.insert("end",("Validation set map to Idx\n"))
        window.update_idletasks()

        validation_dataset = CreateDatatset(
            X_validation, mapFromWordToIdx(X_validation, text), list(map(int, list(Y_validation))))

        resultTextbox.insert("end",("Test set map to Idx\n"))
        window.update_idletasks()

        test_dataset = CreateDatatset(
            X_test, mapFromWordToIdx(X_test, text), list(map(int, list(Y_test))))

        resultTextbox.insert("end",("Dumping Data\n"))
        window.update_idletasks()

        with open(os.path.join(args.dataset, args.pickle_name), "wb") as fp:   # Pickling
            pickle.dump([training_dataset, validation_dataset,
                         test_dataset, text], fp)
            resultTextbox.insert("end",("The Pickle Data Dumped\n"))
            window.update_idletasks()

    else:
        resultTextbox.insert("end",("Loading Existing File: " + args.pickle_name + "\n"))
        window.update_idletasks()
        with open(os.path.join(args.dataset, args.pickle_name), "rb") as fp:   # Unpickling
            training_dataset, validation_dataset, test_dataset, text = pickle.load(
                fp)
            args.vocab_size = len(text.tokens)

    return training_dataset, validation_dataset, test_dataset, text



