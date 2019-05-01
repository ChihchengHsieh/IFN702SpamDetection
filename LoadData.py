
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocessingInputData, matchingURL, mapFromWordToIdx, CreateDatatset, CreateTweetsWithUserInfoDatatset
import itertools
import pickle
from Constants import specialTokenList
import nltk
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
import re





def loadingTweetsAndUserInfoData(args):

    if not os.path.isfile(os.path.join(args.dataset, args.pickle_name)):

        if not os.path.isfile(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx)):

            print("Loading ",  os.path.join(
                args.dataset, "FullTweetsWithUserInfoSelected.html"), " to do the Proprocessing")

            df = pd.read_html(os.path.join(
                args.dataset, "FullTweetsWithUserInfoSelected.html"))

            columnNames = list(df[0].loc[0])
            df = df[0].iloc[1:, :]
            df.columns = columnNames

            def preprocessingInputTextData(colName):
                input = df[colName]
                ps = nltk.stem.PorterStemmer()
                tknzr = TweetTokenizer()
                allText = [i for i in input]
                preprocessedText = [[ps.stem(word) for word in tknzr.tokenize(re.sub(r'\d+', '', re.sub(r"http\S+|www.\S+", matchingURL,
                                                                                                        sentence)).lower()) if word not in nltk.corpus.stopwords.words('english') and len(word) >= 3] for sentence in allText]
                df[colName] = preprocessedText

            def fillingNullValue(colName):
                if args.preprocessingStra[colName]['fillingNullMethod'] == filling_method.MOST_COMMON:
                    df[colName] = df[colName].astype('float')
                    df[colName].fillna(df[colName].mean(), inplace=True)
                elif args.preprocessingStra[colName]['fillingNullMethod'] == filling_method.MEAN:
                    df[colName] = df[colName].astype('category')
                    df[colName].fillna(df[colName].astype(
                        'category').describe()['top'], inplace=True)
                elif args.preprocessingStra[colName]['fillingNullMethod'] == filling_method.CERTAIN_VALUE:
                    df[colName] = df[colName].astype('category')
                    df[colName] = df[colName].cat.add_categories(
                        [args.preprocessingStra[colName]['fillingNullValue']])
                    df[colName].fillna(args.preprocessingStra[colName]
                                       ['fillingNullValue'], inplace=True)

            def TweetsWithUserInfoPreprocessing():
                for colName in args.preprocessingStra.keys():
                    print("Preprocessing column: ", colName)
                    for step in args.preprocessingStra[colName]['steps']:
                        if not step is None:
                            step(colName)
            ############### Hiding Preprocessing Strategy ###############

            args.preprocessingStra = defaultdict(dict)
            args.preprocessingStra['text']['steps'] = [
                preprocessingInputTextData]
            args.preprocessingStra["numberOfHashtags_c"]['steps'] = [None]
            args.preprocessingStra['favorite_count']['steps'] = [None]
            args.preprocessingStra['retweet_count']['steps'] = [None]
            args.preprocessingStra['possibly_sensitive'] = {
                'fillingNullMethod': filling_method.CERTAIN_VALUE,
                'fillingNullValue': 'UNKNOWN',
                'steps': [fillingNullValue],
            }
            args.preprocessingStra['followers_count']['steps'] = [None]
            args.preprocessingStra['friends_count']['steps'] = [None]
            # args.preprocessingStra['description']= {
            #    'steps': [ fillingNullValue,preprocessingInputTextData],
            #     'fillingNullMethod': filling_method.CERTAIN_VALUE,
            #     'fillingNullValue': 'NULLDescription'
            # }
            args.preprocessingStra['default_profile']['steps'] = [None]
            args.preprocessingStra['default_profile_image']['steps'] = [None]
            args.preprocessingStra['favourites_count']['steps'] = [None]
            args.preprocessingStra['listed_count']['steps'] = [None]
            args.preprocessingStra['statuses_count']['steps'] = [None]
            args.preprocessingStra['verified']['steps'] = [None]

            print('Preprocessing Strategy Set')

            #############################################################

            print('Start Preprocessing...')
            TweetsWithUserInfoPreprocessing()  # Apply inplace preprocessing
            df = pd.get_dummies(df, drop_first=True, columns=[
                                'possibly_sensitive', 'default_profile', 'default_profile_image', 'verified'])

            print('Spliting Datasets...')
            X_train, X_test, Y_train, Y_test = train_test_split(df.drop(
                'maliciousMark', axis=1), df['maliciousMark'], test_size=args.validation_portion, stratify=df['maliciousMark'],  random_state=args.random_seed)
            X_validation, X_test, Y_validation, Y_test = train_test_split(
                X_test, Y_test, test_size=args.test_portion, stratify=Y_test, random_state=args.random_seed)

            print('Creating Tweets_text')
            tweets_text = nltk.Text(list(itertools.chain(*X_train['text'])))

            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "wb") as fp:  # Pickling
                pickle.dump([X_train, X_validation, X_test,
                             Y_train, Y_validation, Y_test,  tweets_text], fp)
                print("The Pickle Data beforeMapToIdx Dumped to:", os.path.join(
                    args.dataset, args.pickle_name_beforeMapToIdx))

        else:
            print("Loading Existing BeforeMapToIdx file for Tweets and User: ",
                  os.path.join(args.dataset, args.pickle_name_beforeMapToIdx))
            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "rb") as fp:   # Unpickling
                [X_train, X_validation, X_test,
                 Y_train, Y_validation, Y_test,  tweets_text] = pickle.load(fp)

        args.vocab_size = args.vocab_size or len(tweets_text.tokens)
        if args.vocab_size:  # and this if expression
            tweets_text.tokens = specialTokenList + \
                [w for w, _ in tweets_text.vocab().most_common(
                    args.vocab_size - len(specialTokenList))]
        else:
            tweets_text.tokens = specialTokenList + tweets_text.tokens
        args.vocab_size = len(tweets_text.tokens)  # change the vacab_size
        

        print("Maping Word To Idx: training set")
        X_train['text'] = mapFromWordToIdx(X_train['text'], tweets_text)
        print("Maping Word To Idx: validation set")
        X_validation['text'] = mapFromWordToIdx(
            X_validation['text'], tweets_text)
        print("Maping Word To Idx: test set")
        X_test['text'] = mapFromWordToIdx(X_test['text'], tweets_text)


        print("Creating Torch Datasets...")
        training_dataset = CreateTweetsWithUserInfoDatatset(
            X_train, list(map(int, list(Y_train))))
        validation_dataset = CreateTweetsWithUserInfoDatatset(
            X_validation, list(map(int, list(Y_validation))))
        test_dataset = CreateTweetsWithUserInfoDatatset(
            X_test, list(map(int, list(Y_test))))

        print("Dumping Data")

        with open(os.path.join(args.dataset, args.pickle_name), "wb") as fp:   # Pickling
            pickle.dump([training_dataset, validation_dataset,
                         test_dataset, tweets_text], fp)
            print("The Pickle Data Dumped to: ", os.path.join(args.dataset, args.pickle_name))

    else:
        print("Loading Existing File: ", os.path.join(args.dataset, args.pickle_name))
        with open(os.path.join(args.dataset, args.pickle_name), "rb") as fp:   # Unpickling
            training_dataset, validation_dataset, test_dataset, tweets_text = pickle.load(
                fp)

    args.vocab_size = len(tweets_text.tokens)
    args.num_extra_info = len(training_dataset[0][1])
    args.num_features = len(training_dataset[0][1]) + 1

    return training_dataset, validation_dataset, test_dataset, tweets_text


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
                print("Loading Honeypot dataset")
                df_Nonspammer = pd.read_csv(
                    "./Honeypot/nonspam_tweets.csv", encoding="ISO-8859-1")[['text', 'maliciousMark']]
                df_Spammer = pd.read_csv(
                    "./Honeypot/spam_tweets.csv", encoding="ISO-8859-1")[['text', 'maliciousMark']]
                df = pd.concat([df_Nonspammer, df_Spammer])
                del df_Nonspammer, df_Spammer
            else:
                print("Please input a valid dataset name: HSpam14, Honeypot")
                raise ValueError

            print("Data Splitation")

            X_train, X_test, Y_train, Y_test = train_test_split(
                df['text'], df['maliciousMark'], test_size=args.validation_portion, stratify=df['maliciousMark'], random_state=args.random_seed)
            X_validation, X_test, Y_validation, Y_test = train_test_split(
                X_test, Y_test, test_size=args.test_portion, stratify=Y_test, random_state=args.random_seed)

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

            resultTextbox.insert(
                "end", ("Loading Origin Data and do the Proprocessing\n"))
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
                resultTextbox.insert(
                    "end", ("Please input a valid dataset name: HSpam14, Honeypot\n"))
                raise ValueError

            window.update_idletasks()
            resultTextbox.insert("end", ("Data Splitation\n"))
            window.update_idletasks()
            X_train, X_test, Y_train, Y_test = train_test_split(
                df['text'], df['maliciousMark'], test_size=args.validation_portion, stratify=df['maliciousMark'], random_state=args.random_seed)
            X_validation, X_test, Y_validation, Y_test = train_test_split(
                X_test, Y_test, test_size=args.test_portion, stratify=Y_test, random_state=args.random_seed)

            resultTextbox.insert(
                "end", ("Number of Training Data: " + str(len(X_train)) + "\n"))
            resultTextbox.insert(
                "end", ("Number of Validation Data: " + str(len(X_validation)) + "\n"))
            resultTextbox.insert(
                "end", ("Number of Test Data: " + str(len(X_test)) + "\n"))
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

            resultTextbox.insert(
                "end", ("Original Vocab Size: " + str(len(text.tokens)) + "\n"))
            window.update_idletasks()

            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "wb") as fp:  # Pickling
                pickle.dump([X_train, X_validation, X_test,
                             Y_train, Y_validation, Y_test,  text], fp)
                resultTextbox.insert("end", ("The Pickle Data beforeMapToIdx Dumped to:" + str(os.path.join(
                    args.dataset, args.pickle_name_beforeMapToIdx)) + "\n"))
                window.update_idletasks()

        else:
            resultTextbox.insert("end", ("Loading Existing BeforeMapToIdx file: " + str(
                os.path.join(args.dataset, args.pickle_name_beforeMapToIdx)) + "\n"))
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

        resultTextbox.insert("end", ("Validation set map to Idx\n"))
        window.update_idletasks()

        validation_dataset = CreateDatatset(
            X_validation, mapFromWordToIdx(X_validation, text), list(map(int, list(Y_validation))))

        resultTextbox.insert("end", ("Test set map to Idx\n"))
        window.update_idletasks()

        test_dataset = CreateDatatset(
            X_test, mapFromWordToIdx(X_test, text), list(map(int, list(Y_test))))

        resultTextbox.insert("end", ("Dumping Data\n"))
        window.update_idletasks()

        with open(os.path.join(args.dataset, args.pickle_name), "wb") as fp:   # Pickling
            pickle.dump([training_dataset, validation_dataset,
                         test_dataset, text], fp)
            resultTextbox.insert("end", ("The Pickle Data Dumped\n"))
            window.update_idletasks()

    else:
        resultTextbox.insert(
            "end", ("Loading Existing File: " + args.pickle_name + "\n"))
        window.update_idletasks()
        with open(os.path.join(args.dataset, args.pickle_name), "rb") as fp:   # Unpickling
            training_dataset, validation_dataset, test_dataset, text = pickle.load(
                fp)
            args.vocab_size = len(text.tokens)

    return training_dataset, validation_dataset, test_dataset, text
