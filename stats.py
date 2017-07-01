def compase_results_with_train_labels(prefix, results, mapping):
    exact_mappings = 0
    for fname, prediction in results:
        prediction = one_hot_to_labels(prediction, classifications)
        actual     = one_hot_to_labels(mapping[fname], classifications)
        if actual == prediction:
            exact_mappings +=1
            print("Correctly predicted: "+prediction)
        else:
            print("Predicted: "+prediction)
            print("was " + actual +"\r\n")
    print("Total exact mappings: " + str(exact_mappings))
