import joblib

pred_Dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
numPos = 0
numNeg = 0
numNeutral = 0

def loadModel():
    vectorizer = joblib.load('Model/count_vectorizer.pkl')
    model = joblib.load('Model/NaiveBayes_model.pkl')
    return vectorizer, model


def loadData(name):
    lines = []
    text = open("Data/" + name + ".txt", encoding='utf-8')
    for line in text:
        if line:
            lines.append(line)
    text.close()
    return lines


def predict_Write(vectorizer, model, contents):
    for content in contents:
        content = [content]

        sample_vec = vectorizer.transform(content)
        pred = model.predict(sample_vec)
        prob = model.predict_proba(sample_vec)
        if abs(prob[0][0] - prob[0][1] )<= 0.1:
            print(prob)
            pred = [2]
        f = open("Result/" + str(pred_Dict[pred[0]]) + '.txt', mode='a')
        if content[0] != '\n':
            f.write(content[0]+"\n\n")
        f.close()


def main():
    vectorizer, model = loadModel()
    files = ['2018', '2019', '2020', '2021', '2022']
    for file in files:
        contents = loadData(file)
        predict_Write(vectorizer, model, contents)


if __name__ == '__main__':
    main()
