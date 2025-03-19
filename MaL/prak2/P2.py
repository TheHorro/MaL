import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import regularizers, optimizers, callbacks
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
import gradio as gr

np.random.seed(42)
tf.random.set_seed(42) 
# keras.utils.set_random_seed(42)

(X, Y) = load_breast_cancer(return_X_y=True)
TrainSet = np.random.choice(X.shape[0],int(X.shape[0]*0.70), replace=False)
TestSet  = np.delete(np.arange(0, len(Y) ), TrainSet) 
XTrain   = X[TrainSet,:]
YTrain   = Y[TrainSet]
XTest    = X[TestSet,:]
YTest    = Y[TestSet]
ValSet   = np.random.choice(XTest.shape[0],int(XTest.shape[0]*0.5),replace=False)
TestSet  = np.delete(np.arange(0, len(YTest) ), ValSet) 
XVal     = X[ValSet,:]
YVal     = Y[ValSet]
XTest    = X[TestSet,:]
YTest    = Y[TestSet]

def myPredict(ep, pat, opt, rate, l2val):
    match opt:
        case 'adam': opti = optimizers.Adam(learning_rate=rate)
        case 'SGD': opti = optimizers.SGD(learning_rate=rate)
        case 'RMSprop': opti = optimizers.RMSprop(learning_rate=rate)

    myANN = Sequential()
    myANN.add(BatchNormalization())
    myANN.add(Dense(60, activation='relu',
                    kernel_regularizer=regularizers.l2(l2val),
                    input_dim=XTrain.shape[1]))
    myANN.add(Dense(30, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2val)))
    myANN.add(Dense(20, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2val)))
    myANN.add(Dense(10, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2val)))
    myANN.add(Dense( 1,   activation='sigmoid',
                    kernel_regularizer=regularizers.l2(l2val)))
    myANN.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])

    earlystop  = callbacks.EarlyStopping(monitor='val_loss', patience=pat, verbose=False, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint('bestW.weights.h5', monitor='val_loss', verbose=False, save_weights_only=True, save_best_only=True)
    callbacksList = [checkpoint, earlystop]
    
    myANN.fit(XTrain,YTrain, epochs=ep, validation_data=(XVal, YVal), callbacks=callbacksList, verbose=True)
    
    myANN.load_weights('bestW.weights.h5')
    myANN.save('model.h5')

    return calcResults(myANN.predict(XTest))

def bestPredict(opt):
    match opt:
        case 'adam': model = 'adam.h5'
        case 'SGD': model = 'sgd.h5'
        case 'RMSprop': model = 'rms.h5'

    myANN = tf.keras.models.load_model(model)
    # load current model in model.h5 to use with predictValidation
    myANN.save('model.h5') 
    return calcResults(myANN.predict(XTest))

def calcResults(yp):
    # round to 0 or 1 for classification
    yp = np.round(yp) 
    error = np.mean(np.abs(yp - YTest))
    yp = yp.reshape(yp.shape[0])
    diff = yp - YTest
    errcnt = np.count_nonzero(diff)
    acc = 100 - 100/YTest.shape[0] * errcnt
    return errcnt, error, acc

def predictValidation(index):
    myANN = tf.keras.models.load_model('model.h5')
    if myANN == None:
        print('kein ANN bisher generiert!')
        return False
    yp = np.round(myANN.predict(XVal))
    return yp[index] == YVal[index]

with gr.Blocks() as demo:
    with gr.Group():
        gr.Markdown('# ANN train config')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=200):
                with gr.Row():
                    ep = gr.Number(label='Epochs', value=1000)
                    pat = gr.Number(label='Patience', value=20)
                opt      = gr.Dropdown(choices=['adam', 'SGD', 'RMSprop'], value='adam',label='Optimizer')
                rate     = gr.Slider(minimum=1e-4, maximum=3e-3, step=1e-5, value=5e-4, label='Lernrate')
                l2val    = gr.Slider(minimum=1e-4, maximum=1e-3, step=1e-5, value=5e-4, label='L2-Regularizierung' )
                bn_train = gr.Button("Train")
            with gr.Column(scale=1, min_width=200):
                errcnt = gr.Textbox(label='Error Count')
                error = gr.Textbox(label="Mean Absolute Error")
                acc   = gr.Textbox(label='Accuracy (%)')
                bn_train.click(fn=myPredict, inputs=[ep, pat, opt, rate, l2val], outputs=[errcnt, error, acc])
    with gr.Group():
        gr.Markdown('# Test ValidationSet')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=200):
                index = gr.Slider(minimum=0, maximum=XVal.shape[0]-1, step=1, value=0, label='Index from ValidationSet')
                bn_val = gr.Button('Predict from Validation')
            with gr.Column(scale=1, min_width=200):
                res = gr.Text(label='Ergebnis Validation Test')
                bn_val.click(fn=predictValidation, inputs=[index], outputs=[res])
    with gr.Group():
        gr.Markdown('# Load Best Models')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=200):
                bestopt     = gr.Dropdown(choices=['adam', 'SGD', 'RMSprop'], value='adam',label='Optimizer')
            with gr.Column(scale=1, min_width=200):
                bn_loadBest = gr.Button('load best ANN')
                bn_loadBest.click(fn=bestPredict, inputs=[bestopt], outputs=[errcnt, error, acc])

demo.launch()