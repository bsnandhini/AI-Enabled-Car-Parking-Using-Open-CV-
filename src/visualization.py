import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Project base directory 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output directory 
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_model_loss(History):
    plt.figure()

    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])

    plt.savefig(
        os.path.join(OUTPUT_DIR, "model_loss.png")
    )

    plt.show()


def plot_model_accuracy(History):
    plt.figure() 

    plt.plot(History.history['accuracy']) 
    plt.plot(History.history['val_accuracy']) 

    plt.title('Model Accuracy') 
    plt.ylabel('Accuracy') 
    plt.xlabel('Epochs') 
    plt.legend(['train', 'test']) 
    plt.savefig( 
        os.path.join(OUTPUT_DIR, "model_accuracy.png") 
        ) 
    plt.show()



def plot_properly_classified( x_test, y_test, pred_digits, le ):
    warnings.filterwarnings('always') 
    warnings.filterwarnings('ignore')

    prop_class = []


    for i in range(len(y_test)): 
        if np.argmax(y_test[i]) == pred_digits[i]: 
            prop_class.append(i) 
            if len(prop_class) == 8: 
                break

    count = 0 
    fig, ax = plt.subplots(4, 2) 
    fig.set_size_inches(15, 15)

    for i in range(4):
        for j in range(2):
            ax[i, j].imshow(
                x_test[prop_class[count]]
            )

            ax[i, j].set_title( "Predicted : " + str( le.inverse_transform( [pred_digits[prop_class[count]]] ) ) ) 

            plt.tight_layout()

            count += 1

    plt.savefig( os.path.join( OUTPUT_DIR, "properly_classified_images.png" ) )

    plt.show()

def plot_misclassified( x_test, y_test, pred_digits, le ):
    warnings.filterwarnings('always') 
    warnings.filterwarnings('ignore') 
    
    mis_class = []

    for i in range(len(y_test)): 
        if not np.argmax(y_test[i]) == pred_digits[i]: 
            mis_class.append(i) 
            if len(mis_class) == 8: 
                break

    count = 0

    fig, ax = plt.subplots(4, 2) 
    fig.set_size_inches(15, 15) 
    for i in range(4): 
        for j in range(2):
            ax[i, j].imshow( x_test[mis_class[count]] )

            ax[i, j].set_title( "Predicted : " + str( le.inverse_transform( [pred_digits[mis_class[count]]] ) ) )

            plt.tight_layout()
            count += 1

    plt.savefig( os.path.join( OUTPUT_DIR, "misclassified_images.png" ) ) 
    plt.show()

def plot_confusion_matrix( y_true, y_pred ):
    label = ["free", "full"]

    cm = confusion_matrix( y_true, y_pred )

    plt.figure(figsize=(9, 9))

    ax = sns.heatmap( cm, cmap="rocket_r", fmt=".01f", annot_kws={'size': 16}, annot=True, xticklabels=label, yticklabels=label )
    ax.set_ylabel( 'Actual', fontsize=20 )
    ax.set_xlabel( 'Predicted', fontsize=20 )
    plt.savefig( os.path.join( OUTPUT_DIR, "confusion_matrix.png" ) ) 
    plt.show()


